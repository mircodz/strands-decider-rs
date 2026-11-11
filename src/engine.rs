//! The torso on the GPU, ported from the web engine (`engine/engine.js`): weights stay in one buffer
//! per shard, each forward is one compute pass over a cached per-length plan, and the recurrent
//! state, conv history and K/V can be snapshotted to continue from a shared prefix.

use std::collections::HashMap;
use std::io::Read;
use std::num::NonZeroU64;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::kernels::{self, Fmt, Gdn};
use crate::model::{self, Config, Manifest, Tensor};
use crate::{Result, err};

/// Context capacity of the KV cache and RoPE table, in tokens.
pub const CONTEXT: usize = 4096;

use wgpu::BufferUsages as U;

/// Run a wgpu future to completion; on native backends they are ready at once.
fn block_on<F: std::future::Future>(f: F) -> F::Output {
    let mut f = std::pin::pin!(f);
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    loop {
        if let std::task::Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}

/// A buffer range to bind.
#[derive(Clone)]
struct Res {
    buf: wgpu::Buffer,
    off: u64,
    size: Option<u64>,
}

impl Res {
    fn whole(buf: &wgpu::Buffer) -> Res {
        Res { buf: buf.clone(), off: 0, size: None }
    }
    /// The first `bytes` of `buf` (rounded up to whole words).
    fn slice(buf: &wgpu::Buffer, bytes: usize) -> Res {
        Res { buf: buf.clone(), off: 0, size: Some(bytes.div_ceil(4) as u64 * 4) }
    }
}

struct Step {
    label: &'static str,
    pipe: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    groups: [u32; 3],
}

struct Plan {
    steps: Vec<Step>,
    last: Step,
}

/// Activation buffers for up to `cap` tokens, shared by every plan.
struct Pool {
    cap: usize,
    ids: wgpu::Buffer,
    x: wgpu::Buffer,
    h: wgpu::Buffer,
    big: wgpu::Buffer,
    mid: wgpu::Buffer,
    qkvc: wgpu::Buffer,
    of: wgpu::Buffer,
    o: wgpu::Buffer,
    y: wgpu::Buffer,
    q: wgpu::Buffer,
    rows_all: wgpu::Buffer,
    rows_out: wgpu::Buffer,
    out: wgpu::Buffer,
}

/// The torso state after a prefix: forward the rest of a prompt from it with [`Engine::forward`].
pub struct Snapshot {
    pub len: usize,
    state: HashMap<usize, wgpu::Buffer>,
    hist: HashMap<usize, wgpu::Buffer>,
    k: HashMap<usize, wgpu::Buffer>,
    v: HashMap<usize, wgpu::Buffer>,
}

pub struct Output {
    /// Final hidden states (f32, in the head's basis) of the requested rows, `d` wide.
    pub hidden: Vec<f32>,
    pub d: usize,
    /// GPU time of the forward, when timed and the adapter supports timestamps.
    pub gpu_ms: Option<f64>,
    pub snapshot: Option<Snapshot>,
}

struct Timer {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
}

pub struct Engine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// The adapter in use, for display.
    pub adapter: String,
    /// Whether the kernels use subgroup operations (32-wide subgroups only).
    pub subgroups: bool,
    /// Whether matmuls use cooperative (simdgroup) matrices.
    pub coop: bool,
    checked: bool,
    cfg: Config,
    tensors: HashMap<String, Tensor>,
    shards: Vec<(u64, u64, wgpu::Buffer)>,
    embed: HashMap<String, wgpu::Buffer>,
    linear: Vec<usize>,
    full: Vec<usize>,
    state: HashMap<usize, wgpu::Buffer>,
    hist: HashMap<usize, wgpu::Buffer>,
    kc: HashMap<usize, wgpu::Buffer>,
    vc: HashMap<usize, wgpu::Buffer>,
    lu: wgpu::Buffer,
    rope: wgpu::Buffer,
    pipes: HashMap<String, wgpu::ComputePipeline>,
    pool: Option<Pool>,
    plans: HashMap<usize, Plan>,
    readback: Option<wgpu::Buffer>,
    timer: Option<Timer>,
    errors: Arc<Mutex<Vec<String>>>,
}

fn fmt_of(t: &Tensor) -> Fmt {
    Fmt { bits: t.bits, group: t.group, asym: t.asym }
}

/// Picks the GPU: it must have f16 shaders; discrete over integrated, and Metal or Vulkan over DX12
/// (which only has f16 with dxcompiler.dll). `WGPU_ADAPTER_NAME` narrows it by name, `WGPU_BACKEND`
/// by backend.
fn pick_adapter(instance: &wgpu::Instance) -> Result<wgpu::Adapter> {
    let want = std::env::var("WGPU_ADAPTER_NAME").ok().map(|s| s.to_lowercase());
    let mut seen = Vec::new();
    let mut ok = Vec::new();
    for a in block_on(instance.enumerate_adapters(wgpu::Backends::all())) {
        let i = a.get_info();
        seen.push(format!("{} ({})", i.name, i.backend));
        if want.as_ref().is_some_and(|w| !i.name.to_lowercase().contains(w)) {
            continue;
        }
        if a.features().contains(wgpu::Features::SHADER_F16) {
            ok.push(a);
        }
    }
    ok.sort_by_key(|a| {
        let i = a.get_info();
        let device = match i.device_type {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            wgpu::DeviceType::VirtualGpu => 2,
            wgpu::DeviceType::Other => 3,
            wgpu::DeviceType::Cpu => 4,
        };
        (device, i.backend == wgpu::Backend::Dx12)
    });
    ok.into_iter().next().ok_or_else(|| {
        err(format!(
            "no GPU with 16-bit float shaders (shader-f16) found; adapters: {}",
            if seen.is_empty() { "none".to_string() } else { seen.join(", ") }
        ))
    })
}

impl Engine {
    pub fn load(dir: &Path, m: &Manifest) -> Result<Engine> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pick_adapter(&instance)?;
        let info = adapter.get_info();
        let have = adapter.features();
        // The subgroup kernels assume exactly 32 lanes. wgpu reports a 4..64 range for every Metal
        // device, but Apple GPUs always run 32-wide.
        let width32 = (info.subgroup_min_size == 32 && info.subgroup_max_size == 32) || (info.backend == wgpu::Backend::Metal && info.name.starts_with("Apple"));
        let subgroups = have.contains(wgpu::Features::SUBGROUP) && width32 && std::env::var("STRANDS_DECIDER_SUBGROUPS").as_deref() != Ok("0");
        let mut features = wgpu::Features::SHADER_F16 | (have & wgpu::Features::TIMESTAMP_QUERY);
        if subgroups {
            features |= wgpu::Features::SUBGROUP;
        }
        // 8x8 f16 x f16 -> f32 cooperative matrices (Metal's simdgroup matrices), with room for the
        // kernel's 32 KB of workgroup memory.
        let coop = subgroups
            && have.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
            && adapter.limits().max_compute_workgroup_storage_size >= 32768
            && adapter.cooperative_matrix_properties().iter().any(|p| {
                (p.m_size, p.n_size, p.k_size) == (8, 8, 8) && p.ab_type == wgpu::CooperativeScalarType::F16 && p.cr_type == wgpu::CooperativeScalarType::F32
            })
            && std::env::var("STRANDS_DECIDER_COOP").as_deref() != Ok("0");
        if coop {
            features |= wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
        }
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: features,
            required_limits: adapter.limits(),
            // SAFETY: only the cooperative matrix feature is used, through the kernel validated in tests.
            experimental_features: if coop { unsafe { wgpu::ExperimentalFeatures::enabled() } } else { wgpu::ExperimentalFeatures::disabled() },
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| err(format!("requesting the GPU device: {e}")))?;
        let errors = Arc::new(Mutex::new(Vec::new()));
        let sink = errors.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| sink.lock().unwrap().push(e.to_string())));

        let buf = |size: usize, usage: U| device.create_buffer(&wgpu::BufferDescriptor { label: None, size: size.max(16).div_ceil(16) as u64 * 16, usage, mapped_at_creation: false });
        let mut shards = Vec::new();
        for (i, s) in m.shards.iter().enumerate() {
            let b = buf(s.bytes as usize, U::STORAGE | U::COPY_DST);
            let mut file = std::fs::File::open(dir.join(model::shard_file(i)))?;
            let mut chunk = vec![0u8; 64 << 20];
            let mut off = 0u64;
            while off < s.bytes {
                let n = chunk.len().min((s.bytes - off) as usize);
                file.read_exact(&mut chunk[..n])?;
                queue.write_buffer(&b, off, &chunk[..n.div_ceil(4) * 4]);
                queue.submit([]);
                off += n as u64;
            }
            shards.push((s.start, s.bytes, b));
        }
        let mut embed = HashMap::new();
        if let Some(e) = &m.embed {
            for (name, data) in model::embed_parts(dir, e)? {
                let b = buf(data.len(), U::STORAGE | U::COPY_DST);
                for (k, c) in data.chunks(64 << 20).enumerate() {
                    queue.write_buffer(&b, (k << 26) as u64, c);
                    queue.submit([]);
                }
                embed.insert(name, b);
            }
        }

        let c = m.config.clone();
        let layers = |kind: &str| c.layer_types.iter().enumerate().filter(|(_, k)| *k == kind).map(|(i, _)| i).collect::<Vec<_>>();
        let (linear, full) = (layers("linear_attention"), layers("full_attention"));
        if let Some(k) = c.layer_types.iter().find(|k| !["linear_attention", "full_attention", "adapter"].contains(&k.as_str())) {
            return Err(err(format!("unsupported layer type {k}")));
        }
        if let Some((name, _)) = m.tensors.iter().find(|(_, t)| t.had.is_some()) {
            return Err(err(format!("{name}: online Hadamard rotation is not supported")));
        }
        let (h, dk, dv) = (c.linear_num_key_heads, c.linear_key_head_dim, c.linear_value_head_dim);
        let su = U::STORAGE | U::COPY_SRC | U::COPY_DST;
        let state = linear.iter().map(|&i| (i, buf(h * dk * dv * 4, su))).collect();
        let hist = linear.iter().map(|&i| (i, buf(3 * (2 * h * dk + h * dv) * 2, su))).collect();
        let kv = c.num_key_value_heads * CONTEXT * c.head_dim * 2;
        let kc = full.iter().map(|&i| (i, buf(kv, su))).collect();
        let vc = full.iter().map(|&i| (i, buf(kv, su))).collect();
        let lu = buf(16, U::UNIFORM | U::COPY_DST);
        let rd = c.rotary_dim;
        let mut table = vec![0f32; CONTEXT * rd];
        for t in 0..CONTEXT {
            for i in 0..rd / 2 {
                let a = t as f64 / c.rope_theta.powf((2 * i) as f64 / rd as f64);
                table[t * rd + 2 * i] = a.cos() as f32;
                table[t * rd + 2 * i + 1] = a.sin() as f32;
            }
        }
        let rope = buf(table.len() * 4, U::STORAGE | U::COPY_DST);
        queue.write_buffer(&rope, 0, bytemuck::cast_slice(&table));

        let timer = features.contains(wgpu::Features::TIMESTAMP_QUERY).then(|| Timer {
            set: device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: 2 }),
            resolve: buf(16, U::QUERY_RESOLVE | U::COPY_SRC),
            read: buf(16, U::MAP_READ | U::COPY_DST),
        });
        let adapter = format!("{} ({}, {})", info.name, info.backend, if coop { "cooperative matrices" } else if subgroups { "subgroups" } else { "no subgroups" });
        Ok(Engine {
            device,
            queue,
            adapter,
            subgroups,
            coop,
            checked: std::env::var("STRANDS_DECIDER_CHECKED").is_ok_and(|v| v == "1"),
            cfg: c,
            tensors: m.tensors.clone(),
            shards,
            embed,
            linear,
            full,
            state,
            hist,
            kc,
            vc,
            lu,
            rope,
            pipes: HashMap::new(),
            pool: None,
            plans: HashMap::new(),
            readback: None,
            timer,
            errors,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }

    fn buf(&self, size: usize, usage: U) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: size.max(16).div_ceil(16) as u64 * 16, usage, mapped_at_creation: false })
    }

    fn uniform(&self, words: [u32; 4]) -> wgpu::Buffer {
        let b = self.buf(16, U::UNIFORM | U::COPY_DST);
        self.queue.write_buffer(&b, 0, bytemuck::cast_slice(&words));
        b
    }

    /// The binding of a named weight tensor: a range of its shard's buffer, or an embedding part.
    fn w(&self, name: &str) -> Result<Res> {
        if let Some(b) = self.embed.get(name) {
            return Ok(Res::whole(b));
        }
        let t = self.tensors.get(name).ok_or_else(|| err(format!("no tensor {name}")))?;
        let (start, _, buf) = self.shards.iter().find(|(s, n, _)| t.offset >= *s && t.offset < s + n).ok_or_else(|| err(format!("{name} is in no shard")))?;
        Ok(Res { buf: buf.clone(), off: t.offset - start, size: Some(t.bytes.div_ceil(4) * 4) })
    }

    fn pipe(&mut self, key: &str, code: impl FnOnce() -> String) -> wgpu::ComputePipeline {
        if let Some(p) = self.pipes.get(key) {
            return p.clone();
        }
        let src = if self.subgroups { code() } else { kernels::no_subgroups(&code()) };
        let desc = wgpu::ShaderModuleDescriptor { label: Some(key), source: wgpu::ShaderSource::Wgsl(src.into()) };
        // The kernels index in bounds by construction; skipping wgpu's bounds and loop checks is what
        // a browser cannot do. STRANDS_DECIDER_CHECKED=1 turns them back on.
        let checks = if self.checked { wgpu::ShaderRuntimeChecks::checked() } else { wgpu::ShaderRuntimeChecks::unchecked() };
        let module = unsafe { self.device.create_shader_module_trusted(desc, checks) };
        let p = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(key),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions { constants: &[], zero_initialize_workgroup_memory: false },
            cache: None,
        });
        self.pipes.insert(key.to_string(), p.clone());
        p
    }

    fn step(&mut self, steps: &mut Vec<Step>, label: &'static str, key: &str, code: impl FnOnce() -> String, entries: &[(u32, Res)], groups: [usize; 3]) {
        let pipe = self.pipe(key, code);
        let entries: Vec<wgpu::BindGroupEntry> = entries
            .iter()
            .map(|(b, r)| wgpu::BindGroupEntry {
                binding: *b,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &r.buf, offset: r.off, size: r.size.and_then(NonZeroU64::new) }),
            })
            .collect();
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(key), layout: &pipe.get_bind_group_layout(0), entries: &entries });
        steps.push(Step { label, pipe, bind, groups: groups.map(|g| g as u32) });
    }

    fn ensure_pool(&mut self, l: usize, max_n: usize) {
        let cap = if l <= 128 { 128 } else { l.next_power_of_two() };
        if self.pool.as_ref().is_some_and(|p| p.cap >= cap) {
            return;
        }
        // A bigger pool serves every shorter length too; plans bound to the old one are rebuilt.
        self.plans.clear();
        let c = &self.cfg;
        let (d, i, h, dk, dv) = (c.hidden_size, c.intermediate_size, c.linear_num_key_heads, c.linear_key_head_dim, c.linear_value_head_dim);
        let s = U::STORAGE;
        let pool = Pool {
            cap,
            ids: self.buf(cap * 4, s | U::COPY_DST),
            x: self.buf(cap * d * 4, s),
            h: self.buf(cap * d * 2, s),
            big: self.buf(cap * max_n * 2, s),
            mid: self.buf(cap * i * 2, s),
            qkvc: self.buf(cap * (2 * h * dk + h * dv) * 4, s),
            of: self.buf(cap * h * dv * 4, s),
            o: self.buf(cap * d * 2, s),
            y: self.buf(cap * d * 2, s),
            q: self.buf(c.num_attention_heads * cap * c.head_dim * 2, s),
            rows_all: self.buf(cap * 4, s | U::COPY_DST),
            rows_out: self.buf(cap * 4, s | U::COPY_DST),
            out: self.buf(cap * d * 4, s | U::COPY_SRC),
        };
        let identity: Vec<u32> = (0..cap as u32).collect();
        self.queue.write_buffer(&pool.rows_all, 0, bytemuck::cast_slice(&identity));
        self.pool = Some(pool);
    }

    /// Builds (once per length) the dispatch list for `l` tokens.
    fn plan(&mut self, l: usize) -> Result<()> {
        if self.plans.contains_key(&l) {
            return Ok(());
        }
        let c = self.cfg.clone();
        let (d, i_size, h, dk, dv) = (c.hidden_size, c.intermediate_size, c.linear_num_key_heads, c.linear_key_head_dim, c.linear_value_head_dim);
        let (hq, hk, hd) = (c.num_attention_heads, c.num_key_value_heads, c.head_dim);
        let n_of = |name: String| self.tensors.get(&name).map_or(0, |t| t.n);
        let in_n = self.linear.first().map_or(0, |i| n_of(format!("layers.{i}.in_proj.q")));
        let qkv_n = self.full.first().map_or(0, |i| n_of(format!("layers.{i}.qkv.q")));
        self.ensure_pool(l, (2 * i_size).max(in_n).max(qkv_n));
        let p = self.pool.as_ref().unwrap();
        let (ids, x, hb, big, mid, qkvc, of, o, y, q, rows_all, rows_out, out) =
            (p.ids.clone(), p.x.clone(), p.h.clone(), p.big.clone(), p.mid.clone(), p.qkvc.clone(), p.of.clone(), p.o.clone(), p.y.clone(), p.q.clone(), p.rows_all.clone(), p.rows_out.clone(), p.out.clone());
        let lu = Res::whole(&self.lu);
        let mut steps = Vec::new();
        let mut uniforms: HashMap<(usize, usize), wgpu::Buffer> = HashMap::new();

        // Matmul kernel by M, measured on an M4 Pro. With cooperative matrices: 16-row tiles up to 80
        // rows, then 32-row tiles with one quant block per stage. Without (the web engine's table): the
        // multi-row GEMV up to 20 rows, then one exact-height tile up to 128 rows, two up to 256, 64-row
        // tiles, 128-row ones past 900.
        let coop_tile = self.coop.then_some(if l <= 80 { (16, 2) } else { (32, 1) });
        let tile = match l {
            0..=20 => None,
            21..=128 => Some(l.div_ceil(16) * 16),
            129..=256 => Some(l.div_ceil(32) * 16),
            257..=900 => Some(64),
            _ => Some(128),
        };
        let mut matmul = |e: &mut Self, steps: &mut Vec<Step>, label: &'static str, a: &wgpu::Buffer, name: &str, yb: &wgpu::Buffer| -> Result<usize> {
            let t = e.tensors.get(&format!("{name}.q")).ok_or_else(|| err(format!("no tensor {name}.q")))?.clone();
            let f = fmt_of(&t);
            let p = uniforms.entry((t.n, t.k)).or_insert_with(|| e.uniform([l as u32, t.n as u32, t.k as u32, (t.k / 32) as u32])).clone();
            let mut entries = vec![(0, Res::slice(a, l * t.k * 2)), (1, e.w(&format!("{name}.q"))?), (2, e.w(&format!("{name}.s"))?), (3, Res::slice(yb, l * t.n * 2)), (4, Res::whole(&p))];
            if f.asym {
                entries.push((6, e.w(&format!("{name}.b"))?));
            }
            match (coop_tile, tile) {
                (Some((tm, kb)), _) if f.key().is_empty() && t.n % 64 == 0 => {
                    e.step(steps, label, &format!("coop{tm}x{kb}"), || kernels::matmul_coop(tm, kb), &entries, [t.n / 64, l.div_ceil(tm), 1])
                }
                (_, Some(tm)) if f.key().is_empty() && t.n % 64 == 0 => {
                    e.step(steps, label, &format!("tiled{tm}"), || kernels::matmul_tiled(tm), &entries, [t.n / 64, l.div_ceil(tm), 1])
                }
                _ => e.step(steps, label, &format!("gemv{}", f.key()), || kernels::matmul_gemv(f), &entries, [t.n / 16, l.div_ceil(8), 1]),
            }
            Ok(t.n)
        };

        let ef = fmt_of(self.tensors.get("embed.q").ok_or_else(|| err("no tensor embed.q"))?);
        let mut e = vec![(0, Res::whole(&ids)), (1, self.w("embed.q")?), (2, self.w("embed.s")?), (3, Res::whole(&x))];
        if ef.asym {
            e.push((4, self.w("embed.b")?));
        }
        self.step(&mut steps, "embed", &format!("embed{}", ef.key()), || kernels::embed(d, ef), &e, [l, 1, 1]);
        let add_norm = |e: &mut Self, steps: &mut Vec<Step>, add: bool, w: &str| -> Result<()> {
            let entries = [(0, Res::whole(&x)), (1, Res::whole(&y)), (2, e.w(w)?), (3, Res::whole(&hb)), (4, Res::whole(&rows_all))];
            e.step(steps, "norm", &format!("addnorm{add}"), || kernels::add_norm(d, add, false), &entries, [l, 1, 1]);
            Ok(())
        };
        // big elementwise dispatches: x up to 32768 workgroups, the rest in y
        let grid = |n: usize| {
            let groups = n.div_ceil(256);
            let gx = groups.min(32768);
            [gx, groups.div_ceil(gx), 1]
        };
        for (i, kind) in c.layer_types.iter().enumerate() {
            let pre = format!("layers.{i}.");
            add_norm(self, &mut steps, i > 0, &format!("{pre}in_norm"))?;
            if kind == "adapter" {
                // the linear stand-in for removed layers: x += proj(rms_norm(x)) + bias
                matmul(self, &mut steps, "proj", &hb, &format!("{pre}proj"), &y)?;
                let entries = [(0, Res::slice(&y, l * d * 2)), (1, self.w(&format!("{pre}bias"))?)];
                self.step(&mut steps, "bias", &format!("bias{d}"), || kernels::add_bias(d), &entries, grid(l * d));
                continue;
            }
            if kind == "linear_attention" {
                let n = matmul(self, &mut steps, "in_proj", &hb, &format!("{pre}in_proj"), &big)?;
                let cc = 2 * h * dk + h * dv;
                let entries = [(0, Res::whole(&big)), (1, self.w(&format!("{pre}conv"))?), (2, Res::whole(&qkvc)), (3, Res::whole(&self.hist[&i]))];
                self.step(&mut steps, "conv", &format!("conv{n}"), || kernels::conv(cc, n), &entries, [cc.div_ceil(256), l, 1]);
                let entries = [(0, Res::whole(&big)), (1, Res::whole(&self.hist[&i])), (2, lu.clone())];
                self.step(&mut steps, "hist", &format!("hist{n}"), || kernels::hist_update(cc, n), &entries, [cc.div_ceil(256), 1, 1]);
                let g = Gdn { h, dk, dv, stride: n, a_off: cc + h * dv, b_off: cc + h * dv + h };
                let sub = self.subgroups;
                let entries = [
                    (0, Res::whole(&qkvc)),
                    (1, Res::whole(&big)),
                    (2, self.w(&format!("{pre}neg_exp_A"))?),
                    (3, self.w(&format!("{pre}dt_bias"))?),
                    (4, Res::whole(&of)),
                    (5, lu.clone()),
                    (6, Res::whole(&self.state[&i])),
                ];
                self.step(&mut steps, "gdn", &format!("gdn{n}"), || if sub { g.subgroups() } else { g.no_subgroups() }, &entries, [h * dv / 32, 1, 1]);
                let entries = [(0, Res::whole(&of)), (1, Res::whole(&big)), (2, self.w(&format!("{pre}gnorm"))?), (3, Res::whole(&o))];
                self.step(&mut steps, "gnorm", &format!("gnorm{n}"), || kernels::gnorm(h, dv, n, cc), &entries, [l * h / 4, 1, 1]);
                matmul(self, &mut steps, "out_proj", &o, &format!("{pre}out_proj"), &y)?;
            } else {
                let n = matmul(self, &mut steps, "qkv", &hb, &format!("{pre}qkv"), &big)?;
                let (kb, vb) = (self.kc[&i].clone(), self.vc[&i].clone());
                let entries = [
                    (0, Res::whole(&big)),
                    (1, self.w(&format!("{pre}q_norm"))?),
                    (2, self.w(&format!("{pre}k_norm"))?),
                    (3, Res::whole(&self.rope)),
                    (4, Res::whole(&q)),
                    (5, Res::whole(&kb)),
                    (6, lu.clone()),
                    (7, Res::whole(&vb)),
                ];
                self.step(&mut steps, "qk_prep", &format!("qkprep{n}"), || kernels::qk_prep(hq, hk, hd, c.rotary_dim, n), &entries, [l, hq + hk, 1]);
                let entries = [(0, Res::whole(&q)), (1, Res::whole(&kb)), (2, Res::whole(&big)), (3, Res::whole(&o)), (4, lu.clone()), (5, Res::whole(&vb))];
                self.step(&mut steps, "attn", &format!("attn{n}"), || kernels::attn(hq, hk, hd, n), &entries, [l.div_ceil(16), hq, 1]);
                matmul(self, &mut steps, "o_proj", &o, &format!("{pre}o_proj"), &y)?;
            }
            add_norm(self, &mut steps, true, &format!("{pre}post_norm"))?;
            matmul(self, &mut steps, "gate_up", &hb, &format!("{pre}gate_up"), &big)?;
            let entries = [(0, Res::slice(&big, l * 2 * i_size * 2)), (1, Res::slice(&mid, l * i_size * 2))];
            self.step(&mut steps, "silu", "silu", || kernels::silu_mul(i_size), &entries, grid(l * i_size));
            matmul(self, &mut steps, "down", &mid, &format!("{pre}down"), &y)?;
        }
        let mut last = Vec::new();
        let entries = [(0, Res::whole(&x)), (1, Res::whole(&y)), (2, self.w("final_norm")?), (3, Res::whole(&out)), (4, Res::whole(&rows_out))];
        self.step(&mut last, "final_norm", "finalnorm", || kernels::add_norm(d, true, true), &entries, [0, 1, 1]);
        self.plans.insert(l, Plan { steps, last: last.pop().unwrap() });
        Ok(())
    }

    /// Runs the torso on `ids`, continuing from `cache` (a prefix snapshot) when given. Returns the
    /// final hidden states of `rows` (indices into `ids`) and, with `keep`, a snapshot covering the
    /// cached prefix and `ids`.
    pub fn forward(&mut self, ids: &[u32], rows: &[usize], cache: Option<&Snapshot>, keep: bool, timed: bool) -> Result<Output> {
        let l = ids.len();
        let p = cache.map_or(0, |c| c.len);
        if l == 0 || p + l > CONTEXT {
            return Err(err(format!("context of {} tokens: must be 1 to {CONTEXT}", p + l)));
        }
        if let Some(r) = rows.iter().find(|&&r| r >= l) {
            return Err(err(format!("row {r} out of range for {l} tokens")));
        }
        self.plan(l)?;
        let (d, hk, row) = (self.cfg.hidden_size, self.cfg.num_key_value_heads, self.cfg.head_dim * 2);
        let pool = self.pool.as_ref().unwrap();
        self.queue.write_buffer(&pool.ids, 0, bytemuck::cast_slice(ids));
        let out_rows: Vec<u32> = if rows.is_empty() { vec![0] } else { rows.iter().map(|&r| r as u32).collect() };
        self.queue.write_buffer(&pool.rows_out, 0, bytemuck::cast_slice(&out_rows));
        self.queue.write_buffer(&self.lu, 0, bytemuck::cast_slice(&[l as u32, p as u32, CONTEXT as u32, cache.is_some() as u32]));
        let out_bytes = rows.len() * d * 4;
        if self.readback.as_ref().is_none_or(|b| (b.size() as usize) < out_bytes) {
            self.readback = Some(self.buf(out_bytes, U::MAP_READ | U::COPY_DST));
        }
        let (pool, plan, rb) = (self.pool.as_ref().unwrap(), &self.plans[&l], self.readback.as_ref().unwrap());
        let mut enc = self.device.create_command_encoder(&Default::default());
        let head_stride = (CONTEXT * row) as u64;
        match cache {
            Some(c) => {
                for i in &self.linear {
                    enc.copy_buffer_to_buffer(&c.state[i], 0, &self.state[i], 0, None);
                    enc.copy_buffer_to_buffer(&c.hist[i], 0, &self.hist[i], 0, None);
                }
                for i in &self.full {
                    for (src, dst) in [(&c.k[i], &self.kc[i]), (&c.v[i], &self.vc[i])] {
                        for h in 0..hk as u64 {
                            enc.copy_buffer_to_buffer(src, h * (p * row) as u64, dst, h * head_stride, Some((p * row) as u64));
                        }
                    }
                }
            }
            None => {
                for b in self.hist.values() {
                    enc.clear_buffer(b, 0, None);
                }
            }
        }
        let timer = self.timer.as_ref().filter(|_| timed);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: timer.map(|t| wgpu::ComputePassTimestampWrites { query_set: &t.set, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) }),
            });
            for s in plan.steps.iter().chain(if rows.is_empty() { None } else { Some(&plan.last) }) {
                pass.set_pipeline(&s.pipe);
                pass.set_bind_group(0, &s.bind, &[]);
                let x = if std::ptr::eq(s, &plan.last) { rows.len() as u32 } else { s.groups[0] };
                pass.dispatch_workgroups(x, s.groups[1], s.groups[2]);
            }
        }
        if out_bytes > 0 {
            enc.copy_buffer_to_buffer(&pool.out, 0, rb, 0, Some(out_bytes as u64));
        }
        if let Some(t) = timer {
            enc.resolve_query_set(&t.set, 0..2, &t.resolve, 0);
            enc.copy_buffer_to_buffer(&t.resolve, 0, &t.read, 0, Some(16));
        }
        let snapshot = keep.then(|| {
            let t = p + l;
            let copy = |enc: &mut wgpu::CommandEncoder, src: &wgpu::Buffer| {
                let b = self.buf(src.size() as usize, U::COPY_SRC | U::COPY_DST);
                enc.copy_buffer_to_buffer(src, 0, &b, 0, None);
                b
            };
            let mut s = Snapshot { len: t, state: HashMap::new(), hist: HashMap::new(), k: HashMap::new(), v: HashMap::new() };
            for &i in &self.linear {
                s.state.insert(i, copy(&mut enc, &self.state[&i]));
                s.hist.insert(i, copy(&mut enc, &self.hist[&i]));
            }
            for &i in &self.full {
                for (src, dst) in [(&self.kc[&i], &mut s.k), (&self.vc[&i], &mut s.v)] {
                    let b = self.buf(hk * t * row, U::COPY_SRC | U::COPY_DST);
                    for h in 0..hk as u64 {
                        enc.copy_buffer_to_buffer(src, h * head_stride, &b, h * (t * row) as u64, Some((t * row) as u64));
                    }
                    dst.insert(i, b);
                }
            }
            s
        });
        self.queue.submit([enc.finish()]);

        let mut maps = Vec::new();
        if out_bytes > 0 {
            maps.push(rb.slice(..out_bytes as u64));
        }
        if let Some(t) = timer {
            maps.push(t.read.slice(..));
        }
        let (tx, rx) = std::sync::mpsc::channel();
        for m in &maps {
            let tx = tx.clone();
            m.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        }
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| err(format!("GPU: {e}")))?;
        for _ in &maps {
            rx.recv().unwrap().map_err(|e| err(format!("GPU readback: {e}")))?;
        }
        let mapped = |s: &wgpu::BufferSlice| s.get_mapped_range().map_err(|e| err(format!("GPU readback: {e:?}")));
        let mut hidden: Vec<f32> = if out_bytes > 0 { bytemuck::pod_collect_to_vec(&mapped(&maps[0])?) } else { Vec::new() };
        let gpu_ms = match timer {
            Some(t) => {
                let ts: Vec<u64> = bytemuck::pod_collect_to_vec(&mapped(&t.read.slice(..))?);
                Some(ts[1].saturating_sub(ts[0]) as f64 * self.queue.get_timestamp_period() as f64 / 1e6)
            }
            None => None,
        };
        if out_bytes > 0 {
            rb.unmap();
        }
        if let Some(t) = timer {
            t.read.unmap();
        }
        let errors = std::mem::take(&mut *self.errors.lock().unwrap());
        if !errors.is_empty() {
            return Err(err(format!("GPU error: {}", errors.join("\n"))));
        }
        // A rotated torso: final_norm is 1, and the head's basis is unrotate * FWHT(row).
        if let Some(u) = &self.cfg.unrotate {
            for r in hidden.chunks_exact_mut(d) {
                fwht(r);
                for (x, c) in r.iter_mut().zip(u) {
                    *x = (*x as f64 * c) as f32;
                }
            }
        }
        Ok(Output { hidden, d, gpu_ms, snapshot })
    }
}

impl Engine {
    /// GPU time per kernel kind (ms, slowest first) for a forward over `ids`, with one timestamped
    /// pass per dispatch. Diagnostics only: the separate passes add a little time of their own.
    pub fn profile(&mut self, ids: &[u32]) -> Result<Vec<(&'static str, f64)>> {
        let l = ids.len();
        if self.timer.is_none() || l == 0 || l > CONTEXT {
            return Err(err("profiling needs timestamp queries and 1 to 4096 tokens"));
        }
        self.plan(l)?;
        let pool = self.pool.as_ref().unwrap();
        self.queue.write_buffer(&pool.ids, 0, bytemuck::cast_slice(ids));
        self.queue.write_buffer(&self.lu, 0, bytemuck::cast_slice(&[l as u32, 0, CONTEXT as u32, 0]));
        let steps = &self.plans[&l].steps;
        let n = steps.len() as u32;
        let set = self.device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: 2 * n });
        let resolve = self.buf(16 * n as usize, U::QUERY_RESOLVE | U::COPY_SRC);
        let read = self.buf(16 * n as usize, U::MAP_READ | U::COPY_DST);
        let mut enc = self.device.create_command_encoder(&Default::default());
        for b in self.hist.values() {
            enc.clear_buffer(b, 0, None);
        }
        for (i, s) in steps.iter().enumerate() {
            let w = wgpu::ComputePassTimestampWrites { query_set: &set, beginning_of_pass_write_index: Some(2 * i as u32), end_of_pass_write_index: Some(2 * i as u32 + 1) };
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: Some(w) });
            pass.set_pipeline(&s.pipe);
            pass.set_bind_group(0, &s.bind, &[]);
            pass.dispatch_workgroups(s.groups[0], s.groups[1], s.groups[2]);
        }
        enc.resolve_query_set(&set, 0..2 * n, &resolve, 0);
        enc.copy_buffer_to_buffer(&resolve, 0, &read, 0, None);
        self.queue.submit([enc.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| err(format!("GPU: {e}")))?;
        let ts: Vec<u64> = bytemuck::pod_collect_to_vec(&read.slice(..).get_mapped_range().map_err(|e| err(format!("{e:?}")))?);
        let period = self.queue.get_timestamp_period() as f64 / 1e6;
        let mut by: Vec<(&'static str, f64)> = Vec::new();
        for (i, s) in steps.iter().enumerate() {
            let ms = ts[2 * i + 1].saturating_sub(ts[2 * i]) as f64 * period;
            match by.iter_mut().find(|(k, _)| *k == s.label) {
                Some(e) => e.1 += ms,
                None => by.push((s.label, ms)),
            }
        }
        by.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(by)
    }
}

/// In-place fast Walsh-Hadamard transform (Sylvester order, unnormalized).
fn fwht(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
}
