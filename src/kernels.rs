//! WGSL kernels, ported from the web engine (`engine/wgsl.js`, `gemmbench/kernels.js`). Batch 1, L
//! tokens; residual stream f32, matmul inputs and outputs f16, recurrent math f32.

use std::fmt::Display;

/// A matmul weight format: `bits` per value in blocks of 32, one fp16 scale per `group` values,
/// symmetric (`(q - 2^(bits-1)) * scale`) or with a per-group fp16 bias (`q * scale + bias`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Fmt {
    pub bits: u32,
    pub group: u32,
    pub asym: bool,
}

impl Fmt {
    /// Key suffix for a pipeline; empty for the default int4, group 32, symmetric.
    pub fn key(self) -> String {
        if self.bits == 4 && self.group == 32 && !self.asym { String::new() } else { format!("_{}g{}{}", self.bits, self.group, if self.asym { "a" } else { "" }) }
    }
}

/// `src` with each `${NAME}` replaced by its value.
fn fill(src: &str, vars: &[(&str, &dyn Display)]) -> String {
    let mut s = src.to_string();
    for (k, v) in vars {
        s = s.replace(&format!("${{{k}}}"), &v.to_string());
    }
    assert!(!s.contains("${"), "unfilled template variable in kernel");
    s
}

// No `enable subgroups;`: naga has subgroup builtins without it (and rejects the directive).
const HDR: &str = "enable f16;\n";

/// A kernel without subgroup operations, for GPUs whose subgroups are not exactly 32 wide: `sg` and
/// `lane` become aligned 32-thread groups of the local invocation index, and subgroupAdd/subgroupMax a
/// sum or max through workgroup memory (every call site is in uniform control flow).
pub fn no_subgroups(code: &str) -> String {
    if !code.contains("@builtin(subgroup_id)") {
        return code.to_string();
    }
    let mut code = code.to_string();
    let li = code.split("@builtin(local_invocation_index) ").nth(1).map(|s| s.split(':').next().unwrap().trim().to_string());
    let params = ", @builtin(subgroup_id) sg: u32, @builtin(subgroup_invocation_id) lane: u32";
    code = code.replace(params, if li.is_some() { "" } else { ", @builtin(local_invocation_index) li32: u32" });
    let li = li.unwrap_or("li32".into());
    let body = code.find("fn main(").and_then(|i| code[i..].find('{').map(|j| i + j + 1)).expect("kernel without main");
    code.insert_str(body, &format!("\n  let sg = {li} / 32u; let lane = {li} % 32u;"));
    code = code.replace("subgroupAdd(", &format!("sum32({li}, ")).replace("subgroupMax(", &format!("max32({li}, "));
    let builtins = ["Add", "Max", "Min", "Mul", "And", "Or", "Xor", "Shuffle", "Broadcast", "Ballot", "Elect", "All", "Any", "_id", "_invocation_id", "_size"];
    assert!(!builtins.iter().any(|b| code.contains(&format!("subgroup{b}"))), "no_subgroups: unhandled subgroup use");
    let nt: u32 = code.split("@workgroup_size(").nth(1).and_then(|s| s.split(')').next()).and_then(|s| s.parse().ok()).expect("workgroup size");
    let fold = |name: &str, op: &str| {
        format!(
            "fn {name}(li: u32, x: f32) -> f32 {{\n  workgroupBarrier(); red32[li] = x; workgroupBarrier();\n  let b = li & ~31u; var r = red32[b]; for (var i = 1u; i < 32u; i++) {{ r = {op}; }} return r;\n}}\n"
        )
    };
    let helpers = format!("var<workgroup> red32: array<f32, {nt}>;\n{}{}", fold("sum32", "r + red32[b + i]"), fold("max32", "max(r, red32[b + i])"));
    code.replacen("\n@compute", &format!("\n{helpers}@compute"), 1)
}

/// ids -> x (f32 [L, D]) from the quantized embedding table. One thread per 8 elements.
pub fn embed(d: usize, f: Fmt) -> String {
    let g = if f.group == 32 { "blk".to_string() } else { format!("blk / {}u", f.group / 32) };
    let q = if f.bits == 4 {
        "(Q[blk * 4u + e / 8u] >> (4u * (e % 8u))) & 15u".to_string()
    } else {
        let hi = if f.bits == 3 { " | (((Q[blk * 3u + 2u] >> e) & 1u) << 2u)" } else { "" };
        format!("((Q[blk * {}u + e / 16u] >> (2u * (e % 16u))) & 3u){hi}", f.bits)
    };
    let value = if f.asym { format!("f32({q}) * sc + bi") } else { format!("(f32({q}) - {}.0) * sc", 1 << (f.bits - 1)) };
    let src = r#"
@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> Q: array<u32>;
@group(0) @binding(2) var<storage, read> S: array<f16>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
${BI}
@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_index) w: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let t = wg.x; let id = ids[t];
  let blk = id * ${NB}u + w / 4u; let sc = f32(S[${G}]);${BIV}
  for (var j = 0u; j < 8u; j++) {
    let e = (w % 4u) * 8u + j;
    x[t * ${D}u + w * 8u + j] = ${VALUE};
  }
}"#;
    let (bi, biv) = if f.asym { ("@group(0) @binding(4) var<storage, read> Bi: array<f16>;", format!(" let bi = f32(Bi[{g}]);")) } else { ("", String::new()) };
    HDR.to_string() + &fill(src, &[("BI", &bi), ("WG", &(d / 8)), ("NB", &(d / 32)), ("G", &g), ("BIV", &biv), ("D", &d), ("VALUE", &value)])
}

/// Optional residual add (x += y), then RMSNorm with the (1 + w) weight -> h (f16, or f32 for the
/// final norm, which gathers `rows`). One workgroup per row.
pub fn add_norm(d: usize, add: bool, out_f32: bool) -> String {
    let src = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f16>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> h: array<${OUT}>;
@group(0) @binding(4) var<storage, read> rows: array<u32>;
var<workgroup> red: array<f32, 8>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_id) sg: u32, @builtin(subgroup_invocation_id) lane: u32) {
  let r = rows[wg.x]; let o = r * ${D}u;
  var v: array<f32, ${PER}>; var ss = 0.0;
  for (var i = 0u; i < ${PER}u; i++) {
    let c = li + 256u * i;
    var a = x[o + c];
    ${ADD}
    v[i] = a; ss += a * a;
  }
  ss = subgroupAdd(ss);
  if (lane == 0u) { red[sg] = ss; }
  workgroupBarrier();
  var tot = 0.0; for (var i = 0u; i < 8u; i++) { tot += red[i]; }
  let inv = inverseSqrt(tot / ${D}.0 + 1e-6);
  for (var i = 0u; i < ${PER}u; i++) { let c = li + 256u * i; h[wg.x * ${D}u + c] = ${CAST}(v[i] * inv * w[c]); }
}"#;
    let add = if add { "a += f32(y[o + c]); x[o + c] = a;" } else { "if (false) { a += f32(y[0]); }" };
    HDR.to_string() + &fill(src, &[("OUT", &if out_f32 { "f32" } else { "f16" }), ("D", &d), ("PER", &(d / 256)), ("ADD", &add), ("CAST", &if out_f32 { "" } else { "f16" })])
}

/// mid[t, j] = silu(g[t, j]) * u[t, j] where big[t] = [g (I) | u (I)].
pub fn silu_mul(i: usize) -> String {
    let src = r#"
@group(0) @binding(0) var<storage, read> big: array<f16>;
@group(0) @binding(1) var<storage, read_write> mid: array<f16>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let i = g.x + g.y * nw.x * 256u; let t = i / ${I}u; let j = i % ${I}u;
  if (t >= arrayLength(&mid) / ${I}u) { return; }
  let gv = f32(big[t * ${I2}u + j]); let uv = f32(big[t * ${I2}u + ${I}u + j]);
  mid[i] = f16(gv / (1.0 + exp(-gv)) * uv);
}"#;
    HDR.to_string() + &fill(src, &[("I", &i), ("I2", &(2 * i))])
}

/// y[t, j] += c[j] (the f32 bias of an adapter layer).
pub fn add_bias(n: usize) -> String {
    let src = r#"
@group(0) @binding(0) var<storage, read_write> y: array<f16>;
@group(0) @binding(1) var<storage, read> c: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let i = g.x + g.y * nw.x * 256u;
  if (i >= arrayLength(&y)) { return; }
  y[i] = f16(f32(y[i]) + c[i % ${N}u]);
}"#;
    HDR.to_string() + &fill(src, &[("N", &n)])
}

/// Depthwise causal conv (kernel 4) + SiLU over the qkv channels of the fused in_proj output.
pub fn conv(c: usize, stride: usize) -> String {
    let src = r#"
@group(0) @binding(0) var<storage, read> big: array<f16>;
@group(0) @binding(1) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> hist: array<f16>;     // [3, C] pre-conv rows before t = 0 (zeros if none)
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let c = g.x; let t = g.y;
  if (c >= ${C}u) { return; }
  let k = w[c]; var s = 0.0;
  for (var j = 0u; j < 4u; j++) {
    let tt = i32(t) + i32(j) - 3;
    if (tt >= 0) { s += k[j] * f32(big[u32(tt) * ${STRIDE}u + c]); } else { s += k[j] * f32(hist[u32(tt + 3) * ${C}u + c]); }
  }
  out[t * ${C}u + c] = s / (1.0 + exp(-s));
}"#;
    HDR.to_string() + &fill(src, &[("C", &c), ("STRIDE", &stride)])
}

/// After the conv: hist <- the last 3 pre-conv rows of (old hist ++ this call's rows).
pub fn hist_update(c: usize, stride: usize) -> String {
    let src = r#"
@group(0) @binding(0) var<storage, read> big: array<f16>;
@group(0) @binding(1) var<storage, read_write> hist: array<f16>;
@group(0) @binding(2) var<uniform> Lu: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let c = g.x; if (c >= ${C}u) { return; }
  let L = i32(Lu.x);
  var v: array<f16, 3>;
  for (var r = 0; r < 3; r++) {
    let src = L - 3 + r;                       // row index in this call; < 0 means older history
    if (src >= 0) { v[r] = big[u32(src) * ${STRIDE}u + c]; } else { v[r] = hist[u32(src + 3) * ${C}u + c]; }
  }
  for (var r = 0u; r < 3u; r++) { hist[r * ${C}u + c] = v[r]; }
}"#;
    HDR.to_string() + &fill(src, &[("C", &c), ("STRIDE", &stride)])
}

/// Shapes of a gated-delta layer and where its a/b gates sit in the fused in_proj row.
#[derive(Clone, Copy)]
pub struct Gdn {
    pub h: usize,
    pub dk: usize,
    pub dv: usize,
    pub stride: usize,
    pub a_off: usize,
    pub b_off: usize,
}

impl Gdn {
    fn vars(&self) -> Vec<(&'static str, String)> {
        let (h, dk, dv) = (self.h, self.dk, self.dv);
        vec![
            ("C", (2 * h * dk + h * dv).to_string()),
            ("DV32", (dv / 32).to_string()),
            ("DKDV", (dk * dv).to_string()),
            ("DV", dv.to_string()),
            ("DV2", (2 * dv).to_string()),
            ("DV3", (3 * dv).to_string()),
            ("DK", dk.to_string()),
            ("HDK", (h * dk).to_string()),
            ("HDK2", (2 * h * dk).to_string()),
            ("HDV", (h * dv).to_string()),
            ("STRIDE", self.stride.to_string()),
            ("AOFF", self.a_off.to_string()),
            ("BOFF", self.b_off.to_string()),
        ]
    }

    fn fill(&self, src: &str) -> String {
        let vars = self.vars();
        let refs: Vec<(&str, &dyn Display)> = vars.iter().map(|(k, v)| (*k, v as &dyn Display)).collect();
        HDR.to_string() + &fill(src, &refs)
    }

    /// Gated delta rule, sequential over tokens. WG = 4 subgroups for one head and 32 value columns;
    /// each subgroup owns 8 value columns, lane l owns key dims 4l..4l+3, so the 128x8 state slice
    /// lives in 32 registers per lane and every reduction over key dims is a subgroupAdd.
    /// Statements for each of the 8 value columns c (WGSL `{c}` in `line`), unrolled: naga turns
    /// loops into `while (true)`, which Metal does not unroll, and the state would leave registers.
    fn columns(line: &str) -> String {
        (0..8).map(|c| line.replace("{c}", &c.to_string())).collect::<Vec<_>>().join("\n    ")
    }

    fn state_io(&self) -> Vec<(&'static str, String)> {
        vec![
            ("DECLARE", Self::columns("var S{c} = vec4<f32>(0.0);")),
            ("LOAD", Self::columns("S{c} = vec4<f32>(state[sb + {c}u], state[sb + ${DV}u + {c}u], state[sb + ${DV2}u + {c}u], state[sb + ${DV3}u + {c}u]);")),
            ("STORE", Self::columns("state[sb + {c}u] = S{c}.x; state[sb + ${DV}u + {c}u] = S{c}.y; state[sb + ${DV2}u + {c}u] = S{c}.z; state[sb + ${DV3}u + {c}u] = S{c}.w;")),
        ]
    }

    fn fill_with(&self, src: &str, extra: &[(&'static str, String)]) -> String {
        let mut src = src.to_string();
        for (k, v) in self.state_io().iter().chain(extra) {
            src = src.replace(&format!("${{{k}}}"), v);
        }
        self.fill(&src)
    }

    /// Gated delta rule, sequential over tokens. WG = 4 subgroups for one head and 32 value columns;
    /// each subgroup owns 8 value columns, lane l owns key dims 4l..4l+3, so the 128x8 state slice
    /// lives in 32 registers per lane and every reduction over key dims is a subgroupAdd.
    pub fn subgroups(&self) -> String {
        let update = Self::columns(
            "S{c} *= decay; { let kv = subgroupAdd(dot(S{c}, kn)); let delta = (qkv[vb + {c}u] - kv) * beta; S{c} += kn * delta; let oc = subgroupAdd(dot(S{c}, qn)); if (lane == {c}u) { mine = oc; } }",
        );
        self.fill_with(
            r#"
@group(0) @binding(0) var<storage, read> qkv: array<f32>;       // [L, 2*H*DK + H*DV] after conv+silu
@group(0) @binding(1) var<storage, read> big: array<f16>;       // fused in_proj output (for a, b)
@group(0) @binding(2) var<storage, read> negA: array<f32>;
@group(0) @binding(3) var<storage, read> dtb: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;   // [L, H*DV]
@group(0) @binding(5) var<uniform> Lu: vec4<u32>;           // x = L, y = P, z = LT, w = 1 if state holds a prefix state
@group(0) @binding(6) var<storage, read_write> state: array<f32>;   // [H][DK][DV]: prefix state in, final state out
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_id) sg: u32, @builtin(subgroup_invocation_id) lane: u32) {
  let h = wg.x / ${DV32}u; let dv0 = (wg.x % ${DV32}u) * 32u + sg * 8u;
  let C = ${C}u;
  // S{c} = state[4*lane .. 4*lane+3][dv0 + c]
  ${DECLARE}
  let sb = h * ${DKDV}u + lane * 4u * ${DV}u + dv0;
  if (Lu.w == 1u) {
    ${LOAD}
  }
  let na = negA[h]; let db = dtb[h];
  let qs = 1.0 / sqrt(${DK}.0);
  for (var t = 0u; t < Lu.x; t++) {
    let base = t * C;
    let q4 = vec4<f32>(qkv[base + h * ${DK}u + lane * 4u], qkv[base + h * ${DK}u + lane * 4u + 1u], qkv[base + h * ${DK}u + lane * 4u + 2u], qkv[base + h * ${DK}u + lane * 4u + 3u]);
    let kb = base + ${HDK}u + h * ${DK}u + lane * 4u;
    let k4 = vec4<f32>(qkv[kb], qkv[kb + 1u], qkv[kb + 2u], qkv[kb + 3u]);
    let qn = q4 * inverseSqrt(subgroupAdd(dot(q4, q4)) + 1e-6) * qs;
    let kn = k4 * inverseSqrt(subgroupAdd(dot(k4, k4)) + 1e-6);
    let a = f32(big[t * ${STRIDE}u + ${AOFF}u + h]); let b = f32(big[t * ${STRIDE}u + ${BOFF}u + h]);
    let beta = 1.0 / (1.0 + exp(-b));
    let x = a + db; let sp = select(log(1.0 + exp(x)), x, x > 20.0);
    let decay = exp(na * sp);
    let vb = base + ${HDK2}u + h * ${DV}u + dv0;
    var mine = 0.0;
    ${UPDATE}
    if (lane < 8u) { o[t * ${HDV}u + h * ${DV}u + dv0 + lane] = mine; }
  }
  ${STORE}
}"#,
            &[("UPDATE", update)],
        )
    }

    /// The gated delta rule without subgroups: the same layout, with each token's reductions batched
    /// into two rounds through workgroup memory: (S k for the 8 columns, |q|^2, |k|^2), then S q.
    pub fn no_subgroups(&self) -> String {
        let decay = Self::columns("S{c} *= decay;");
        let update = (0..8).map(|c| format!("S{c} += kn * ((qkv[vb + {c}u] - A[{}][{}] * rk) * beta);", c / 4, c % 4)).collect::<Vec<_>>().join("\n    ");
        self.fill_with(
            r#"
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> big: array<f16>;
@group(0) @binding(2) var<storage, read> negA: array<f32>;
@group(0) @binding(3) var<storage, read> dtb: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@group(0) @binding(5) var<uniform> Lu: vec4<u32>;
@group(0) @binding(6) var<storage, read_write> state: array<f32>;
var<workgroup> part: array<vec4<f32>, 384>;    // [thread][slot]: up to 3 vec4 partial sums per thread
var<workgroup> chunk: array<vec4<f32>, 48>;    // [group][slot][8-lane chunk]
/** Sums over this thread's 32-thread group of up to 3 vec4 slots (ns used; the others are stale);
    every thread of the group gets them. */
fn sums(li: u32, ns: u32, x0: vec4<f32>, x1: vec4<f32>, x2: vec4<f32>) -> array<vec4<f32>, 3> {
  part[li * 3u] = x0; part[li * 3u + 1u] = x1; part[li * 3u + 2u] = x2;
  workgroupBarrier();
  let g = li / 32u; let l = li % 32u;
  if (l < ns * 4u) {
    let j = l / 4u; let k = l % 4u; var s = vec4<f32>(0.0);
    for (var i = 0u; i < 8u; i++) { s += part[(g * 32u + k * 8u + i) * 3u + j]; }
    chunk[(g * 3u + j) * 4u + k] = s;
  }
  workgroupBarrier();
  let b = g * 12u;
  return array<vec4<f32>, 3>((chunk[b] + chunk[b + 1u]) + (chunk[b + 2u] + chunk[b + 3u]),
    (chunk[b + 4u] + chunk[b + 5u]) + (chunk[b + 6u] + chunk[b + 7u]), (chunk[b + 8u] + chunk[b + 9u]) + (chunk[b + 10u] + chunk[b + 11u]));
}
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let sg = li / 32u; let lane = li % 32u;
  let h = wg.x / ${DV32}u; let dv0 = (wg.x % ${DV32}u) * 32u + sg * 8u;
  let C = ${C}u;
  ${DECLARE}
  let sb = h * ${DKDV}u + lane * 4u * ${DV}u + dv0;
  if (Lu.w == 1u) {
    ${LOAD}
  }
  let na = negA[h]; let db = dtb[h];
  let qs = 1.0 / sqrt(${DK}.0);
  for (var t = 0u; t < Lu.x; t++) {
    let base = t * C;
    let q4 = vec4<f32>(qkv[base + h * ${DK}u + lane * 4u], qkv[base + h * ${DK}u + lane * 4u + 1u], qkv[base + h * ${DK}u + lane * 4u + 2u], qkv[base + h * ${DK}u + lane * 4u + 3u]);
    let kb = base + ${HDK}u + h * ${DK}u + lane * 4u;
    let k4 = vec4<f32>(qkv[kb], qkv[kb + 1u], qkv[kb + 2u], qkv[kb + 3u]);
    let a = f32(big[t * ${STRIDE}u + ${AOFF}u + h]); let b = f32(big[t * ${STRIDE}u + ${BOFF}u + h]);
    let beta = 1.0 / (1.0 + exp(-b));
    let x = a + db; let sp = select(log(1.0 + exp(x)), x, x > 20.0);
    let decay = exp(na * sp);
    ${DECAY}
    let A = sums(li, 3u, vec4<f32>(dot(S0, k4), dot(S1, k4), dot(S2, k4), dot(S3, k4)),
      vec4<f32>(dot(S4, k4), dot(S5, k4), dot(S6, k4), dot(S7, k4)), vec4<f32>(dot(q4, q4), dot(k4, k4), 0.0, 0.0));
    let rk = inverseSqrt(A[2].y + 1e-6);
    let qn = q4 * inverseSqrt(A[2].x + 1e-6) * qs; let kn = k4 * rk;
    let vb = base + ${HDK2}u + h * ${DV}u + dv0;
    ${UPDATE}
    let B = sums(li, 2u, vec4<f32>(dot(S0, qn), dot(S1, qn), dot(S2, qn), dot(S3, qn)),
      vec4<f32>(dot(S4, qn), dot(S5, qn), dot(S6, qn), dot(S7, qn)), vec4<f32>(0.0));
    if (lane < 8u) { o[t * ${HDV}u + h * ${DV}u + dv0 + lane] = select(B[1], B[0], lane < 4u)[lane % 4u]; }
  }
  ${STORE}
}"#,
            &[("DECAY", decay), ("UPDATE", update)],
        )
    }
}

/// Per-head gated RMSNorm: out = w * norm(o) * silu(z), z from the fused in_proj output. One
/// subgroup per (token, head).
pub fn gnorm(h: usize, dv: usize, stride: usize, z_off: usize) -> String {
    let src = r#"
@group(0) @binding(0) var<storage, read> o: array<f32>;
@group(0) @binding(1) var<storage, read> big: array<f16>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f16>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_id) sg: u32, @builtin(subgroup_invocation_id) lane: u32) {
  let row = wg.x * 4u + sg; let t = row / ${H}u; let h = row % ${H}u;
  let b = t * ${HDV}u + h * ${DV}u;
  var v: array<f32, ${PER}>; var ss = 0.0;
  for (var i = 0u; i < ${PER}u; i++) { v[i] = o[b + lane + 32u * i]; ss += v[i] * v[i]; }
  let inv = inverseSqrt(subgroupAdd(ss) / ${DV}.0 + 1e-6);
  for (var i = 0u; i < ${PER}u; i++) {
    let c = lane + 32u * i; let z = f32(big[t * ${STRIDE}u + ${ZOFF}u + h * ${DV}u + c]);
    out[b + c] = f16(w[c] * v[i] * inv * z / (1.0 + exp(-z)));
  }
}"#;
    HDR.to_string() + &fill(src, &[("H", &h), ("HDV", &(h * dv)), ("DV", &dv), ("PER", &(dv / 32)), ("STRIDE", &stride), ("ZOFF", &z_off)])
}

/// q/k RMSNorm (1 + w) and partial rotate-half RoPE; also stores V. One workgroup per (token, head
/// slot); slots 0..HQ-1 are q heads, HQ.. k heads.
pub fn qk_prep(hq: usize, hk: usize, hd: usize, rd: usize, stride: usize) -> String {
    let src = r#"
@group(0) @binding(0) var<storage, read> big: array<f16>;
@group(0) @binding(1) var<storage, read> qw: array<f32>;
@group(0) @binding(2) var<storage, read> kw: array<f32>;
@group(0) @binding(3) var<storage, read> rope: array<vec2<f32>>;  // [L][RD/2] (cos, sin)
@group(0) @binding(4) var<storage, read_write> Qo: array<f16>;    // [HQ][L][HD]
@group(0) @binding(5) var<storage, read_write> Ko: array<f16>;    // [HK][LT][HD], rows P.. for this call
@group(0) @binding(6) var<uniform> L: vec4<u32>;                   // x = L (new tokens), y = P (cached), z = LT capacity
@group(0) @binding(7) var<storage, read_write> Vo: array<f16>;    // [HK][LT][HD]
var<workgroup> red: array<f32, 8>;
var<workgroup> nv: array<f32, ${HD}>;
@compute @workgroup_size(${HD})
fn main(@builtin(local_invocation_index) d: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_id) sg: u32, @builtin(subgroup_invocation_id) lane: u32) {
  let t = wg.x; let s = wg.y; let isq = s < ${HQ}u; let pos = L.y + t;
  if (s >= ${HQ}u) { let kh = s - ${HQ}u; Vo[(kh * L.z + pos) * ${HD}u + d] = big[t * ${STRIDE}u + ${VOFF}u + kh * ${HD}u + d]; }
  let src = select(t * ${STRIDE}u + ${KOFF}u + (s - ${HQ}u) * ${HD}u + d, t * ${STRIDE}u + s * ${HD2}u + d, isq);
  let v = f32(big[src]);
  let ss = subgroupAdd(v * v);
  if (lane == 0u) { red[sg] = ss; }
  workgroupBarrier();
  var tot = 0.0; for (var i = 0u; i < ${HD32}u; i++) { tot += red[i]; }
  let wv = select(kw[d], qw[d], isq);
  let n = v * inverseSqrt(tot / ${HD}.0 + 1e-6) * wv;
  nv[d] = n;
  workgroupBarrier();
  var r = n;
  if (d < ${RD}u) {
    let i = d % ${RD2}u; let cs = rope[pos * ${RD2}u + i];
    if (d < ${RD2}u) { r = n * cs.x - nv[d + ${RD2}u] * cs.y; } else { r = n * cs.x + nv[d - ${RD2}u] * cs.y; }
  }
  if (isq) { Qo[(s * L.x + t) * ${HD}u + d] = f16(r); } else { Ko[((s - ${HQ}u) * L.z + pos) * ${HD}u + d] = f16(r); }
}"#;
    HDR.to_string()
        + &fill(
            src,
            &[
                ("HD", &hd),
                ("HQ", &hq),
                ("STRIDE", &stride),
                ("VOFF", &(hq * hd * 2 + hk * hd)),
                ("KOFF", &(hq * hd * 2)),
                ("HD2", &(2 * hd)),
                ("HD32", &(hd / 32)),
                ("RD", &rd),
                ("RD2", &(rd / 2)),
            ],
        )
}

/// Flash-style causal GQA attention, output gated by sigmoid(gate). WG = (block of 16 query tokens, q
/// head). Keys in chunks of 256 with an online softmax: thread j scores key j against all 16 queries,
/// then thread d accumulates the 16 outputs for head dim d.
pub fn attn(hq: usize, hk: usize, hd: usize, stride: usize) -> String {
    const BQ: usize = 16;
    let src = r#"
@group(0) @binding(0) var<storage, read> Q: array<f16>;
@group(0) @binding(1) var<storage, read> K: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> big: array<f16>;
@group(0) @binding(3) var<storage, read_write> out: array<f16>;
@group(0) @binding(4) var<uniform> L: vec4<u32>;   // x = new tokens, y = P, z = LT
@group(0) @binding(5) var<storage, read> V: array<f16>;
const BQ = ${BQ}u; const HD = ${HD}u; const CH = 256u;
var<workgroup> q: array<vec4<f16>, ${QN}>;
var<workgroup> S: array<f32, ${SN}>;
var<workgroup> mrow: array<f32, ${BQ}>;
var<workgroup> lrow: array<f32, ${BQ}>;
var<workgroup> alpha: array<f32, ${BQ}>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_id) sg: u32, @builtin(subgroup_invocation_id) lane: u32) {
  let n = L.x; let P = L.y; let LT = L.z; let t0 = wg.x * BQ; let h = wg.y; let kh = h / ${GQA}u;
  let nq = min(BQ, n - t0); let tlast = P + t0 + nq - 1u;   // last absolute key position
  for (var e = li; e < BQ * HD / 4u; e += 256u) {
    let i = e / (HD / 4u); let d4 = e % (HD / 4u);
    var v = vec4<f32>(0.0);
    if (i < nq) { let b = ((h * n + t0 + i) * HD) + d4 * 4u; v = vec4<f32>(f32(Q[b]), f32(Q[b + 1u]), f32(Q[b + 2u]), f32(Q[b + 3u])) * ${SCALE}; }
    q[e] = vec4<f16>(v);
  }
  if (li < BQ) { mrow[li] = -1e30; lrow[li] = 0.0; }
  var acc: array<f32, ${BQ}>;
  workgroupBarrier();
  for (var j0 = 0u; j0 <= tlast; j0 += CH) {
    // phase 1: scores for key j = j0 + li
    let j = j0 + li;
    var s: array<f32, ${BQ}>;
    if (j <= tlast) {
      let kb = (kh * LT + j) * (HD / 4u);
      for (var d4 = 0u; d4 < HD / 4u; d4++) {
        let k = vec4<f32>(K[kb + d4]);
        for (var i = 0u; i < BQ; i++) { s[i] += dot(vec4<f32>(q[i * (HD / 4u) + d4]), k); }
      }
    }
    for (var i = 0u; i < BQ; i++) { S[i * CH + li] = select(-1e30, s[i], j <= P + t0 + i && i < nq); }
    workgroupBarrier();
    // online softmax update: subgroup sg handles queries 2sg, 2sg+1 (BQ = 16, 8 subgroups)
    for (var r = 0u; r < ${BQ8}u; r++) {
      let i = sg * ${BQ8}u + r;
      var mx = -1e30;
      for (var c = lane; c < CH; c += 32u) { mx = max(mx, S[i * CH + c]); }
      mx = subgroupMax(mx);
      let mnew = max(mrow[i], mx);
      var sum = 0.0;
      for (var c = lane; c < CH; c += 32u) { let p = exp(S[i * CH + c] - mnew); S[i * CH + c] = p; sum += p; }
      sum = subgroupAdd(sum);
      let a = exp(mrow[i] - mnew);
      workgroupBarrier();
      if (lane == 0u) { alpha[i] = a; lrow[i] = lrow[i] * a + sum; mrow[i] = mnew; }
    }
    workgroupBarrier();
    // phase 2: thread d accumulates outputs for all queries
    let d = li;
    for (var i = 0u; i < BQ; i++) { acc[i] *= alpha[i]; }
    let jend = min(j0 + CH, tlast + 1u);
    for (var jj = j0; jj < jend; jj++) {
      let v = f32(V[(kh * LT + jj) * HD + d]);
      for (var i = 0u; i < BQ; i++) { acc[i] += S[i * CH + (jj - j0)] * v; }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < nq; i++) {
    let t = t0 + i;
    let g = f32(big[t * ${STRIDE}u + h * ${HD2}u + HD + li]);
    out[t * ${HQHD}u + h * HD + li] = f16(acc[i] / lrow[i] / (1.0 + exp(-g)));
  }
}"#;
    HDR.to_string()
        + &fill(
            src,
            &[
                ("BQ8", &(BQ / 8)),
                ("BQ", &BQ),
                ("HD2", &(2 * hd)),
                ("HD", &hd),
                ("QN", &(BQ * hd / 4)),
                ("SN", &(BQ * 256)),
                ("GQA", &(hq / hk)),
                ("SCALE", &format!("{:?}", 1.0 / (hd as f64).sqrt())),
                ("STRIDE", &stride),
                ("HQHD", &(hq * hd)),
            ],
        )
}

/// WGSL dequantizing quant block `blk` (a u32 expression): `put(i, v)` receives elements 4i..4i+3 as
/// a vec4 of `t`.
fn decode_block(f: Fmt, t: &str, blk: &str, put: impl Fn(usize, String) -> String) -> String {
    decode_from("B", f, t, blk, put)
}

/// `decode_block` reading the weights from the array named `b`.
fn decode_from(b: &str, f: Fmt, t: &str, blk: &str, put: impl Fn(usize, String) -> String) -> String {
    let mut out = vec![if f.bits == 3 {
        format!("let w0 = {b}[({blk}) * 3u]; let w1 = {b}[({blk}) * 3u + 1u]; let w2 = {b}[({blk}) * 3u + 2u];")
    } else {
        format!("let wv = {b}[{blk}];")
    }];
    let word = |i: usize| if f.bits == 3 { format!("w{i}") } else { format!("wv[{i}]") };
    let g = if f.group == 32 { blk.to_string() } else { format!("({blk}) / {}u", f.group / 32) };
    out.push(format!("let sc = {t}(Sc[{g}]);"));
    if f.asym {
        out.push(format!("let bi = vec4<{t}>({t}(Bi[{g}]));"));
    }
    for i in 0..8 {
        let e = 4 * i;
        let mut q = if f.bits == 4 {
            format!("(vec4<u32>({} >> {}u) >> vec4<u32>(0u, 4u, 8u, 12u)) & vec4<u32>(15u)", word(e >> 3), 4 * (e & 7))
        } else {
            format!("(vec4<u32>({} >> {}u) >> vec4<u32>(0u, 2u, 4u, 6u)) & vec4<u32>(3u)", word(e >> 4), 2 * (e & 15))
        };
        if f.bits == 3 {
            q = format!("({q}) | (((vec4<u32>(w2 >> {e}u) >> vec4<u32>(0u, 1u, 2u, 3u)) & vec4<u32>(1u)) << vec4<u32>(2u))");
        }
        let v = if f.asym { format!("fma(vec4<{t}>({q}), vec4<{t}>(sc), bi)") } else { format!("(vec4<{t}>({q}) - {t}({})) * sc", 1 << (f.bits - 1)) };
        out.push(put(i, v));
    }
    out.join("\n")
}

fn weight_bindings(f: Fmt) -> String {
    let ty = match f.bits {
        4 => "vec4<u32>",
        3 => "u32",
        _ => "vec2<u32>",
    };
    let bias = if f.asym { "@group(0) @binding(6) var<storage, read> Bi: array<f16>;" } else { "" };
    format!("@group(0) @binding(1) var<storage, read> B: array<{ty}>;\n@group(0) @binding(2) var<storage, read> Sc: array<f16>;\n{bias}")
}

/// Multi-row GEMV for small M (any weight format). Workgroup = 16 output columns x 8 split-K lanes,
/// 8 rows. Y[M][N] = A[M][K] W[N][K]^T. Rows and weight vectors are unrolled into registers.
pub fn matmul_gemv(f: Fmt) -> String {
    const MT: usize = 8;
    let src = r#"
enable f16;
struct P { M: u32, N: u32, K: u32, blocks: u32 };
@group(0) @binding(0) var<storage, read> A: array<vec4<f16>>;
${WEIGHTS}
@group(0) @binding(3) var<storage, read_write> Y: array<f16>;
@group(0) @binding(4) var<uniform> p: P;
const C = 16u; const S = 8u; const MT = 8u;
var<workgroup> part: array<f32, 1024>;
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let c = li / S; let s = li % S;
  let n = wg.x * C + c; let m0 = wg.y * MT; let K4 = p.K / 4u;
  ${ACC}
  if (n < p.N) {
    for (var b = s; b < p.blocks; b += S) {
      ${DECODE}
      ${ROWS}
    }
  }
  ${PART}
  workgroupBarrier();
  if (s == 0u && n < p.N) {
    ${OUT}
  }
}"#;
    let rows = |line: &dyn Fn(usize) -> String| (0..MT).map(line).collect::<Vec<_>>().join("\n    ");
    let dots = (1..8).fold("dot(vec4<f32>(A[a]), wq0)".to_string(), |acc, i| format!("({acc} + dot(vec4<f32>(A[a + {i}u]), wq{i}))"));
    let decode = decode_block(f, "f32", "n * p.blocks + b", |i, v| format!("let wq{i} = {v};"));
    fill(
        src,
        &[
            ("WEIGHTS", &weight_bindings(f)),
            ("ACC", &rows(&|m| format!("var acc{m} = 0.0;"))),
            ("DECODE", &decode),
            ("ROWS", &rows(&|m| format!("if (m0 + {m}u < p.M) {{ let a = (m0 + {m}u) * K4 + b * 8u; acc{m} += {dots}; }}"))),
            ("PART", &rows(&|m| format!("part[li * MT + {m}u] = acc{m};"))),
            (
                "OUT",
                &rows(&|m| format!("if (m0 + {m}u < p.M) {{ var t = 0.0; for (var i = 0u; i < S; i++) {{ t += part[(c * S + i) * MT + {m}u]; }} Y[(m0 + {m}u) * p.N + n] = f16(t); }}")),
            ),
        ],
    )
}

/// Portable tiled GEMM (no subgroup-matrix) for int4 group-32 symmetric weights: tile tm x 64,
/// 16 x 16 threads, each accumulating tm/16 rows x 4 columns in f32. Per 32-wide K block, A and the
/// dequantized B are staged in workgroup memory as f16 rows of 8 vec4 (padded to 9 against bank conflicts).
pub fn matmul_tiled(tm: usize) -> String {
    const TN: usize = 64;
    const RN: usize = 4;
    let rm = tm / 16;
    let (tx, ty, ld) = (TN / RN, tm / rm, 9);
    let join = |n: usize, f: &dyn Fn(usize) -> String| (0..n).map(f).collect::<Vec<_>>().join(" ");
    let src = r#"
enable f16;
struct P { M: u32, N: u32, K: u32, blocks: u32 };
@group(0) @binding(0) var<storage, read> A: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read> Sc: array<f16>;
@group(0) @binding(3) var<storage, read_write> Y: array<f16>;
@group(0) @binding(4) var<uniform> p: P;
const TM = ${TM}u; const TN = ${TN}u; const TX = ${TX}u; const TY = ${TY}u; const NT = ${NT}u; const LD = ${LD}u;
var<workgroup> As: array<vec4<f16>, ${AS}>;
var<workgroup> Bs: array<vec4<f16>, ${BS}>;
@compute @workgroup_size(${NT})
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let tx = li % TX; let ty = li / TX;
  let n0 = wg.x * TN; let m0 = wg.y * TM; let K4 = p.K / 4u;
  ${ACC}
  for (var b = 0u; b < p.blocks; b++) {
    for (var e = li; e < TM * 8u; e += NT) {
      let r = e / 8u; let m = m0 + r;
      var v = vec4<f16>(0.0); if (m < p.M) { v = A[m * K4 + b * 8u + e % 8u]; }
      As[r * LD + e % 8u] = v;
    }
    for (var e = li; e < TN * 4u; e += NT) {
      let c = e / 4u; let wd = e % 4u; let blk = (n0 + c) * p.blocks + b;
      let w = vec4<u32>(B[blk * 4u + wd]); let sc = Sc[blk];
      Bs[c * LD + wd * 2u] = (vec4<f16>((w >> vec4<u32>(0u, 4u, 8u, 12u)) & vec4<u32>(15u)) - vec4<f16>(8.0)) * sc;
      Bs[c * LD + wd * 2u + 1u] = (vec4<f16>((w >> vec4<u32>(16u, 20u, 24u, 28u)) & vec4<u32>(15u)) - vec4<f16>(8.0)) * sc;
    }
    workgroupBarrier();
    for (var q = 0u; q < 8u; q++) {
      ${LOAD_A}
      ${LOAD_B}
      ${FMA}
    }
    workgroupBarrier();
  }
  ${STORE}
}"#;
    let acc = join(rm, &|i| format!("var c{i} = vec4<f32>(0.0);"));
    let load_a = join(rm, &|i| format!("let a{i} = vec4<f32>(As[(ty + {}u) * LD + q]);", i * ty));
    let load_b = join(RN, &|j| format!("let b{j} = vec4<f32>(Bs[(tx + {}u) * LD + q]);", j * tx));
    let fma = join(rm, &|i| join(RN, &|j| format!("c{i}[{j}] += dot(a{i}, b{j});")));
    let store = join(rm, &|i| {
        format!("{{ let m = m0 + ty + {}u; if (m < p.M) {{ {} }} }}", i * ty, join(RN, &|j| format!("Y[m * p.N + n0 + tx + {}u] = f16(c{i}[{j}]);", j * tx)))
    });
    fill(
        src,
        &[
            ("TM", &tm),
            ("TN", &TN),
            ("TX", &tx),
            ("TY", &ty),
            ("NT", &(tx * ty)),
            ("LD", &ld),
            ("AS", &(tm * ld)),
            ("BS", &(TN * ld)),
            ("ACC", &acc),
            ("LOAD_A", &load_a),
            ("LOAD_B", &load_b),
            ("FMA", &fma),
            ("STORE", &store),
        ],
    )
}

/// Tiled GEMM on 8x8 cooperative (simdgroup) matrices for int4 group-32 symmetric weights: tile
/// tm x 64, 4 subgroups in a 2x2 grid each owning (tm/2) x 32, f16 operands with f32 accumulators.
/// Each stage stages `kb` quant blocks (32 kb of K) of A and of dequantized B in workgroup memory.
/// The web engine's subgroup-matrix kernel (`kernelV4`), with f32 accumulation and every
/// fixed-count loop unrolled.
pub fn matmul_coop(tm: usize, kb: usize) -> String {
    let (i_n, j_n, ks) = (tm / 16, 4, 32 * kb);
    let lines = |v: Vec<String>| v.join("\n  ");
    let acc = lines((0..i_n).flat_map(|i| (0..j_n).map(move |j| format!("var c{i}{j} = coop_mat8x8<f32, C>();"))).collect());
    let stage_a = lines(
        (0..tm * ks / 4 / 128)
            .map(|u| {
                format!(
                    "{{ let idx = li + {}u; let r = idx / {ks4}u; let q = idx % {ks4}u; var v = vec4<f16>(0.0); if (m0 + r < p.M) {{ v = Act[(m0 + r) * K4 + b0 * 8u + q]; }} let o = r * {ks}u + q * 4u; stA[o] = v.x; stA[o + 1u] = v.y; stA[o + 2u] = v.z; stA[o + 3u] = v.w; }}",
                    128 * u,
                    ks4 = ks / 4
                )
            })
            .collect(),
    );
    let fmt = Fmt { bits: 4, group: 32, asym: false };
    let stage_b = lines(
        (0..(64 * kb).div_ceil(128))
            .map(|u| {
                let decode = decode_from("Wq", fmt, "f16", "blk", |i, v| format!("{{ let v = {v}; stB[o + {}u] = v.x; stB[o + {}u] = v.y; stB[o + {}u] = v.z; stB[o + {}u] = v.w; }}", 4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3));
                // 64 columns x kb blocks, one block per thread (half the threads idle when kb = 1)
                let guard = if 128 * (u + 1) > 64 * kb { format!("if (idx < {}u) ", 64 * kb) } else { String::new() };
                format!("{{ let idx = li + {}u; {guard}{{ let c = idx / {kb}u; let kb = idx % {kb}u; let blk = (n0 + c) * p.blocks + b0 + kb; let o = c * {ks}u + kb * 32u;\n    {decode} }} }}", 128 * u)
            })
            .collect(),
    );
    let mma = lines(
        (0..ks / 8)
            .flat_map(|k| {
                let kk = 8 * k;
                (0..i_n)
                    .map(move |i| format!("let a{i}_{k} = coopLoadT<coop_mat8x8<f16, A>>(&stA[ao + {}u], {ks}u);", i * 8 * ks + kk))
                    .chain((0..j_n).map(move |j| format!("let b{j}_{k} = coopLoad<coop_mat8x8<f16, B>>(&stB[bo + {}u], {ks}u);", j * 8 * ks + kk)))
                    .chain((0..i_n).flat_map(move |i| (0..j_n).map(move |j| format!("c{i}{j} = coopMultiplyAdd(a{i}_{k}, b{j}_{k}, c{i}{j});"))))
            })
            .collect(),
    );
    let store = lines((0..i_n).flat_map(|i| (0..j_n).map(move |j| format!("coopStoreT(c{i}{j}, &stC[sg * {}u + {}u], 32u);", tm / 2 * 32, i * 8 * 32 + j * 8))).collect());
    let src = r#"
enable f16;
enable wgpu_cooperative_matrix;
struct P { M: u32, N: u32, K: u32, blocks: u32 };
@group(0) @binding(0) var<storage, read> Act: array<vec4<f16>>;   // not A/B: those name the matrix roles
@group(0) @binding(1) var<storage, read> Wq: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> Sc: array<f16>;
@group(0) @binding(3) var<storage, read_write> Y: array<f16>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> stA: array<f16, ${SA}>;
var<workgroup> stB: array<f16, ${SB}>;
var<workgroup> stC: array<f32, ${SC}>;
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_id) sg: u32) {
  let n0 = wg.x * 64u; let m0 = wg.y * ${TM}u; let K4 = p.K / 4u;
  let ao = (sg / 2u) * ${HALF}u * ${KS}u; let bo = (sg % 2u) * 32u * ${KS}u;
  ${ACC}
  for (var b0 = 0u; b0 < p.blocks; b0 += ${KB}u) {
  ${STAGE_A}
  ${STAGE_B}
  workgroupBarrier();
  ${MMA}
  workgroupBarrier();
  }
  ${STORE}
  workgroupBarrier();
  for (var e = li; e < ${SC}u; e += 128u) {
    let s = e / ${SUB}u; let l = e % ${SUB}u;
    let r = m0 + (s / 2u) * ${HALF}u + l / 32u; let c = n0 + (s % 2u) * 32u + l % 32u;
    if (r < p.M) { Y[r * p.N + c] = f16(stC[e]); }
  }
}"#;
    fill(
        src,
        &[
            ("SA", &(tm * ks)),
            ("SB", &(64 * ks)),
            ("SC", &(tm * 64)),
            ("TM", &tm),
            ("HALF", &(tm / 2)),
            ("KS", &ks),
            ("KB", &kb),
            ("SUB", &(tm / 2 * 32)),
            ("ACC", &acc),
            ("STAGE_A", &stage_a),
            ("STAGE_B", &stage_b),
            ("MMA", &mma),
            ("STORE", &store),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kernel, with and without subgroups, parses and validates in naga.
    #[test]
    fn kernels_validate() {
        let f4 = Fmt { bits: 4, group: 32, asym: false };
        let f3 = Fmt { bits: 3, group: 32, asym: false };
        let g = Gdn { h: 16, dk: 128, dv: 128, stride: 8256, a_off: 8192, b_off: 8208 };
        let mut all = vec![
            embed(2048, f3),
            embed(2048, f4),
            add_norm(2048, true, false),
            add_norm(2048, false, false),
            add_norm(2048, true, true),
            silu_mul(6144),
            add_bias(2048),
            conv(6144, 8256),
            hist_update(6144, 8256),
            g.subgroups(),
            g.no_subgroups(),
            gnorm(16, 128, 8256, 6144),
            qk_prep(8, 2, 256, 64, 5120),
            attn(8, 2, 256, 5120),
            matmul_gemv(f4),
            matmul_gemv(f3),
            matmul_tiled(16),
            matmul_tiled(80),
            matmul_tiled(128),
        ];
        all.extend(all.clone().iter().map(|k| no_subgroups(k)));
        // cooperative matrices are only used along with subgroups
        all.extend([matmul_coop(32, 2), matmul_coop(64, 2)]);
        for src in all {
            let module = wgpu::naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{}\n{src}", e.emit_to_string(&src)));
            wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{e:?}\n{src}"));
        }
    }
}
