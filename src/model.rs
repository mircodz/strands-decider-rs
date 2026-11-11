//! Model files: the manifest, downloading the hosted build, and turning its packed download format
//! (`engine/wire.js` in the web build) into the layout the kernels read, once, at download time.
//!
//! A model directory holds `manifest.json`, `weights.<i>.bin` (each shard in GPU layout),
//! the embedding rows file, `tokenizer.json`, `head.safetensors` and `hobson_config.json`.

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use serde::Deserialize;

use crate::{Result, err};

/// The hosted v21 build, pinned to a revision so a cached download never goes stale.
pub const REPO: &str = "alxnahas/strands-decider-v21-webgpu";
pub const REVISION: &str = "808299822720057dd12e01ed3bdfd26f0a81be9b";

#[derive(Deserialize, Clone)]
pub struct Manifest {
    pub config: Config,
    pub shards: Vec<Shard>,
    pub tensors: HashMap<String, Tensor>,
    pub embed: Option<Embed>,
    pub tokenizer: Option<String>,
}

#[derive(Deserialize, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_types: Vec<String>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rotary_dim: usize,
    /// A rotated torso: the original basis of a final hidden row is `unrotate * FWHT(row)`.
    pub unrotate: Option<Vec<f64>>,
}

#[derive(Deserialize, Clone)]
pub struct Shard {
    pub path: String,
    pub start: u64,
    pub bytes: u64,
    /// Set for a packed shard: its tensors back to back, scales as 8-bit codes.
    pub wire: Option<u64>,
}

#[derive(Deserialize, Clone)]
pub struct Tensor {
    #[serde(default)]
    pub offset: u64,
    #[serde(default)]
    pub bytes: u64,
    #[serde(default)]
    pub shape: Vec<usize>,
    #[serde(rename = "N", default)]
    pub n: usize,
    #[serde(rename = "K", default)]
    pub k: usize,
    #[serde(default = "four")]
    pub bits: u32,
    #[serde(default = "thirty_two")]
    pub group: u32,
    #[serde(default)]
    pub asym: bool,
    /// Online Hadamard rotation of the input (not used by the hosted build, and not supported).
    pub had: Option<u32>,
    /// Stored in the packed shard as 8-bit scale codes.
    #[serde(default)]
    pub u8: bool,
    /// Offset in the packed shard.
    pub wire: Option<u64>,
}

fn four() -> u32 {
    4
}
fn thirty_two() -> u32 {
    32
}

/// The embedding table, kept out of the shards: one file of rows, in token id order.
#[derive(Deserialize, Clone)]
pub struct Embed {
    pub rows: String,
    pub vocab: usize,
    pub row_bytes: usize,
    /// (tensor name, bytes per row in GPU layout), in the order the row stores them.
    pub parts: Vec<(String, usize)>,
    pub wire_row_bytes: Option<usize>,
    #[serde(default)]
    pub u8: bool,
}

pub fn read_manifest(dir: &Path) -> Result<Manifest> {
    let text = fs::read_to_string(dir.join("manifest.json")).map_err(|e| err(format!("{}: {e}", dir.join("manifest.json").display())))?;
    serde_json::from_str(&text).map_err(|e| err(format!("manifest.json: {e}")))
}

pub fn shard_file(i: usize) -> String {
    format!("weights.{i}.bin")
}

/// Where the default model is cached: the platform's cache directory.
pub fn default_dir() -> PathBuf {
    let env = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    let base = if cfg!(windows) {
        env("LOCALAPPDATA")
    } else if cfg!(target_os = "macos") {
        env("HOME").map(|h| h.join("Library/Caches"))
    } else {
        env("XDG_CACHE_HOME").or_else(|| env("HOME").map(|h| h.join(".cache")))
    };
    base.unwrap_or_else(std::env::temp_dir).join("strands-decider-rs").join(REPO.replace('/', "--")).join(REVISION)
}

/// Whether `dir` holds a complete converted model.
fn complete(dir: &Path) -> bool {
    let Ok(m) = read_manifest(dir) else { return false };
    let files = (0..m.shards.len()).map(shard_file).chain(m.embed.map(|e| e.rows)).chain(["tokenizer.json", "head.safetensors", "hobson_config.json"].map(String::from));
    files.into_iter().all(|f| dir.join(f).exists())
}

/// Downloads the pinned hosted build from Hugging Face into `dir`, converting it to the layout the
/// engine reads, unless `dir` already holds it. Downloads go through a scratch Hugging Face cache in
/// `dir`, removed once every file is converted.
pub fn download(dir: &Path) -> Result<()> {
    if complete(dir) {
        return Ok(());
    }
    let hub = dir.join("hub");
    fs::create_dir_all(dir)?;
    let api = ApiBuilder::from_env().with_cache_dir(hub.clone()).with_progress(true).build().map_err(|e| err(format!("Hugging Face: {e}")))?;
    let repo = api.repo(Repo::with_revision(REPO.into(), RepoType::Model, REVISION.into()));
    // The downloaded file (its blob, not the cache's link to it).
    let get = |path: &str| -> Result<PathBuf> { Ok(fs::canonicalize(repo.get(path).map_err(|e| err(format!("downloading {path}: {e}")))?)?) };
    // Moved out of the scratch cache unchanged.
    let take = |path: &str, dest: &Path| -> Result<()> {
        if !dest.exists() {
            fs::rename(get(path)?, dest)?;
        }
        Ok(())
    };
    take("engine-weights/manifest.json", &dir.join("manifest.json"))?;
    let m = read_manifest(dir)?;
    for (i, s) in m.shards.iter().enumerate() {
        let out = dir.join(shard_file(i));
        if out.exists() {
            continue;
        }
        let data = read_maybe_gz(&get(&format!("engine-weights/{}", s.path))?, &s.path, s.wire.unwrap_or(s.bytes) as usize)?;
        let gpu = match s.wire {
            Some(_) => unpack_shard(&data, s, &m.tensors)?,
            None => data,
        };
        if gpu.len() as u64 != s.bytes {
            return Err(err(format!("{}: {} bytes, the manifest says {}", s.path, gpu.len(), s.bytes)));
        }
        write_atomic(&out, &gpu)?;
    }
    if let Some(e) = &m.embed {
        take(&format!("engine-weights/{}", e.rows), &dir.join(&e.rows))?;
    }
    let tokenizer = dir.join("tokenizer.json");
    if !tokenizer.exists() {
        let name = m.tokenizer.as_deref().unwrap_or("tokenizer.json");
        write_atomic(&tokenizer, &read_maybe_gz(&get(&format!("model/{name}"))?, name, 0)?)?;
    }
    for name in ["head.safetensors", "hobson_config.json"] {
        take(&format!("model/{name}"), &dir.join(name))?;
    }
    fs::remove_dir_all(&hub)?;
    Ok(())
}

/// The bytes of `path`, gunzipped when `name` (its file name in the repository) ends in `.gz`.
fn read_maybe_gz(path: &Path, name: &str, size_hint: usize) -> Result<Vec<u8>> {
    let raw = fs::read(path)?;
    if !name.ends_with(".gz") {
        return Ok(raw);
    }
    let mut out = Vec::with_capacity(size_hint);
    flate2::read::MultiGzDecoder::new(&raw[..]).read_to_end(&mut out).map_err(|e| err(format!("{name}: {e}")))?;
    Ok(out)
}

fn write_atomic(path: &Path, data: &[u8]) -> Result<()> {
    let tmp = PathBuf::from(format!("{}.tmp", path.display()));
    fs::write(&tmp, data)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// `2^(-i / 4096)`, grown on demand: the per-row log-scale steps of the 8-bit scale codes.
#[derive(Default)]
struct Pow(Vec<f64>);

impl Pow {
    fn upto(&mut self, n: usize) -> &[f64] {
        if self.0.len() <= n {
            self.0 = (0..=n).map(|i| 2f64.powf(-(i as f64) / 4096.0)).collect();
        }
        &self.0
    }
}

/// `n` rows of `g` scale codes at `src` (a 4-byte header per row: m as fp16 bits and j as u16, then
/// the codes `sign << 7 | k`) as fp16 bytes into `dst`: |scale| = fp16(f32(m 2^(-k j / 4096))).
fn decode_scales(src: &[u8], n: usize, g: usize, dst: &mut [u8], pow: &mut Pow) {
    let head = |r: usize, i: usize| u16::from_le_bytes([src[4 * r + 2 * i], src[4 * r + 2 * i + 1]]);
    let jmax = (0..n).map(|r| head(r, 1) as usize).max().unwrap_or(0);
    let p = pow.upto(127 * jmax);
    let codes = &src[4 * n..4 * n + n * g];
    for r in 0..n {
        let (m, j) = (half::f16::from_bits(head(r, 0)).to_f64(), head(r, 1) as usize);
        let mut row = [u16::MAX; 128];
        for (i, &c) in codes[r * g..(r + 1) * g].iter().enumerate() {
            let k = (c & 127) as usize;
            if row[k] == u16::MAX {
                row[k] = half::f16::from_f32((m * p[k * j]) as f32).to_bits();
            }
            let h = row[k] | ((c as u16 & 128) << 8);
            dst[2 * (r * g + i)..2 * (r * g + i) + 2].copy_from_slice(&h.to_le_bytes());
        }
    }
}

/// A packed shard (`wire`) in GPU layout: tensors at their offsets, scale codes decoded to fp16.
fn unpack_shard(wire: &[u8], s: &Shard, tensors: &HashMap<String, Tensor>) -> Result<Vec<u8>> {
    let mut out = vec![0u8; s.bytes as usize];
    let mut pow = Pow::default();
    for (name, t) in tensors {
        let Some(w) = t.wire.map(|w| w as usize) else { continue };
        if t.offset < s.start || t.offset >= s.start + s.bytes {
            continue;
        }
        let dst = &mut out[(t.offset - s.start) as usize..(t.offset - s.start + t.bytes) as usize];
        if t.u8 {
            let (n, g) = (t.shape[0], t.shape[1]);
            let src = wire.get(w..w + 4 * n + n * g).ok_or_else(|| err(format!("packed shard: {name} out of range")))?;
            decode_scales(src, n, g, dst, &mut pow);
        } else {
            dst.copy_from_slice(wire.get(w..w + t.bytes as usize).ok_or_else(|| err(format!("packed shard: {name} out of range")))?);
        }
    }
    Ok(out)
}

/// The embedding table in GPU layout, one buffer per part (`embed.q`, `embed.s`), all rows.
pub fn embed_parts(dir: &Path, e: &Embed) -> Result<Vec<(String, Vec<u8>)>> {
    let data = fs::read(dir.join(&e.rows))?;
    let rb = e.wire_row_bytes.unwrap_or(e.row_bytes);
    if data.len() != e.vocab * rb {
        return Err(err(format!("{}: {} bytes, expected {} rows of {rb}", e.rows, data.len(), e.vocab)));
    }
    let mut parts: Vec<(String, Vec<u8>)> = e.parts.iter().map(|(n, b)| (n.clone(), vec![0u8; e.vocab * b])).collect();
    let mut pow = Pow::default();
    for (r, row) in data.chunks_exact(rb).enumerate() {
        let mut off = 0;
        for (name, buf) in parts.iter_mut() {
            let b = buf.len() / e.vocab;
            let dst = &mut buf[r * b..(r + 1) * b];
            if e.u8 && name == "embed.s" {
                decode_scales(&row[off..off + 4 + b / 2], 1, b / 2, dst, &mut pow);
                off += 4 + b / 2;
            } else {
                dst.copy_from_slice(&row[off..off + b]);
                off += b;
            }
        }
    }
    Ok(parts)
}
