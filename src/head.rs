//! The pointer head (`PointerHead` in the official `modeling.py`): LayerNorm, then a query from the
//! `<answer>` token and a key from the last token of each option line; logits are their scaled dot
//! products. It runs on the CPU in f64: it is tiny next to the torso.

use std::collections::HashMap;

use serde::Deserialize;

use crate::{Result, err};

pub struct Head {
    t: HashMap<String, Vec<f32>>,
    dim: usize,
}

#[derive(Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

impl Head {
    /// From `head.safetensors` (all tensors f32).
    pub fn load(bytes: &[u8]) -> Result<Head> {
        let bad = || err("head.safetensors: malformed");
        let n = u64::from_le_bytes(bytes.get(..8).ok_or_else(bad)?.try_into().unwrap()) as usize;
        let header: HashMap<String, serde_json::Value> = serde_json::from_slice(bytes.get(8..8 + n).ok_or_else(bad)?).map_err(|_| bad())?;
        let data = &bytes[8 + n..];
        let mut t = HashMap::new();
        let mut dim = 0;
        for (name, v) in header {
            if name == "__metadata__" {
                continue;
            }
            let e: Entry = serde_json::from_value(v).map_err(|_| bad())?;
            if e.dtype != "F32" {
                return Err(err(format!("head tensor {name} is {}, expected F32", e.dtype)));
            }
            if name == "q.weight" {
                dim = e.shape[0];
            }
            let raw = data.get(e.data_offsets.0..e.data_offsets.1).ok_or_else(bad)?;
            t.insert(name, bytemuck::pod_collect_to_vec::<u8, f32>(raw));
        }
        for k in ["norm.weight", "norm.bias", "q.weight", "q.bias", "k.weight", "k.bias"] {
            if !t.contains_key(k) {
                return Err(err(format!("head.safetensors: no {k}")));
            }
        }
        Ok(Head { t, dim })
    }

    /// LayerNorm (eps 1e-5) of a hidden row, then the `q` or `k` projection.
    fn project(&self, which: &str, x: &[f32]) -> Vec<f64> {
        let d = x.len() as f64;
        let mean = x.iter().map(|&v| v as f64).sum::<f64>() / d;
        let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d;
        let r = 1.0 / (var + 1e-5).sqrt();
        let (nw, nb) = (&self.t["norm.weight"], &self.t["norm.bias"]);
        let xn: Vec<f64> = x.iter().enumerate().map(|(j, &v)| (v as f64 - mean) * r * nw[j] as f64 + nb[j] as f64).collect();
        let (w, b) = (&self.t[&format!("{which}.weight")], &self.t[&format!("{which}.bias")]);
        (0..self.dim).map(|o| b[o] as f64 + w[o * x.len()..(o + 1) * x.len()].iter().zip(&xn).map(|(&w, x)| w as f64 * x).sum::<f64>()).collect()
    }

    /// Option probabilities from hidden rows `[option 0, .., option n-1, answer]` (each `d` wide).
    pub fn probs(&self, hidden: &[f32], d: usize, temperature: f64) -> Vec<f64> {
        let rows: Vec<&[f32]> = hidden.chunks_exact(d).collect();
        let (answer, options) = rows.split_last().expect("no hidden rows");
        let q = self.project("q", answer);
        let scale = (self.dim as f64).powf(-0.5) / temperature.max(1e-6);
        let logits: Vec<f64> = options.iter().map(|o| self.project("k", o).iter().zip(&q).map(|(k, q)| k * q).sum::<f64>() * scale).collect();
        let max = logits.iter().cloned().fold(f64::MIN, f64::max);
        let e: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
        let z: f64 = e.iter().sum();
        e.iter().map(|x| x / z).collect()
    }
}
