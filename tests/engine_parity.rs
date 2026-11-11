//! GPU parity with the web engine: option probabilities for the token ids in expected.json, against
//! the browser engine's (tests/fixtures/browser.mjs). Needs a GPU and a downloaded model; skipped
//! without a model.

use std::path::PathBuf;

use serde_json::Value;
use strands_decider::engine::Engine;
use strands_decider::head::Head;
use strands_decider::model;

fn fixture(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap()
}

#[test]
fn matches_the_web_engine() {
    let dir = std::env::var_os("STRANDS_DECIDER_MODEL").map(PathBuf::from).unwrap_or_else(model::default_dir);
    if !dir.join("weights.0.bin").exists() {
        eprintln!("skipped: no model in {} (run `strands-decider-rs download`)", dir.display());
        return;
    }
    let m = model::read_manifest(&dir).unwrap();
    let mut engine = Engine::load(&dir, &m).unwrap();
    let head = Head::load(&std::fs::read(dir.join("head.safetensors")).unwrap()).unwrap();
    let cfg: Value = serde_json::from_slice(&std::fs::read(dir.join("hobson_config.json")).unwrap()).unwrap();
    let browser = fixture("browser.json");
    let mut want = browser["res"].as_array().unwrap().iter();
    let (mut worst, mut worst_name, mut same_top) = (0f64, String::new(), 0);
    let mut n = 0;
    for case in fixture("expected.json")["cases"].as_array().unwrap() {
        let state: Vec<u32> = serde_json::from_value(case["state_ids"].clone()).unwrap();
        let kinds: Vec<&Value> = case["request"]["questions"].as_object().unwrap().values().collect();
        for (i, q) in case["question_ids"].as_array().unwrap().iter().enumerate() {
            let ids = [state.clone(), serde_json::from_value::<Vec<u32>>(q.clone()).unwrap()].concat();
            let opts: Vec<usize> = serde_json::from_value(case["option_idx"][i].clone()).unwrap();
            let rows: Vec<usize> = opts.iter().map(|j| state.len() + j).chain([ids.len() - 1]).collect();
            let out = engine.forward(&ids, &rows, None, false, false).unwrap();
            let kind = kinds[i]["type"].as_str().unwrap();
            let t = cfg["temperature_by_kind"][kind].as_f64().unwrap_or(cfg["temperature"].as_f64().unwrap());
            let probs = head.probs(&out.hidden, out.d, t);
            let w = want.next().unwrap();
            let theirs: Vec<f64> = serde_json::from_value(w["probs"].clone()).unwrap();
            let diff = probs.iter().zip(&theirs).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            let argmax = |p: &[f64]| (0..p.len()).fold(0, |b, i| if p[i] > p[b] { i } else { b });
            same_top += (argmax(&probs) == argmax(&theirs)) as usize;
            if diff > worst {
                (worst, worst_name) = (diff, w["name"].as_str().unwrap().to_string());
            }
            n += 1;
        }
    }
    eprintln!("{}: {n} prompts, {same_top} same top answer, max |dp| {worst:.5} ({worst_name})", engine.adapter);
    assert_eq!(same_top, n);
    assert!(worst < 0.01, "max probability difference {worst} ({worst_name})");
}

/// A question continued from a snapshot of its state answers as the whole prompt does.
#[test]
fn prefix_cache_matches_full_forwards() {
    let dir = std::env::var_os("STRANDS_DECIDER_MODEL").map(PathBuf::from).unwrap_or_else(model::default_dir);
    if !dir.join("weights.0.bin").exists() {
        eprintln!("skipped: no model in {}", dir.display());
        return;
    }
    let mut engine = Engine::load(&dir, &model::read_manifest(&dir).unwrap()).unwrap();
    let head = Head::load(&std::fs::read(dir.join("head.safetensors")).unwrap()).unwrap();
    let mut worst = 0f64;
    for case in fixture("expected.json")["cases"].as_array().unwrap() {
        let state: Vec<u32> = serde_json::from_value(case["state_ids"].clone()).unwrap();
        let prefix = engine.forward(&state, &[], None, true, false).unwrap().snapshot.unwrap();
        for (i, q) in case["question_ids"].as_array().unwrap().iter().enumerate() {
            let q: Vec<u32> = serde_json::from_value(q.clone()).unwrap();
            let opts: Vec<usize> = serde_json::from_value(case["option_idx"][i].clone()).unwrap();
            let rows: Vec<usize> = opts.iter().copied().chain([q.len() - 1]).collect();
            let cached = engine.forward(&q, &rows, Some(&prefix), false, false).unwrap();
            let full_rows: Vec<usize> = rows.iter().map(|r| state.len() + r).collect();
            let full = engine.forward(&[state.clone(), q].concat(), &full_rows, None, false, false).unwrap();
            let (a, b) = (head.probs(&cached.hidden, cached.d, 1.0), head.probs(&full.hidden, full.d, 1.0));
            worst = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(worst, f64::max);
        }
    }
    eprintln!("{}: prefix vs full forward, max |dp| {worst:.6}", engine.adapter);
    assert!(worst < 2e-3, "{worst}");
}
