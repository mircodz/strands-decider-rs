//! Parity with the official package and Hugging Face `tokenizers` (fixtures from tests/fixtures/make.py).
//! The tokenizer tests need a downloaded model (`strands-decider-rs download`, or STRANDS_DECIDER_MODEL)
//! and are skipped without one.

use std::path::PathBuf;

use serde_json::Value;
use strands_decider::tokenizer::Tokenizer;
use strands_decider::{Request, model, prompt};
use unicode_normalization::UnicodeNormalization;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/expected.json")).unwrap()
}

fn tokenizer() -> Option<Tokenizer> {
    let dir = std::env::var_os("STRANDS_DECIDER_MODEL").map(PathBuf::from).unwrap_or_else(model::default_dir);
    let Ok(text) = std::fs::read_to_string(dir.join("tokenizer.json")) else {
        eprintln!("skipped: no tokenizer.json in {} (run `strands-decider-rs download`)", dir.display());
        return None;
    };
    Some(Tokenizer::from_json(&text).unwrap())
}

fn u32s(v: &Value) -> Vec<u32> {
    serde_json::from_value(v.clone()).unwrap()
}

#[test]
fn renders_questions_like_the_official_package() {
    for case in fixture()["cases"].as_array().unwrap() {
        let req = Request::from_json(&case["request"]).unwrap();
        for ((_, q), want) in req.questions.iter().zip(case["question_text"].as_array().unwrap()) {
            // Ours is NFC up front (the tokenizer normalizes the official text the same way).
            assert_eq!(prompt::render_question(q).text, want.as_str().unwrap().nfc().collect::<String>(), "{}", case["name"]);
        }
    }
}

#[test]
fn tokenizes_prompts_like_the_official_package() {
    let Some(tok) = tokenizer() else { return };
    for case in fixture()["cases"].as_array().unwrap() {
        let name = &case["name"];
        let req = Request::from_json(&case["request"]).unwrap();
        assert_eq!(tok.encode(&prompt::render_state(&req.state)).unwrap().ids, u32s(&case["state_ids"]), "{name}: state");
        for (i, (_, q)) in req.questions.iter().enumerate() {
            let rq = prompt::render_question(q);
            let enc = tok.encode(&rq.text).unwrap();
            assert_eq!(enc.ids, u32s(&case["question_ids"][i]), "{name}: question {i}");
            let opts: Vec<usize> = serde_json::from_value(case["option_idx"][i].clone()).unwrap();
            assert_eq!(prompt::option_token_index(&enc.offsets, &rq.spans).unwrap(), opts, "{name}: options {i}");
        }
    }
}

#[test]
fn tokenizes_strings_like_hugging_face() {
    let Some(tok) = tokenizer() else { return };
    for case in fixture()["raw"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let enc = tok.encode(text).unwrap();
        assert_eq!(enc.ids, u32s(&case["ids"]), "{text:?}");
        // Offsets are into the NFC text, so they match only where normalization changes nothing.
        if text.nfc().eq(text.chars()) {
            let want: Vec<(usize, usize)> = serde_json::from_value(case["offsets"].clone()).unwrap();
            assert_eq!(enc.offsets, want, "{text:?}");
        }
    }
}
