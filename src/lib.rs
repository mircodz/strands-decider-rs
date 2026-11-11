#![doc = include_str!("../README.md")]

pub mod engine;
pub mod head;
pub mod kernels;
pub mod model;
pub mod prompt;
pub mod tokenizer;

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use serde::Deserialize;
use serde_json::{Value, json};

use engine::Engine;
use head::Head;
pub use prompt::{Answer, IntoChoice, Question};
use prompt::Rendered;
use tokenizer::Tokenizer;

#[derive(Debug)]
pub struct Error(pub String);

pub type Result<T> = std::result::Result<T, Error>;

pub fn err(msg: impl Into<String>) -> Error {
    Error(msg.into())
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Error {
        Error(e.to_string())
    }
}

/// A System One request: a state and named questions about it, in order.
pub struct Request {
    pub state: Value,
    pub questions: Vec<(String, Question)>,
}

impl Request {
    /// From the `POST /v1/systemone` body: `{"state": ..., "questions": {"<name>": {...}}}`.
    pub fn from_json(v: &Value) -> Result<Request> {
        let state = v.get("state").filter(|s| s.is_string() || s.is_object() || s.is_array()).ok_or_else(|| err("state must be a string, object or array"))?;
        if v.get("images").and_then(Value::as_array).is_some_and(|i| !i.is_empty()) {
            return Err(err("this engine has no vision tower; images are not supported"));
        }
        let qs = v.get("questions").and_then(Value::as_object).filter(|q| !q.is_empty()).ok_or_else(|| err("questions must be a non-empty object"))?;
        let questions = qs.iter().map(|(name, q)| Ok((name.clone(), Question::from_json(q).map_err(|e| err(format!("question {name}: {e}")))?))).collect::<Result<_>>()?;
        Ok(Request { state: state.clone(), questions })
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Timings {
    pub tokenize_ms: f64,
    pub forward_ms: f64,
    pub total_ms: f64,
}

pub struct Response {
    pub answers: Vec<(String, Answer)>,
    pub input_tokens: usize,
    pub timings: Timings,
}

impl Response {
    /// The System One response body.
    pub fn to_json(&self) -> Value {
        json!({
            "model": format!("strands-decider-rs-{}", env!("CARGO_PKG_VERSION")),
            "answers": Value::Object(self.answers.iter().map(|(k, a)| (k.clone(), a.to_json())).collect()),
            "usage": {"input_tokens": self.input_tokens, "output_tokens": self.answers.len()},
        })
    }
}

/// The checkpoint's decider settings (`hobson_config.json`).
#[derive(Deserialize)]
struct DeciderConfig {
    max_length: usize,
    temperature: f64,
    #[serde(default)]
    temperature_by_kind: HashMap<String, f64>,
    #[serde(default)]
    ordinal_smoothing: f64,
}

pub struct Decider {
    pub engine: Engine,
    tokenizer: Tokenizer,
    head: Head,
    cfg: DeciderConfig,
}

/// Questions per forward chunk in the official engine (`max_batch`); only the token count depends on it.
const MAX_BATCH: usize = 32;

impl Decider {
    /// The model from Hugging Face: downloaded (about 650 MB) and converted on first use, into the
    /// platform's cache directory, then loaded from there.
    pub fn new() -> Result<Decider> {
        let dir = model::default_dir();
        model::download(&dir)?;
        Decider::load(&dir)
    }

    /// The model in `dir` (see [`model::download`]).
    pub fn load(dir: impl AsRef<Path>) -> Result<Decider> {
        let dir = dir.as_ref();
        let read = |name: &str| std::fs::read(dir.join(name)).map_err(|e| err(format!("{}: {e}", dir.join(name).display())));
        let manifest = model::read_manifest(dir)?;
        // The tokenizer parses on another thread while the weights upload.
        let tokenizer = std::thread::scope(|s| {
            let tok = s.spawn(|| Tokenizer::from_json(&String::from_utf8_lossy(&read("tokenizer.json")?)));
            let engine = Engine::load(dir, &manifest);
            (engine, tok.join().unwrap())
        });
        let (engine, tokenizer) = (tokenizer.0?, tokenizer.1?);
        let head = Head::load(&read("head.safetensors")?)?;
        let cfg: DeciderConfig = serde_json::from_slice(&read("hobson_config.json")?).map_err(|e| err(format!("hobson_config.json: {e}")))?;
        Ok(Decider { engine, tokenizer, head, cfg })
    }

    /// Answers a question about `state`: text, or JSON content (`serde_json::Value`).
    pub fn ask(&mut self, state: impl Into<Value>, question: Question) -> Result<Answer> {
        Ok(self.ask_all(state, [question])?.remove(0))
    }

    /// Answers several questions about one state, in order. The state is encoded once.
    pub fn ask_all(&mut self, state: impl Into<Value>, questions: impl IntoIterator<Item = Question>) -> Result<Vec<Answer>> {
        let questions: Vec<(String, Question)> = questions.into_iter().enumerate().map(|(i, q)| (i.to_string(), q)).collect();
        Ok(self.run(&state.into(), &questions)?.answers.into_iter().map(|(_, a)| a).collect())
    }

    /// Answers a System One request (`POST /v1/systemone`), with usage and timings.
    pub fn evaluate(&mut self, req: &Request) -> Result<Response> {
        self.run(&req.state, &req.questions)
    }

    fn temperature(&self, rq: &Rendered) -> f64 {
        *self.cfg.temperature_by_kind.get(rq.kind.name()).unwrap_or(&self.cfg.temperature)
    }

    /// Answers every question. Several questions share one encoding of the state: it is forwarded once,
    /// snapshotted, and each question continues from the snapshot.
    fn run(&mut self, state: &Value, named: &[(String, Question)]) -> Result<Response> {
        let t0 = Instant::now();
        if !(state.is_string() || state.is_object() || state.is_array()) {
            return Err(err("state must be a string, object or array"));
        }
        if named.is_empty() {
            return Err(err("ask at least one question"));
        }
        for (_, q) in named {
            q.validate()?;
        }
        let rendered: Vec<Rendered> = named.iter().map(|(_, q)| prompt::render_question(q)).collect();
        let state = self.tokenizer.encode(&prompt::render_state(state))?.ids;
        let questions = rendered.iter().map(|r| self.tokenizer.encode(&r.text)).collect::<Result<Vec<_>>>()?;
        let longest = questions.iter().map(|q| q.ids.len()).max().unwrap_or(0);
        if state.len() + longest > self.cfg.max_length {
            return Err(err(format!(
                "prompt of {} tokens ({} state + {longest} question) exceeds the context window of {} tokens",
                state.len() + longest,
                state.len(),
                self.cfg.max_length
            )));
        }
        let options = questions.iter().zip(&rendered).map(|(q, r)| prompt::option_token_index(&q.offsets, &r.spans)).collect::<Result<Vec<_>>>()?;
        let tokenized = Instant::now();
        let mut answers = Vec::new();
        let d = self.engine.hidden_size();
        if questions.len() == 1 {
            let ids = [state.as_slice(), &questions[0].ids].concat();
            let rows: Vec<usize> = options[0].iter().map(|i| state.len() + i).chain([ids.len() - 1]).collect();
            let out = self.engine.forward(&ids, &rows, None, false, false)?;
            let probs = self.head.probs(&out.hidden, d, self.temperature(&rendered[0]));
            answers.push((named[0].0.clone(), prompt::to_answer(&rendered[0], &probs, self.cfg.ordinal_smoothing)));
        } else {
            let prefix = self.engine.forward(&state, &[], None, true, false)?.snapshot.unwrap();
            for (i, q) in questions.iter().enumerate() {
                let rows: Vec<usize> = options[i].iter().copied().chain([q.ids.len() - 1]).collect();
                let out = self.engine.forward(&q.ids, &rows, Some(&prefix), false, false)?;
                let probs = self.head.probs(&out.hidden, d, self.temperature(&rendered[i]));
                answers.push((named[i].0.clone(), prompt::to_answer(&rendered[i], &probs, self.cfg.ordinal_smoothing)));
            }
        }
        // Usage as the official engine counts it: chunks of questions, each with the state once.
        let input_tokens = questions
            .chunks(MAX_BATCH)
            .map(|c| if c.len() > 1 { state.len() + c.iter().map(|q| q.ids.len()).sum::<usize>() } else { state.len() + c[0].ids.len() })
            .sum();
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
        let done = Instant::now();
        Ok(Response { answers, input_tokens, timings: Timings { tokenize_ms: ms(t0, tokenized), forward_ms: ms(tokenized, done), total_ms: ms(t0, done) } })
    }
}
