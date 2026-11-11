//! Prompt rendering and answer reading, ported from the official package's `prompting.py` and
//! `schema.py`: a question becomes numbered option lines, and the head's distribution over those
//! options is read back as a noul probability, a choice or a score.

use serde_json::{Value, json};
use unicode_normalization::UnicodeNormalization;

use crate::{Result, err};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Noul,
    Choice,
    Score,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Noul => "noul",
            Kind::Choice => "choice",
            Kind::Score => "score",
        }
    }
}

/// A question about a state: its instructions and its options as (label, description) pairs.
#[derive(Clone, Debug)]
pub struct Question {
    pub kind: Kind,
    pub instructions: Value,
    pub options: Vec<(String, Value)>,
}

const NOUL_DEFAULT: [(&str, &str); 2] =
    [("false", "the statement does not hold for this state"), ("true", "the statement holds for this state")];

fn is_content(v: &Value) -> bool {
    v.is_string() || v.is_object() || v.is_array()
}

/// A choice option: a label, or a (label, description) pair.
pub trait IntoChoice {
    fn into_choice(self) -> (String, Value);
}

impl IntoChoice for &str {
    fn into_choice(self) -> (String, Value) {
        (self.into(), Value::Null)
    }
}

impl IntoChoice for String {
    fn into_choice(self) -> (String, Value) {
        (self, Value::Null)
    }
}

impl<N: Into<String>, D: Into<Value>> IntoChoice for (N, D) {
    fn into_choice(self) -> (String, Value) {
        (self.0.into(), self.1.into())
    }
}

impl Question {
    /// Yes or no: is the statement true of the state? Answers P(true).
    pub fn noul(instructions: impl Into<Value>) -> Question {
        let options = NOUL_DEFAULT.iter().map(|(l, d)| (l.to_string(), json!(d))).collect();
        Question { kind: Kind::Noul, instructions: instructions.into(), options }
    }

    /// One of 2 to 255 options, each a label or a (label, description) pair.
    pub fn choice<O: IntoChoice>(instructions: impl Into<Value>, options: impl IntoIterator<Item = O>) -> Question {
        Question { kind: Kind::Choice, instructions: instructions.into(), options: options.into_iter().map(O::into_choice).collect() }
    }

    /// A level on an ordered scale of 2 to 10 descriptions, lowest first. Answers the expected level.
    pub fn score<S: Into<String>>(instructions: impl Into<Value>, levels: impl IntoIterator<Item = S>) -> Question {
        let options = levels.into_iter().enumerate().map(|(i, d)| (i.to_string(), Value::String(d.into()))).collect();
        Question { kind: Kind::Score, instructions: instructions.into(), options }
    }

    /// Checks the question as the official schema does.
    pub fn validate(&self) -> Result<()> {
        if !is_content(&self.instructions) {
            return Err(err("question instructions must be a string, object or array"));
        }
        if self.options.iter().any(|(_, d)| !d.is_null() && !is_content(d)) {
            return Err(err("option descriptions must be strings, objects, arrays or null"));
        }
        let n = self.options.len();
        match self.kind {
            Kind::Noul if self.options.iter().map(|(l, _)| l.as_str()).ne(["false", "true"]) => Err(err("noul options must be false, true")),
            Kind::Choice if !(2..=255).contains(&n) => Err(err("choice requires between 2 and 255 options")),
            Kind::Choice if (1..n).any(|i| self.options[..i].iter().any(|(l, _)| *l == self.options[i].0)) => Err(err("choice options must be unique")),
            Kind::Score if !(2..=10).contains(&n) => Err(err("score requires between 2 and 10 levels")),
            Kind::Score if self.options.iter().any(|(_, d)| !d.is_string()) => Err(err("score levels must be strings")),
            _ => Ok(()),
        }
    }

    /// A question as the System One API takes it (`{"type", "instructions", "criteria"}`), validated
    /// as the official schema does.
    pub fn from_json(v: &Value) -> Result<Question> {
        let o = v.as_object().ok_or_else(|| err("a question must be an object"))?;
        let criteria = o.get("criteria").filter(|c| !c.is_null());
        let kind = match o.get("type").map(|t| t.as_str()) {
            Some(Some("noul")) => Kind::Noul,
            Some(Some("choice")) => Kind::Choice,
            Some(Some("score")) => Kind::Score,
            None => match criteria {
                None => Kind::Noul,
                Some(Value::Array(_)) => Kind::Score,
                Some(Value::Object(c)) if c.keys().all(|k| k == "true" || k == "false") => Kind::Noul,
                Some(_) => Kind::Choice,
            },
            _ => return Err(err("question type must be noul, choice or score")),
        };
        let instructions = o.get("instructions").cloned().unwrap_or(Value::Null);
        let q = match (kind, criteria) {
            (Kind::Noul, None) => Question::noul(instructions),
            (Kind::Noul, Some(Value::Object(c))) if c.keys().all(|k| k == "true" || k == "false") => {
                let mut q = Question::noul(instructions);
                for (label, desc) in q.options.iter_mut() {
                    if let Some(d) = c.get(label.as_str()) {
                        *desc = d.clone();
                    }
                }
                q
            }
            (Kind::Noul, _) => return Err(err("noul criteria keys must be a subset of {'true', 'false'}")),
            (Kind::Choice, Some(Value::Object(c))) => Question::choice(instructions, c.iter().map(|(k, d)| (k.clone(), d.clone()))),
            (Kind::Choice, _) => return Err(err("choice criteria must map option names to descriptions")),
            (Kind::Score, Some(Value::Array(c))) if c.iter().all(Value::is_string) => Question::score(instructions, c.iter().map(|d| d.as_str().unwrap())),
            (Kind::Score, _) => return Err(err("score criteria must be a list of level descriptions")),
        };
        q.validate()?;
        Ok(q)
    }
}

/// A question as prompt text, with what is needed to read the answer back.
#[derive(Clone, Debug)]
pub struct Rendered {
    pub text: String,
    pub kind: Kind,
    pub labels: Vec<String>,
    pub descriptions: Vec<String>,
    /// Byte range of each option's line in `text`.
    pub spans: Vec<(usize, usize)>,
}

/// Python's `str.isspace()`.
fn py_space(c: char) -> bool {
    matches!(c, '\t'..='\r' | '\x1c'..='\x20' | '\u{85}' | '\u{a0}' | '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}')
}

/// `render_content`: strings stripped, structured content as Python's indented JSON.
pub fn render_content(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.trim_matches(py_space).to_string(),
        v => py_dumps(v),
    }
}

pub fn render_state(state: &Value) -> String {
    format!("<state>\n{}\n</state>\n", render_content(state))
}

/// `render_question`. The pieces are NFC-normalized first, as the tokenizer would normalize the whole
/// text, so that byte offsets into the tokenized text are offsets into `text`.
pub fn render_question(q: &Question) -> Rendered {
    let nfc = |s: &str| s.nfc().collect::<String>();
    let header = match q.kind {
        Kind::Noul => "Decide whether the statement is true of the state.",
        Kind::Choice => "Select exactly one option.",
        Kind::Score => "Rate the state against the ordered levels below (lowest first).",
    };
    let mut text = format!("<question type=\"{}\">\n{header}\n{}\n<options>\n", q.kind.name(), nfc(&render_content(&q.instructions)));
    let mut spans = Vec::new();
    for (i, (name, desc)) in q.options.iter().enumerate() {
        let desc = render_content(desc).split(py_space).filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" ");
        let mut line = format!("{}. {}", i + 1, nfc(name));
        if !desc.is_empty() {
            line += &format!(" \u{2014} {}", nfc(&desc));
        }
        spans.push((text.len(), text.len() + line.len()));
        text += &line;
        text.push('\n');
    }
    text += "</options>\n</question>\n<answer>";
    Rendered {
        text,
        kind: q.kind,
        labels: q.options.iter().map(|(n, _)| n.clone()).collect(),
        descriptions: q.options.iter().map(|(_, d)| render_content(d)).collect(),
        spans,
    }
}

/// `option_token_index`: the last token that lies inside each option's line.
pub fn option_token_index(offsets: &[(usize, usize)], spans: &[(usize, usize)]) -> Result<Vec<usize>> {
    spans
        .iter()
        .map(|&(a, b)| {
            offsets
                .iter()
                .rposition(|&(lo, hi)| hi > lo && lo >= a && hi <= b)
                .ok_or_else(|| err(format!("option span ({a},{b}) has no tokens left; the prompt was truncated through its option list")))
        })
        .collect()
}

#[derive(Clone, Debug)]
pub enum Answer {
    Noul { noul: f64 },
    Choice { choice: String, probabilities: Vec<(String, f64)>, confidence: f64 },
    Score { score: f64, legend: Vec<(String, String)>, probabilities: Vec<(String, f64)>, confidence: f64 },
}

/// Python's `round(x, 4)`.
fn round4(x: f64) -> f64 {
    format!("{x:.4}").parse().unwrap()
}

/// `derive_confidence`: normalized max probability.
pub fn choice_confidence(p: &[f64]) -> f64 {
    let n = p.len() as f64;
    if p.len() <= 1 {
        return 1.0;
    }
    let max = p.iter().cloned().fold(f64::MIN, f64::max);
    ((n * max - 1.0) / (n - 1.0)).clamp(0.0, 1.0)
}

/// `derive_score_confidence`: how tightly the mass clusters on the ordered scale.
pub fn score_confidence(p: &[f64], ordinal_smoothing: f64) -> f64 {
    let n = p.len();
    if n <= 1 {
        return 1.0;
    }
    let total: f64 = p.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let p: Vec<f64> = p.iter().map(|x| x / total).collect();
    let mean: f64 = p.iter().enumerate().map(|(i, pi)| i as f64 * pi).sum();
    let sigma = p.iter().enumerate().map(|(i, pi)| pi * (i as f64 - mean).powi(2)).sum::<f64>().sqrt();
    let max = (n - 1) as f64 / 2.0;
    let floor = if ordinal_smoothing > 0.0 { ordinal_smoothing.sqrt() } else { 0.0 };
    if max <= floor {
        return if sigma <= floor { 1.0 } else { 0.0 };
    }
    ((max - sigma) / (max - floor)).clamp(0.0, 1.0)
}

/// `to_answer`: read the option distribution back as the question's answer type.
pub fn to_answer(rq: &Rendered, probs: &[f64], ordinal_smoothing: f64) -> Answer {
    match rq.kind {
        Kind::Noul => Answer::Noul { noul: round4(probs[rq.labels.iter().position(|l| l == "true").unwrap()]) },
        Kind::Choice => {
            let best = (0..probs.len()).fold(0, |b, i| if probs[i] > probs[b] { i } else { b });
            Answer::Choice {
                choice: rq.labels[best].clone(),
                probabilities: rq.labels.iter().zip(probs).map(|(l, p)| (l.clone(), round4(*p))).collect(),
                confidence: round4(choice_confidence(probs)),
            }
        }
        Kind::Score => Answer::Score {
            score: round4(probs.iter().enumerate().map(|(i, p)| i as f64 * p).sum()),
            legend: rq.labels.iter().cloned().zip(rq.descriptions.iter().cloned()).collect(),
            probabilities: rq.labels.iter().zip(probs).map(|(l, p)| (l.clone(), round4(*p))).collect(),
            confidence: round4(score_confidence(probs, ordinal_smoothing)),
        },
    }
}

impl Answer {
    pub fn to_json(&self) -> Value {
        let probs = |p: &[(String, f64)]| Value::Object(p.iter().map(|(k, v)| (k.clone(), json!(v))).collect());
        match self {
            Answer::Noul { noul } => json!({"type": "noul", "noul": noul}),
            Answer::Choice { choice, probabilities, confidence } => {
                json!({"type": "choice", "choice": choice, "probabilities": probs(probabilities), "confidence": confidence})
            }
            Answer::Score { score, legend, probabilities, confidence } => json!({
                "type": "score",
                "score": score,
                "legend": Value::Object(legend.iter().map(|(k, v)| (k.clone(), json!(v))).collect()),
                "probabilities": probs(probabilities),
                "confidence": confidence,
            }),
        }
    }
}

/// One line: `P(true) = 0.833`, `billing (confidence 0.810)` or `1.20 (confidence 0.544)`.
impl std::fmt::Display for Answer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Answer::Noul { noul } => write!(f, "P(true) = {noul:.3}"),
            Answer::Choice { choice, confidence, .. } => write!(f, "{choice} (confidence {confidence:.3})"),
            Answer::Score { score, confidence, .. } => write!(f, "{score:.2} (confidence {confidence:.3})"),
        }
    }
}

/// Python's `json.dumps(v, indent=2, ensure_ascii=False)`.
pub fn py_dumps(v: &Value) -> String {
    fn dump(v: &Value, level: usize, out: &mut String) {
        let pad = |out: &mut String, l: usize| out.extend(std::iter::repeat_n(' ', 2 * l));
        match v {
            Value::Number(n) if n.is_f64() => out.push_str(&py_float(n.as_f64().unwrap())),
            Value::Array(a) if !a.is_empty() => {
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    out.push_str(if i == 0 { "\n" } else { ",\n" });
                    pad(out, level + 1);
                    dump(x, level + 1, out);
                }
                out.push('\n');
                pad(out, level);
                out.push(']');
            }
            Value::Object(o) if !o.is_empty() => {
                out.push('{');
                for (i, (k, x)) in o.iter().enumerate() {
                    out.push_str(if i == 0 { "\n" } else { ",\n" });
                    pad(out, level + 1);
                    out.push_str(&Value::from(k.as_str()).to_string());
                    out.push_str(": ");
                    dump(x, level + 1, out);
                }
                out.push('\n');
                pad(out, level);
                out.push('}');
            }
            // null, booleans, integers, strings (serde_json escapes as Python does) and empty containers
            v => out.push_str(&v.to_string()),
        }
    }
    let mut out = String::new();
    dump(v, 0, &mut out);
    out
}

/// Python's `repr(float)`: shortest round-trip digits, exponent form below 1e-4 and from 1e16.
pub fn py_float(x: f64) -> String {
    if !x.is_finite() {
        return if x.is_nan() { "NaN" } else if x > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if x == 0.0 {
        return format!("{sign}0.0");
    }
    let e = format!("{:e}", x.abs());
    let (mant, exp) = e.split_once('e').unwrap();
    let digits = mant.replace('.', "");
    let (n, point) = (digits.len() as i32, exp.parse::<i32>().unwrap() + 1);
    let body = if point <= -4 || point > 16 {
        let frac = if n > 1 { format!(".{}", &digits[1..]) } else { String::new() };
        format!("{}{frac}e{}{:02}", &digits[..1], if point > 0 { '+' } else { '-' }, (point - 1).abs())
    } else if point <= 0 {
        format!("0.{}{digits}", "0".repeat(-point as usize))
    } else if point >= n {
        format!("{digits}{}.0", "0".repeat((point - n) as usize))
    } else {
        format!("{}.{}", &digits[..point as usize], &digits[point as usize..])
    };
    format!("{sign}{body}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_formatting() {
        for (x, s) in [(0.1, "0.1"), (1.0, "1.0"), (0.0001, "0.0001"), (1e-5, "1e-05"), (1.2345678901234568e17, "1.2345678901234568e+17"), (1e15, "1000000000000000.0"), (-2.5e-7, "-2.5e-07"), (0.8442, "0.8442")] {
            assert_eq!(py_float(x), s);
        }
        let v: Value = serde_json::from_str(r#"{"a": [1, 2.5, {"b": null}], "c": "é\n\u0001\"", "d": {}, "e": [], "g": 1e-5, "i": 100.0, "a": 7}"#).unwrap();
        assert_eq!(
            py_dumps(&v),
            "{\n  \"a\": 7,\n  \"c\": \"é\\n\\u0001\\\"\",\n  \"d\": {},\n  \"e\": [],\n  \"g\": 1e-05,\n  \"i\": 100.0\n}"
        );
    }

    #[test]
    fn renders_like_python() {
        let q = Question::from_json(&json!({"type": "choice", "instructions": "  Route it. ", "criteria": {"billing": "money\n matters", "sales": null}})).unwrap();
        let r = render_question(&q);
        assert_eq!(r.text, "<question type=\"choice\">\nSelect exactly one option.\nRoute it.\n<options>\n1. billing \u{2014} money matters\n2. sales\n</options>\n</question>\n<answer>");
        assert_eq!(&r.text[r.spans[1].0..r.spans[1].1], "2. sales");
        assert_eq!(choice_confidence(&[0.5, 0.5]), 0.0);
        assert!((score_confidence(&[0.0, 1.0, 0.0], 0.1) - 1.0).abs() < 1e-12);
    }
}
