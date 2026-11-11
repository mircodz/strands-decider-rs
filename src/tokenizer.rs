//! The Qwen3.5 tokenizer from its `tokenizer.json`: added tokens, NFC, the Split pre-tokenizer regex,
//! then byte-level BPE. Only the encode direction, with byte offsets, as `fit_window` uses it.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use fancy_regex::Regex;
use serde::Deserialize;
use serde_json::{Value, json};
use unicode_normalization::UnicodeNormalization;

use crate::{Result, err};

pub struct Tokenizer {
    byte_id: [u32; 256],
    /// (left id, right id) -> (rank, merged id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Added tokens, matched before anything else (leftmost-longest).
    added: Vec<(String, u32)>,
    split: Regex,
}

/// Token ids and each token's byte range in the normalized text.
#[derive(Debug, Default)]
pub struct Encoding {
    pub ids: Vec<u32>,
    pub offsets: Vec<(usize, usize)>,
}

#[derive(Deserialize)]
struct File {
    added_tokens: Vec<AddedToken>,
    normalizer: Option<Value>,
    pre_tokenizer: Value,
    model: Model,
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
}

#[derive(Deserialize)]
struct Model {
    #[serde(rename = "type")]
    kind: String,
    vocab: HashMap<String, u32>,
    merges: Vec<Merge>,
    #[serde(default)]
    byte_fallback: bool,
    #[serde(default)]
    ignore_merges: bool,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Merge {
    Pair(String, String),
    Joined(String),
}

/// GPT-2's byte -> printable char map, which byte-level BPE vocabularies are written in.
fn byte_chars() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut n = 0;
    for b in 0..256u32 {
        let printable = (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
        map[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    map
}

impl Tokenizer {
    pub fn from_json(text: &str) -> Result<Tokenizer> {
        let f: File = serde_json::from_str(text).map_err(|e| err(format!("tokenizer.json: {e}")))?;
        let unsupported = |what: &str| err(format!("tokenizer.json: unsupported {what}"));
        if f.normalizer.as_ref().is_some_and(|n| n != &json!({"type": "NFC"})) {
            return Err(unsupported("normalizer"));
        }
        // Sequence[Split(regex, isolated), ByteLevel(no prefix space, no regex)], as Qwen ships it.
        let p = &f.pre_tokenizer;
        let (split, bl) = (&p["pretokenizers"][0], &p["pretokenizers"][1]);
        let pattern = split["pattern"]["Regex"].as_str();
        if p["type"] != "Sequence"
            || p["pretokenizers"].as_array().map(Vec::len) != Some(2)
            || split["type"] != "Split"
            || split["behavior"] != "Isolated"
            || split["invert"] != false
            || pattern.is_none()
            || bl["type"] != "ByteLevel"
            || bl["add_prefix_space"] != false
            || bl["use_regex"] != false
        {
            return Err(unsupported("pre_tokenizer"));
        }
        let m = f.model;
        if m.kind != "BPE" || m.byte_fallback || m.ignore_merges {
            return Err(unsupported("model"));
        }
        if f.added_tokens.iter().any(|t| t.single_word || t.lstrip || t.rstrip || t.normalized) {
            return Err(unsupported("added token options"));
        }
        let chars = byte_chars();
        let mut byte_id = [0u32; 256];
        for (b, c) in chars.iter().enumerate() {
            byte_id[b] = *m.vocab.get(&c.to_string()).ok_or_else(|| unsupported("vocabulary (missing a byte)"))?;
        }
        let mut merges = HashMap::with_capacity(m.merges.len());
        for (rank, merge) in m.merges.iter().enumerate() {
            let (a, b) = match merge {
                Merge::Pair(a, b) => (a.as_str(), b.as_str()),
                Merge::Joined(s) => s.split_once(' ').ok_or_else(|| unsupported("merge"))?,
            };
            let id = |s: &str| m.vocab.get(s).copied().ok_or_else(|| unsupported("merge (token not in vocabulary)"));
            merges.insert((id(a)?, id(b)?), (rank as u32, id(&format!("{a}{b}"))?));
        }
        let mut added: Vec<(String, u32)> = f.added_tokens.into_iter().map(|t| (t.content, t.id)).collect();
        added.sort_by_key(|(s, _)| Reverse(s.len()));
        let split = Regex::new(pattern.unwrap()).map_err(|e| err(format!("tokenizer.json: pre-tokenizer regex: {e}")))?;
        Ok(Tokenizer { byte_id, merges, added, split })
    }

    /// Encodes `text` without special tokens (Qwen adds none). Offsets are into the NFC form of the
    /// text, which is the text itself when it is already NFC.
    pub fn encode(&self, text: &str) -> Result<Encoding> {
        let mut enc = Encoding::default();
        let mut base = 0; // offset of the current piece in the normalized text
        let mut rest = text;
        while !rest.is_empty() {
            let (gap, special) = self.next_added(rest);
            let segment: String = rest[..gap].nfc().collect();
            self.encode_segment(&segment, base, &mut enc)?;
            base += segment.len();
            match special {
                Some((content, id)) => {
                    enc.ids.push(id);
                    enc.offsets.push((base, base + content.len()));
                    base += content.len();
                    rest = &rest[gap + content.len()..];
                }
                None => rest = "",
            }
        }
        Ok(enc)
    }

    /// Position of the leftmost added token in `s` (the longest one there), or the end of `s`.
    fn next_added(&self, s: &str) -> (usize, Option<(&str, u32)>) {
        for (i, _) in s.char_indices() {
            if let Some((t, id)) = self.added.iter().find(|(t, _)| s[i..].starts_with(t.as_str())) {
                return (i, Some((t.as_str(), *id)));
            }
        }
        (s.len(), None)
    }

    fn encode_segment(&self, s: &str, base: usize, enc: &mut Encoding) -> Result<()> {
        let mut last = 0;
        for m in self.split.find_iter(s) {
            let m = m.map_err(|e| err(format!("tokenizer regex: {e}")))?;
            if m.start() > last {
                self.bpe(s, last, m.start(), base, enc);
            }
            self.bpe(s, m.start(), m.end(), base, enc);
            last = m.end();
        }
        if last < s.len() {
            self.bpe(s, last, s.len(), base, enc);
        }
        Ok(())
    }

    /// Byte-level BPE of the pre-token `s[from..to]`, merging the lowest-ranked pair first (leftmost
    /// on ties), exactly as the `tokenizers` crate does.
    fn bpe(&self, s: &str, from: usize, to: usize, base: usize, enc: &mut Encoding) {
        let bytes = &s.as_bytes()[from..to];
        struct Sym {
            id: u32,
            start: usize,
            len: usize,
            prev: usize,
            next: usize,
        }
        const NONE: usize = usize::MAX;
        let n = bytes.len();
        let mut syms: Vec<Sym> = bytes
            .iter()
            .enumerate()
            .map(|(i, &b)| Sym { id: self.byte_id[b as usize], start: i, len: 1, prev: i.wrapping_sub(1), next: if i + 1 < n { i + 1 } else { NONE } })
            .collect();
        let mut queue = BinaryHeap::new();
        for i in 0..n.saturating_sub(1) {
            if let Some(&(rank, id)) = self.merges.get(&(syms[i].id, syms[i + 1].id)) {
                queue.push(Reverse((rank, i, id)));
            }
        }
        while let Some(Reverse((_, pos, new_id))) = queue.pop() {
            let next = syms[pos].next;
            // Skip entries made stale by earlier merges.
            if syms[pos].len == 0 || next == NONE || self.merges.get(&(syms[pos].id, syms[next].id)).map(|m| m.1) != Some(new_id) {
                continue;
            }
            syms[pos].id = new_id;
            syms[pos].len += syms[next].len;
            syms[pos].next = syms[next].next;
            syms[next].len = 0;
            let after = syms[pos].next;
            if after != NONE {
                syms[after].prev = pos;
            }
            let prev = syms[pos].prev;
            if prev != NONE
                && let Some(&(rank, id)) = self.merges.get(&(syms[prev].id, new_id))
            {
                queue.push(Reverse((rank, prev, id)));
            }
            if after != NONE
                && let Some(&(rank, id)) = self.merges.get(&(new_id, syms[after].id))
            {
                queue.push(Reverse((rank, pos, id)));
            }
        }
        for sym in syms.iter().filter(|sym| sym.len > 0) {
            // A token holding part of a character spans the whole character, as in `tokenizers`.
            let (mut lo, mut hi) = (from + sym.start, from + sym.start + sym.len);
            while !s.is_char_boundary(lo) {
                lo -= 1;
            }
            while !s.is_char_boundary(hi) {
                hi += 1;
            }
            enc.ids.push(sym.id);
            enc.offsets.push((base + lo, base + hi));
        }
    }
}
