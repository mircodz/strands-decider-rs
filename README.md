# strands-decider-rs

[Strands Decider](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21) 2B on your GPU, from Rust:
typed answers (yes/no, choice, score) about a piece of text. Metal on macOS, Vulkan on Linux and Windows. It is a
port of the WebGPU engine in [strands-decider-web](https://github.com/alxnahas/strands-decider-web).

```toml
[dependencies]
strands-decider-rs = { git = "https://github.com/mircodz/strands-decider-rs" }

# Optional: load the model faster in debug builds.
[profile.dev.package."*"]
opt-level = 3
```

```rust,no_run
use strands_decider::{Decider, Question};

fn main() -> Result<(), strands_decider::Error> {
    // Downloads the model from Hugging Face on first use (about 650 MB), then loads it from the cache.
    let mut decider = Decider::new()?;

    let answer = decider.ask(
        "Help! My payouts have been failing for 3 days!",
        Question::choice("Which team should handle this?", ["billing", "sales", "retail"]),
    )?;
    println!("{answer}"); // billing (confidence 0.810)
    Ok(())
}
```

## Questions

| Question | Answer |
|---|---|
| `Question::noul("Does this convey urgency?")` | `Answer::Noul { noul }`: P(true) |
| `Question::choice("Which team?", ["billing", "sales"])` | `Answer::Choice { choice, probabilities, confidence }` |
| `Question::score("How frustrated?", ["calm", "upset", "angry"])` | `Answer::Score { score, probabilities, confidence, .. }`: the expected level, from 0 |

Choice options can have descriptions: `[("billing", "money"), ("technical", "bugs")]`. The state and the
instructions can also be JSON (`serde_json::json!`).

## Several questions about one state

`ask_all` encodes the state once, then answers each question from it:

```rust,no_run
# use strands_decider::{Decider, Question};
# let mut decider = Decider::new()?;
let answers = decider.ask_all(
    "Help! My payouts have been failing for 3 days!",
    [
        Question::noul("Does this convey urgency?"),
        Question::choice("Which team should handle this?", ["billing", "technical", "sales"]),
        Question::score("How frustrated is the writer?", ["calm", "frustrated", "very angry"]),
    ],
)?;
# Ok::<(), strands_decider::Error>(())
```

## More

- `Decider::load(dir)` loads a model directory (see `model::download`) instead of the cached one.
- [docs/README.md](docs/README.md): the command line tool, results, how the engine works, tests.
