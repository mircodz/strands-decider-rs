use std::io::Read;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use serde_json::{Value, json};
use strands_decider::prompt::{Answer, Question, py_dumps};
use strands_decider::{Decider, Request, engine::Engine, err, model};

const HELP: &str = "\
strands-decider-rs: Strands Decider 2B on the GPU (Metal, Vulkan, DX12)

USAGE:
  strands-decider-rs ask --state TEXT [--noul Q]... [--choice 'Q=a,b,c']... [--score 'Q=low,mid,high']...
  strands-decider-rs ask --request FILE      a System One request body as JSON ('-' reads stdin)
  strands-decider-rs bench [--lengths 8,64,512,2048] [--runs 5] [--profile]
  strands-decider-rs download

OPTIONS:
  -m, --model DIR   model directory (default: the platform cache; downloaded on first use)
  --json            print the System One response as JSON
  -v, --verbose     print the GPU and timings to stderr
";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    use lexopt::prelude::*;
    let mut args = lexopt::Parser::from_env();
    let cmd = match args.next()? {
        Some(Value(v)) => v.string()?,
        Some(Long("help") | Short('h')) | None => {
            print!("{HELP}");
            return Ok(());
        }
        Some(a) => return Err(a.unexpected().into()),
    };
    let (mut dir, mut state, mut request, mut json_out, mut verbose) = (None, None, None, false, false);
    let (mut nouls, mut choices, mut scores) = (Vec::new(), Vec::new(), Vec::new());
    let (mut lengths, mut runs, mut profile) = (vec![8usize, 64, 512, 2048], 5usize, false);
    while let Some(arg) = args.next()? {
        match arg {
            Short('m') | Long("model") => dir = Some(PathBuf::from(args.value()?)),
            Short('s') | Long("state") => state = Some(args.value()?.string()?),
            Long("noul") => nouls.push(args.value()?.string()?),
            Long("choice") => choices.push(args.value()?.string()?),
            Long("score") => scores.push(args.value()?.string()?),
            Short('r') | Long("request") => request = Some(args.value()?.string()?),
            Long("json") => json_out = true,
            Short('v') | Long("verbose") => verbose = true,
            Long("lengths") => lengths = args.value()?.string()?.split(',').map(|n| n.trim().parse()).collect::<Result<_, _>>()?,
            Long("runs") => runs = args.value()?.parse()?,
            Long("profile") => profile = true,
            Short('h') | Long("help") => {
                print!("{HELP}");
                return Ok(());
            }
            _ => return Err(arg.unexpected().into()),
        }
    }
    let dir = dir.unwrap_or_else(model::default_dir);
    match cmd.as_str() {
        "download" => {
            model::download(&dir)?;
            println!("{}", dir.display());
        }
        "ask" => {
            let req = match request {
                Some(path) => {
                    let mut text = String::new();
                    if path == "-" {
                        std::io::stdin().read_to_string(&mut text)?;
                    } else {
                        text = std::fs::read_to_string(&path)?;
                    }
                    Request::from_json(&serde_json::from_str(&text)?)?
                }
                None => Request { state: serde_json::Value::String(state.ok_or("--state is required (or --request)")?), questions: questions(&nouls, &choices, &scores)? },
            };
            model::download(&dir)?;
            let t = Instant::now();
            let mut decider = Decider::load(&dir)?;
            let load_ms = t.elapsed().as_secs_f64() * 1e3;
            let res = decider.evaluate(&req)?;
            if verbose {
                let t = res.timings;
                eprintln!("{}: load {load_ms:.0} ms, tokenize {:.1} ms, forward {:.1} ms, {} tokens", decider.engine.adapter, t.tokenize_ms, t.forward_ms, res.input_tokens);
            }
            if json_out {
                println!("{}", py_dumps(&res.to_json()));
            } else {
                print_answers(&res.answers);
            }
        }
        "bench" => {
            model::download(&dir)?;
            let m = model::read_manifest(&dir)?;
            let t = Instant::now();
            let mut engine = Engine::load(&dir, &m)?;
            println!("{}: weights loaded in {:.0} ms", engine.adapter, t.elapsed().as_secs_f64() * 1e3);
            println!("{:>7} {:>10} {:>10}", "tokens", "wall ms", "GPU ms");
            for &n in &lengths {
                let ids = vec![198u32; n];
                let t = Instant::now();
                engine.forward(&ids, &[n - 1], None, false, false)?;
                let first = t.elapsed().as_secs_f64() * 1e3;
                let (mut wall, mut gpu) = (Vec::new(), Vec::new());
                for _ in 0..runs {
                    let t = Instant::now();
                    let out = engine.forward(&ids, &[n - 1], None, false, true)?;
                    wall.push(t.elapsed().as_secs_f64() * 1e3);
                    gpu.extend(out.gpu_ms);
                }
                let gpu = if gpu.is_empty() { "-".to_string() } else { format!("{:.1}", median(&mut gpu)) };
                println!("{n:>7} {:>10.1} {gpu:>10}   (first run {first:.0} ms, with pipeline compilation)", median(&mut wall));
                if profile {
                    let p = engine.profile(&ids)?;
                    println!("        {}", p.iter().map(|(k, ms)| format!("{k} {ms:.1}")).collect::<Vec<_>>().join(", "));
                }
            }
        }
        _ => return Err(format!("unknown command {cmd:?}; see --help").into()),
    }
    Ok(())
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Questions from the command line, named as the official CLI names them.
fn questions(nouls: &[String], choices: &[String], scores: &[String]) -> Result<Vec<(String, Question)>, Box<dyn std::error::Error>> {
    let split = |spec: &str, flag: &str| -> Result<(String, Vec<String>), Box<dyn std::error::Error>> {
        let (q, opts) = spec.split_once('=').filter(|(_, o)| !o.is_empty()).ok_or_else(|| err(format!("--{flag} needs 'question=opt1,opt2': {spec:?}")))?;
        Ok((q.to_string(), opts.split(',').map(str::trim).filter(|o| !o.is_empty()).map(String::from).collect()))
    };
    let mut out = Vec::new();
    for (i, q) in nouls.iter().enumerate() {
        out.push((format!("noul_{i}"), Question::noul(q.as_str())));
    }
    for (i, spec) in choices.iter().enumerate() {
        let (q, opts) = split(spec, "choice")?;
        let criteria: serde_json::Map<String, Value> = opts.into_iter().map(|o| (o, json!(""))).collect();
        out.push((format!("choice_{i}"), Question::from_json(&json!({"type": "choice", "instructions": q, "criteria": criteria}))?));
    }
    for (i, spec) in scores.iter().enumerate() {
        let (q, levels) = split(spec, "score")?;
        out.push((format!("score_{i}"), Question::from_json(&json!({"type": "score", "instructions": q, "criteria": levels}))?));
    }
    if out.is_empty() {
        return Err("ask at least one --noul / --choice / --score question".into());
    }
    Ok(out)
}

/// The official CLI's answer listing.
fn print_answers(answers: &[(String, Answer)]) {
    for (name, a) in answers {
        match a {
            Answer::Noul { noul } => println!("{name} noul = {noul:.3}"),
            Answer::Choice { choice, probabilities, confidence } => {
                println!("{name} -> {choice} (confidence {confidence:.3})");
                let mut p = probabilities.clone();
                p.sort_by(|a, b| b.1.total_cmp(&a.1));
                for (opt, p) in p {
                    println!("    {opt:<24} {p:.3}");
                }
            }
            Answer::Score { score, legend, probabilities, confidence } => {
                println!("{name} score = {score:.2} (confidence {confidence:.3})");
                for ((lvl, p), (_, desc)) in probabilities.iter().zip(legend) {
                    println!("    {lvl}: {desc:<40} {p:.3}");
                }
            }
        }
    }
}
