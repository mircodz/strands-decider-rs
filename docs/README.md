# strands-decider-rs in detail

The command line tool, results, how the engine works, and tests. For the library, see the [README](../README.md).

[Strands Decider](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21) 2B on your GPU from Rust. It runs
the compressed v21 build that [strands-decider-web](https://github.com/alxnahas/strands-decider-web) hosts, with one set of WGSL kernels on Metal
(macOS), Vulkan (Linux, Windows) and DX12 (Windows) through [wgpu](https://wgpu.rs). It is a port of the web
repository's hand-written WebGPU engine. Prompts, tokenization and answers follow the official
[`strands-decider`](https://github.com/strands-agents) package: noul, choice and score questions, the same prompt
text, the same confidence formulas.

```bash
cargo build --release
./target/release/strands-decider-rs ask --state "Help! My payouts have been failing for 3 days!" \
  --choice "Which team should handle this?=billing,sales,retail"
```

```
choice_0 -> billing (confidence 0.810)
    billing                  0.874
    sales                    0.065
    retail                   0.062
```

The first run downloads the model (about 650 MB, with Hugging Face's `hf-hub` client) into the platform cache
directory, at a pinned revision of
[alxnahas/strands-decider-v21-webgpu](https://huggingface.co/alxnahas/strands-decider-v21-webgpu), and converts it
once. `HF_ENDPOINT` points it at a mirror. Later runs load it in about 0.25 s; the command above takes 0.29 s end to end on an M4 Pro.

| Command | |
|---|---|
| `ask --state TEXT [--noul Q] [--choice 'Q=a,b'] [--score 'Q=low,high']` | flags and output of the official `strands-decider ask` |
| `ask --request FILE` | a `POST /v1/systemone` body (`-` reads stdin); structured states and criteria work |
| `ask ... --json` | the System One response |
| `bench [--lengths 8,64,512,2048] [--runs 5] [--profile]` | forward latency; `--profile` adds GPU time per kernel |
| `download` | fetch and convert the model without asking anything |

`-m DIR` keeps the model somewhere else; `-v` prints the GPU in use and timings. Several questions about one state
share one encoding of it: the state is forwarded once and each question continues from a snapshot of the recurrent
state, conv history and K/V, as in the official server.

## Results (Apple M4 Pro)

**Answers.** On 39 prompts, the engine was compared with the web engine running in Chrome on the same weights and token
ids. The prompts are the web repository's 34 fixture and benchmark prompts, plus requests with structured content,
eight languages and non-NFC text. Both engines give the same top answer on all 39, and the largest difference in an
option probability is 0.0005. The tokens are identical to the official package's (rendering, Hugging Face
`tokenizers`, option positions), as are the ids of 21 edge-case strings, and their offsets where NFC leaves the
text unchanged.

**Speed.** GPU time per forward in ms, for the hosted build. The two Chrome columns are the web engine on the same
machine:

| tokens | 8 | 16 | 68 | 512 | 2048 |
|---|---|---|---|---|---|
| this engine | 8.3 | 8.7 | 28.3 | 165 | 689 |
| web engine, portable matmul (what stock Chrome runs) | 7.9 | 14.0 | 37.8 | 206 | 838 |
| web engine, subgroup-matrix matmul (`--enable-unsafe-webgpu`) | 7.9 | 8.9 | 29.6 | 163 | 665 |

Wall time adds about 0.5 ms (submit and readback).

## How it differs from the web engine

- **Kernels.** The WGSL of `engine/wgsl.js` and the portable matmuls of `gemmbench/kernels.js`, with every
  fixed-count loop over registers unrolled. naga, wgpu's shader compiler, emits WGSL loops as `while (true)`, which
  Metal does not unroll; arrays indexed in them left registers, which made the GEMV and the gated-delta recurrence
  2 to 3 times slower than in Chrome until unrolled. Shaders are built without bounds checks and without workgroup
  memory zeroing, which a browser always adds; `STRANDS_DECIDER_CHECKED=1` restores the checks (same answers).
- **Matmul.** Where the GPU has 8×8 cooperative matrices with f16 inputs and an f32 accumulator (Apple GPUs, through
  wgpu's experimental cooperative-matrix feature), a port of the web engine's subgroup-matrix kernel does the matmuls:
  16-row tiles up to 80 tokens, then 32-row tiles. It accumulates in f32 where Chrome's accumulates in f16. Elsewhere,
  the web engine's portable kernels and tile table apply: the multi-row GEMV up to 20 tokens, then tiled.
- **Embedding.** The whole int3 table is on the GPU (222 MB); the browser's on-demand rows are not needed natively.
- **Model files.** The packed shard is unpacked into the GPU layout once, at download time, not on every load.
- **Prompts.** As the official package does, a prompt over 4,096 tokens is refused; the web port truncates instead.
  A `null` noul criterion renders as empty, also as in the official package.

## Platforms

The engine needs a GPU with 16-bit float shaders (`shader-f16`). It picks one by itself (discrete before integrated,
Metal or Vulkan before DX12); `WGPU_ADAPTER_NAME` and `WGPU_BACKEND` override the choice. The kernels adapt to the
GPU:

| GPU | Kernels |
|---|---|
| Apple | subgroups, and cooperative-matrix matmuls |
| 32-wide subgroups (NVIDIA) | subgroups, portable matmuls |
| other subgroup widths (AMD, Intel, mobile) | subgroup-free kernels (reductions through workgroup memory), portable matmuls |

Only the Apple M4 Pro (Metal) has run the engine, on all three kernel paths (`STRANDS_DECIDER_COOP=0` and
`STRANDS_DECIDER_SUBGROUPS=0` force the other two). The crate type-checks for Linux and Windows (x86-64 and ARM64),
but the Vulkan and DX12 backends have not been run. DX12 has 16-bit floats only with `dxcompiler.dll` on the path;
without it, the engine uses the Vulkan adapter, which GPU drivers on Windows provide.

## Dependencies

wgpu; `hf-hub`, Hugging Face's client, for the download (on ureq and rustls); and small crates for parsing:
`serde`/`serde_json` (manifest, tokenizer, requests), `flate2` (the gzipped downloads), `fancy-regex` and
`unicode-normalization` (the tokenizer's pre-tokenizer regex and NFC), and `lexopt` (command line). `half` and
`bytemuck` were already in wgpu's tree. That makes 130 crates on macOS: 56 for wgpu, 55 more for `hf-hub`. Byte-level
BPE, the pointer head and the model's file formats are implemented here.

## Tests

```bash
cargo test --release    # parity tests need the model (run `download` first) and a GPU
```

Unit tests validate every kernel in naga, with and without subgroups, and check Python's JSON and float formatting.
`tests/parity.rs` checks rendering and tokenization against the official package and Hugging Face `tokenizers`.
`tests/engine_parity.rs` checks the engine against the web engine, and the prefix cache against full forwards. The
expected values are in `tests/fixtures`:

```bash
# expected.json: needs tokenizers and pydantic, and checkouts of strands-decider and strands-decider-web
python tests/fixtures/make.py <strands-decider> <model dir> <strands-decider-web>
# browser.json: needs Google Chrome; the web engine needs embed_bundle.bin from the model repository
cd tests/fixtures && npm i --no-save playwright-core
node browser.mjs <strands-decider-web> <model dir> <dir with embed_bundle.bin>
```

## License

Apache-2.0, as strands-decider-web. The model weights keep their Apache-2.0 license.
