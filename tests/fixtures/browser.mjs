// Writes tests/fixtures/browser.json: the web engine's option probabilities (engine/engine.js +
// js/head.js in Chrome, portable matmul) for the token ids in expected.json, as the GPU parity
// reference. Needs playwright-core, Google Chrome and a converted model directory:
//
//   node browser.mjs <strands-decider-web checkout> <model dir> <dir with the repository's embed_bundle.bin>
import fs from "node:fs";
import path from "node:path";
import { chromium } from "playwright-core";

const [web, model, extra] = process.argv.slice(2).map((p) => path.resolve(p));
const { startStatic } = await import(path.join(web, "web/test/static-server.mjs"));

// The browser engine reads the hosted layout; serve the converted shard unpacked (no "wire").
const assets = fs.mkdtempSync("/tmp/decider-assets-");
for (const d of ["engine-weights", "model"]) fs.mkdirSync(path.join(assets, d));
const m = JSON.parse(fs.readFileSync(path.join(model, "manifest.json")));
m.shards = m.shards.map((s, i) => ({ path: `weights.${i}.bin`, start: s.start, bytes: s.bytes }));
m.tokenizer = "tokenizer.json";
fs.writeFileSync(path.join(assets, "engine-weights/manifest.json"), JSON.stringify(m));
const link = (src, dst) => fs.symlinkSync(src, path.join(assets, dst));
m.shards.forEach((s) => link(path.join(model, s.path), `engine-weights/${s.path}`));
link(path.join(model, m.embed.rows), `engine-weights/${m.embed.rows}`);
link(path.join(extra, m.embed.bundle), `engine-weights/${m.embed.bundle}`);
for (const f of ["head.safetensors", "hobson_config.json", "tokenizer.json"]) link(path.join(model, f), `model/${f}`);

const site = await startStatic({ root: path.join(web, "web/public"), port: 8790 });
const files = await startStatic({ root: assets, port: 8791, cors: true });
const browser = await chromium.launch({ channel: "chrome", headless: true });
const page = await browser.newPage();
page.on("console", (msg) => console.log("page:", msg.text()));
await page.goto("http://127.0.0.1:8790/index.html");

const expected = JSON.parse(fs.readFileSync(new URL("./expected.json", import.meta.url)));
const jobs = expected.cases.flatMap((c) => c.question_ids.map((q, i) => {
  const ids = [...c.state_ids, ...q];
  const kind = Object.values(c.request.questions)[i].type;
  return { name: `${c.name}/${i}`, kind, ids, rows: [...c.option_idx[i].map((j) => c.state_ids.length + j), ids.length - 1] };
}));

const out = await page.evaluate(async (jobs) => {
  const { Engine } = await import("/engine/engine.js");
  const { loadHead, pointerLogits } = await import("/js/head.js");
  const base = "http://127.0.0.1:8791";
  const engine = await Engine.create(`${base}/engine-weights`);
  engine.sgm = false;  // the portable matmul, as the Rust port runs it
  const head = loadHead(await (await fetch(`${base}/model/head.safetensors`)).arrayBuffer());
  const cfg = await (await fetch(`${base}/model/hobson_config.json`)).json();
  const res = [];
  for (const j of jobs) {
    const r = await engine.forward(j.ids, j.rows);
    const n = j.rows.length - 1;
    const { probs } = pointerLogits(head, r.hidden, r.d, n, [...Array(n).keys()], cfg.temperature_by_kind[j.kind] ?? cfg.temperature);
    res.push({ name: j.name, kernel: r.kernel, probs });
  }
  return { adapter: engine.sg32 ? "subgroups" : "no subgroups", res };
}, jobs);

fs.writeFileSync(new URL("./browser.json", import.meta.url), JSON.stringify({ chrome: browser.version(), ...out }));
console.log(`${out.res.length} forwards (${out.adapter}, kernels ${[...new Set(out.res.map((r) => r.kernel))].join(" ")})`);
await browser.close();
site.close();
files.close();
fs.rmSync(assets, { recursive: true });
