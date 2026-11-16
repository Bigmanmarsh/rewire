// Statistics for the held-out results: 95% confidence intervals and paired significance tests.
// Re-runs the same deterministic benchmark as bench.js, but keeps a hit/miss per test question so
// methods can be compared question by question. Writes data/stats.json.
//   CIs:   paired bootstrap over the 400 test questions (10,000 resamples, fixed seed)
//   tests: exact two-sided McNemar test on the discordant questions
import { readFile, writeFile } from "node:fs/promises";
import { Corpus, topK, hit } from "../src/core.js";
import { Learner } from "../src/learn.js";
import { MemoryStore } from "../src/stores/memory.js";
import { BM25 } from "../src/bm25.js";
import { readSims } from "../src/data.js";

const dataDir = new URL("../data/", import.meta.url);
const load = async (f) => JSON.parse(await readFile(new URL(f, dataDir), "utf8"));
const f32 = async (f) => { const b = await readFile(new URL(f, dataDir)); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };

const docs = await load("corpus.json");
const questions = (await load("questions.json")).filter((q) => q.gold.every((g) => g !== undefined));
const corpus = new Corpus(docs, await f32("emb-docs.f32"));
const simsAll = await readSims(dataDir);
const ruleEdges = await load("rule-edges.json");
const train = questions.filter((q) => q.split === "train");
const test = questions.filter((q) => q.split === "test");
const ROUNDS = 10, per = Math.ceil(train.length / ROUNDS);
const mk = () => new Learner({ corpus, store: new MemoryStore(), questions, simsAll, ruleEdges });
const hits = async (L) => { const out = []; for (const q of test) out.push(hit(await L.run(q, L.params, L.rules, false, L.serve), q.gold) ? 1 : 0); return out; };

// ---- per-question hits for each method ----
const V = mk(); await V.reset({ withGraph: false });
const vector = await hits(V);

const bm = new BM25(docs);
const bmS = new Map(questions.map((q) => [q.i, bm.score(q.question)]));
const hybridHits = (qs, a) => qs.map((q) => {
  const s = simsAll.subarray(q.i * docs.length, (q.i + 1) * docs.length), b = bmS.get(q.i);
  const mix = new Float32Array(docs.length); for (let d = 0; d < docs.length; d++) mix[d] = s[d] + a * b[d];
  return hit(topK(mix, 5), q.gold) ? 1 : 0;
});
const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
let alpha = 0, best = -1;
for (const a of [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2]) { const r = mean(hybridHits(train, a)); if (r > best) { best = r; alpha = a; } }
const hybrid = hybridHits(test, alpha);

async function learnt(opts = {}) {
  const L = mk(); await L.reset({ withGraph: true });
  if (opts.flat) { L.params.lam = 0; L.gridOverride = { lam: [0] }; }
  const r0 = await hits(L);
  for (let r = 0; r < ROUNDS; r++) await L.round(train.slice(r * per, (r + 1) * per));
  return { r0, final: await hits(L) };
}
const rewire = await learnt();
const flat = await learnt({ flat: true });

// ---- statistics ----
let seed = 20261116;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const B = 10000, n = test.length;
const idx = Array.from({ length: B }, () => Array.from({ length: n }, () => Math.floor(rand() * n)));
const q = (arr, p) => { const s = [...arr].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const ci = (f) => { const v = idx.map((ix) => f(ix)); return [q(v, 0.025), q(v, 0.975)]; };
const accCI = (h) => ci((ix) => ix.reduce((s, i) => s + h[i], 0) / n);
const logChoose = (n, k) => { let s = 0; for (let i = 1; i <= k; i++) s += Math.log(n - k + i) - Math.log(i); return s; };
function mcnemar(a, b) { // exact two-sided binomial test on discordant pairs
  let onlyA = 0, onlyB = 0;
  for (let i = 0; i < n; i++) { if (a[i] && !b[i]) onlyA++; if (b[i] && !a[i]) onlyB++; }
  const m = onlyA + onlyB, k = Math.min(onlyA, onlyB);
  let p = 0; for (let j = 0; j <= k; j++) p += Math.exp(logChoose(m, j) - m * Math.LN2);
  return { onlyA, onlyB, p: Math.min(1, 2 * p) };
}
const diff = (a, b) => {
  const d = mean(a) - mean(b), [lo, hi] = ci((ix) => ix.reduce((s, i) => s + a[i] - b[i], 0) / n);
  return { diff: d, ci: [lo, hi], ...mcnemar(a, b) };
};

const methods = { vector, bm25: hybridHits(test, 1e6), hybrid, rewire_round0: rewire.r0, rewire_final: rewire.final, flat_final: flat.final };
const acc = Object.fromEntries(Object.entries(methods).map(([k, h]) => [k, { acc: mean(h), ci: accCI(h) }]));
const comparisons = {
  "Rewire final vs best baseline (hybrid)": diff(rewire.final, hybrid),
  "Rewire final vs vector search": diff(rewire.final, vector),
  "Learning gain: Rewire final vs Rewire round 0": diff(rewire.final, rewire.r0),
  "Graph effect: Rewire final vs same loop, traversal off": diff(rewire.final, flat.final),
  "Rewire round 0 vs best baseline (hybrid)": diff(rewire.r0, hybrid),
};

const pc = (x) => (100 * x).toFixed(1);
const pv = (p) => (p < 1e-4 ? p.toExponential(1) : p.toFixed(4));
console.log(`hybrid alpha ${alpha}; ${n} held-out questions; ${B} bootstrap resamples`);
for (const [k, v] of Object.entries(acc)) console.log(`${k.padEnd(14)} ${pc(v.acc)}%  95% CI ${pc(v.ci[0])}–${pc(v.ci[1])}`);
for (const [k, v] of Object.entries(comparisons)) console.log(`${k}: ${v.diff >= 0 ? "+" : ""}${pc(v.diff)} pts, 95% CI ${pc(v.ci[0])} to ${pc(v.ci[1])}; only first right ${v.onlyA}, only second right ${v.onlyB}; McNemar p = ${pv(v.p)}`);
await writeFile(new URL("stats.json", dataDir), JSON.stringify({ when: new Date().toISOString(), n, bootstrap: B, alpha, acc, comparisons }, null, 1));
