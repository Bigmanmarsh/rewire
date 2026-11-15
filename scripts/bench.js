// Offline benchmark. Writes data/bench.json. Every number is on the held-out TEST split; anything
// tuned (baseline weights included) is tuned on the TRAIN split only.
//   vector    MiniLM cosine, top 5
//   hybrid    BM25 + MiniLM, mixing weight tuned on train (the stronger baseline)
//   rewire    graph + learning, per round                (main curve)
//   weights   graph, but only the traversal weights learn; no new rules
//   flat      the full learning loop with traversal switched off (rules still link question words)
//   orders    rewire's final accuracy under 5 shuffled training orders (is it a one-off?)
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
const ROUNDS = Number(process.env.ROUNDS || 10);
const per = Math.ceil(train.length / ROUNDS);
const pct = (x) => (100 * x).toFixed(1) + "%";
const fmt = (e) => `${pct(e.acc)} (bridge ${pct(e.byType.bridge ?? 0)}, comparison ${pct(e.byType.comparison ?? 0)})`;
console.log(`${docs.length} paragraphs, ${train.length} train, ${test.length} test, ${ROUNDS} rounds of ${per}`);
const mk = () => new Learner({ corpus, store: new MemoryStore(), questions, simsAll, ruleEdges });

// ---- baselines ----
const V = mk(); await V.reset({ withGraph: false });
const vector = await V.evaluate(test);
console.log(`vector (MiniLM):          ${fmt(vector)}`);

const bm = new BM25(docs);
const bmS = new Map(questions.map((q) => [q.i, bm.score(q.question)]));
const hybridAcc = (qs, a) => {
  const by = {}; let n = 0;
  for (const q of qs) {
    const s = simsAll.subarray(q.i * docs.length, (q.i + 1) * docs.length), b = bmS.get(q.i);
    const mix = new Float32Array(docs.length); for (let d = 0; d < docs.length; d++) mix[d] = s[d] + a * b[d];
    const ok = hit(topK(mix, 5), q.gold); n += ok; by[q.type] ??= [0, 0]; by[q.type][0] += ok; by[q.type][1]++;
  }
  return { acc: n / qs.length, n, total: qs.length, byType: Object.fromEntries(Object.entries(by).map(([t, [h, c]]) => [t, h / c])) };
};
let alpha = 0, bestTrain = -1;
for (const a of [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2]) { const r = hybridAcc(train, a).acc; if (r > bestTrain) { bestTrain = r; alpha = a; } }
const hybrid = { ...hybridAcc(test, alpha), alpha };
const bm25 = hybridAcc(test, 1e6);
console.log(`BM25 only:                ${fmt(bm25)}`);
console.log(`hybrid (BM25+MiniLM a=${alpha}): ${fmt(hybrid)}`);

// ---- learning curves ----
async function curve(opts = {}, order = train, quiet = false) {
  const L = mk();
  await L.reset({ withGraph: true });
  if (opts.noRules) L.diagnose = () => [];
  if (opts.flat) { L.params.lam = 0; L.gridOverride = { lam: [0] }; }
  const pts = [{ round: 0, test: await L.evaluate(test), edges: await L.store.edgeCount(), rules: [...L.rules], params: JSON.parse(JSON.stringify(L.params)), adopted: [], rejected: [] }];
  for (let r = 0; r < ROUNDS; r++) {
    const log = await L.round(order.slice(r * per, (r + 1) * per));
    const e = await L.evaluate(test);
    pts.push({ ...log, test: e });
    if (!quiet) console.log(`  r${r + 1}: ${fmt(e)} | train ${log.trainHits}/${log.seen} | edges ${log.edges} | +${log.adopted.map((a) => `${a.rule}@${a.trust}`).join(",") || "-"}`);
  }
  return pts;
}
console.log("Rewire (graph + learning):");
const rewire = await curve();
console.log("Weights only (no new rules):");
const weights = await curve({ noRules: true }, train, true);
console.log(`  r0 ${pct(weights[0].test.acc)} -> r${ROUNDS} ${pct(weights.at(-1).test.acc)}`);
console.log("Same loop, traversal off:");
const flat = await curve({ flat: true }, train, true);
console.log(`  r0 ${pct(flat[0].test.acc)} -> r${ROUNDS} ${pct(flat.at(-1).test.acc)}`);

let seed = 7;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const orders = [];
for (let k = 0; k < 5; k++) {
  const o = [...train].sort(() => rand() - 0.5);
  const c = await curve({}, o, true);
  orders.push({ first: c[1].test.acc, final: c.at(-1).test.acc, rules: c.at(-1).rules });
}
console.log(`5 shuffled training orders, final: ${orders.map((o) => pct(o.final)).join(", ")}`);

await writeFile(new URL("bench.json", dataDir), JSON.stringify({
  when: new Date().toISOString(), paragraphs: docs.length, train: train.length, test: test.length, rounds: ROUNDS, perRound: per,
  metric: "both gold paragraphs in the top 5, held-out test split",
  vector, bm25, hybrid, rewire, weights, flat, orders,
}, null, 1));
