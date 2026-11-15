// Rewire live server. Serves the UI and an API over one shared learning session.
// STORE=hydra (default in the container) traverses the real HydraDB engine; STORE=memory for local dev.
import express from "express";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { Corpus, RULES, hit } from "./core.js";
import { readSims } from "./data.js";
import { Learner } from "./learn.js";
import { MemoryStore } from "./stores/memory.js";
import { HydraStore } from "./stores/hydradb.js";

const dataDir = new URL("../data/", import.meta.url);
const json = async (f) => JSON.parse(await readFile(new URL(f, dataDir), "utf8"));
const f32 = async (f) => { const b = await readFile(new URL(f, dataDir)); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };

const PORT = Number(process.env.PORT || 3000);
const ROUND_SIZE = Number(process.env.ROUND_SIZE || 20);
const STORE = process.env.STORE || "memory";

const docs = await json("corpus.json");
const questions = (await json("questions.json")).filter((q) => q.gold.every((g) => g !== undefined));
const corpus = new Corpus(docs, null); // live server uses precomputed similarities, no model at runtime
const simsAll = await readSims(dataDir);
const ruleEdges = await json("rule-edges.json");
const bench = await json("bench.json").catch(() => null);
const train = questions.filter((q) => q.split === "train");
const test = questions.filter((q) => q.split === "test");

// The port opens first (the page shows "booting"), then the graph is loaded into HydraDB.
let serve, L, vector, ready = false, bootError = null;
async function boot() {
  serve = new MemoryStore();
  if (STORE === "hydra") {
    const token = (await readFile(process.env.HYDRA_TOKEN_FILE || "/tmp/hydra/secrets/auth-token", "utf8")).trim();
    serve = new HydraStore({ url: process.env.HYDRA_URL || "http://127.0.0.1:8443", token });
    await waitFor(serve);
    const t0 = Date.now();
    await serve.loadDocs(docs);
    console.log(`[rewire] loaded ${docs.length} Doc nodes into HydraDB in ${Date.now() - t0}ms`);
  }
  L = new Learner({ corpus, store: new MemoryStore(), serve, questions, simsAll, ruleEdges });
  const V = new Learner({ corpus, store: new MemoryStore(), questions, simsAll, ruleEdges });
  await V.reset({ withGraph: false });
  vector = await V.evaluate(test);
  await reset();
  ready = true;
}

let curve = [];
let busy = Promise.resolve();
const locked = (fn) => (busy = busy.then(fn, fn));

async function reset() {
  const t0 = Date.now();
  await L.reset({ withGraph: true });
  curve = [{ round: 0, test: await L.evaluate(test), edges: await L.serve.edgeCount(), rules: [...L.rules], adopted: [], rejected: [], params: JSON.parse(JSON.stringify(L.params)) }];
  console.log(`[rewire] reset in ${Date.now() - t0}ms, round 0 test ${(curve[0].test.acc * 100).toFixed(1)}%`);
}
const view = (id) => ({ id, title: docs[id].title, text: docs[id].text.slice(0, 280) });

// Everything the UI needs to draw one question: top 5, why each is there, and the subgraph.
async function explainQ(q) {
  const top = await L.run(q, L.params, L.rules, true, L.serve);
  const ids = new Set([...top.map((t) => t.id), ...q.gold]);
  for (const t of top) if (t.via) ids.add(t.via.via);
  const nb = await L.serve.neighbors([...ids]);
  const links = [];
  for (const [a, list] of nb.out) for (const [b, rules] of list) if (ids.has(b)) links.push([a, b, [...rules]]);
  return {
    q: { i: q.i, question: q.question, answer: q.answer, type: q.type, split: q.split, gold: q.gold },
    hit: hit(top.map((t) => t.id), q.gold),
    top: top.map((t) => ({ ...view(t.id), sim: t.sim, score: t.score, via: t.via, linked: t.linked, gold: q.gold.includes(t.id) })),
    nodes: [...ids].map(view),
    links,
  };
}

function state() {
  return {
    store: STORE === "hydra" ? "HydraDB 0.4.0 (open-source engine, same container)" : "in-memory reference store",
    storeShort: STORE === "hydra" ? "HydraDB" : "the in-memory store",
    paragraphs: docs.length, train: train.length, test: test.length, roundSize: ROUND_SIZE,
    seen: L.seen.length, done: L.seen.length >= train.length,
    rules: L.rules, params: L.params,
    ruleInfo: Object.fromEntries(Object.entries(RULES).map(([k, r]) => [k, { label: r.label, desc: r.desc }])),
    curve, vector, bench,
    queries: serve.queries ?? null,
  };
}

const app = express();
app.use(express.json({ limit: "20kb" }));
app.use(express.static(fileURLToPath(new URL("../public/", import.meta.url))));
app.get("/healthz", (_, res) => res.send("ok"));
// Fixed dialect probe on throwaway :Probe nodes (no user input reaches these queries).
app.get("/probe", async (req, res) => {
  if (!(serve instanceof HydraStore)) return res.json({ error: "not on HydraDB" });
  const R = [{ id: 900001, t: "a" }, { id: 900002, t: "b" }, { id: 900003, t: "c" }];
  const E = [{ a: 900001, b: 900002, r: "exact" }, { a: 900002, b: 900003, r: "paren" }];
  const tests = [
    ["return id", "MATCH (n:Doc) RETURN n.id AS id LIMIT 2"],
    ["unwind merge nodes", "UNWIND $rows AS row MERGE (n {id: row.id}) SET n:Probe, n.t = row.t", { rows: R }],
    ["unwind merge nodes (no prop)", "UNWIND $rows AS row MERGE (n {id: row.id}) SET n:Probe", { rows: R }],
    ["list probe nodes", "MATCH (n:Probe) RETURN n.id AS id"],
    ["edge unwind id", "UNWIND $rows AS row MATCH (a:Probe {id: row.a}), (b:Probe {id: row.b}) MERGE (a)-[e:MENTIONS {id: row.id}]->(b) SET e.rule = row.r", { rows: E.map((e, k) => ({ ...e, id: 990000 + k })) }],
    ["edge unwind id again (idempotent)", "UNWIND $rows AS row MATCH (a:Probe {id: row.a}), (b:Probe {id: row.b}) MERGE (a)-[e:MENTIONS {id: row.id}]->(b) SET e.rule = row.r", { rows: E.map((e, k) => ({ ...e, id: 990000 + k })) }],
    ["out IN", "MATCH (a:Probe)-[r:MENTIONS]->(b:Probe) WHERE a.id IN $ids RETURN a.id AS a, b.id AS b, r.rule AS rule", { ids: [900001, 900002] }],
    ["in IN", "MATCH (a:Probe)<-[r:MENTIONS]-(b:Probe) WHERE a.id IN $ids RETURN a.id AS a, b.id AS b, r.rule AS rule", { ids: [900002, 900003] }],
    ["list edges", "MATCH (a:Probe)-[r:MENTIONS]->(b:Probe) RETURN a.id AS a, b.id AS b, r.rule AS rule"],
    ["doc edges in db", "MATCH (a:Doc)-[r:MENTIONS]->(b:Doc) RETURN a.id AS a, b.id AS b, r.rule AS rule LIMIT 3"],
    ["delete by rule", "MATCH (a:Probe)-[r:MENTIONS]->(b:Probe) WHERE r.rule = $rule DELETE r", { rule: "paren" }],
    ["count after delete", "MATCH (a:Probe)-[r:MENTIONS]->(b:Probe) RETURN r.rule AS rule"],
    ["delete all probe edges", "MATCH (a:Probe)-[r:MENTIONS]->(b:Probe) DELETE r"],
  ];
  const out = [];
  for (const [name, q, p] of tests) {
    try { const rows = await serve.q(q, p || {}); out.push({ name, ok: true, rows: rows.slice(0, 6) }); }
    catch (e) { out.push({ name, ok: false, error: e.message.slice(0, 220) }); }
  }
  res.json(out);
});
app.use("/api", (req, res, next) => (ready ? next() : res.status(503).json({ error: bootError ? `boot failed: ${bootError}` : "booting: loading the graph into HydraDB" })));
app.get("/api/state", (_, res) => res.json(state()));

// Training questions the graph currently gets wrong and has not learned from yet: the demo picks one.
app.get("/api/misses", (req, res) => locked(async () => {
  const pool = train.slice(L.seen.length);
  const out = [];
  for (const q of pool) {
    if (out.length >= 12) break;
    if (!hit(await L.run(q, L.params, L.rules, false, L.serve), q.gold)) {
      const d = L.diagnose(q);
      if (d.length) out.push({ i: q.i, question: q.question, type: q.type, fixable: d.map((x) => x.rule) });
    }
  }
  res.json(out);
}).catch((e) => res.status(500).json({ error: e.message })));

app.get("/api/q/:i", (req, res) => locked(async () => {
  const q = questions[Number(req.params.i)];
  if (!q) return res.status(404).json({ error: "no such question" });
  res.json(await explainQ(q));
}).catch((e) => res.status(500).json({ error: e.message })));

// Learn from this miss: diagnose it, then run one learning round over it plus the next training batch.
// Learn from this miss: run one learning round over it plus the next training batch, then pick a
// showcase: a training question this round's change actually fixed (the clicked one when it was).
app.post("/api/learn", (req, res) => locked(async () => {
  if (L.seen.length >= train.length) return res.status(409).json({ error: "all training questions used; reset to replay" });
  const focus = questions[Number(req.body?.i)];
  const useFocus = focus && focus.split === "train" && !L.seen.includes(focus);
  const rest = train.filter((q) => !L.seen.includes(q) && q !== focus).slice(0, ROUND_SIZE - (useFocus ? 1 : 0));
  const batch = useFocus ? [focus, ...rest] : rest;
  // snapshot every candidate before learning (scratch graph, cheap): its top 5 and why it missed
  const pre = new Map();
  for (const q of [...L.seen, ...batch]) {
    const top = await L.run(q, L.params, L.rules, true);
    if (!hit(top.map((t) => t.id), q.gold)) pre.set(q.i, { top: top.map((t) => ({ ...view(t.id), gold: q.gold.includes(t.id) })), diagnosis: L.diagnose(q) });
  }
  const t0 = Date.now();
  const log = await L.round(batch);
  const t1 = Date.now();
  const point = { ...log, paramsBefore: curve.at(-1).params, test: await L.evaluate(test), learnMs: t1 - t0, evalMs: Date.now() - t1 };
  curve.push(point);
  console.log(`[rewire] round ${log.round}: adopted ${log.adopted.map((a) => a.rule).join(",") || "-"}, test ${(point.test.acc * 100).toFixed(1)}%, ${t1 - t0}ms learn`);

  const learned = new Set(log.adopted.map((x) => x.rule));
  let show = null, fixedCount = 0;
  const order = useFocus ? [focus, ...[...pre.keys()].map((i) => questions[i]).filter((q) => q !== focus)] : [...pre.keys()].map((i) => questions[i]);
  for (const q of order) {
    if (!pre.has(q.i)) continue;
    const top = await L.run(q, L.params, L.rules, true);
    if (!hit(top.map((t) => t.id), q.gold)) continue;
    fixedCount++;
    const viaNew = top.some((t) => t.via?.rules?.some((r) => learned.has(r)));
    const score = (viaNew ? 2 : 0) + (q === focus ? 1 : 0);
    if (!show || score > show.score) show = { q, score };
  }
  let showcase = null;
  if (show) {
    const p = pre.get(show.q.i);
    const d = p.diagnosis.find((x) => learned.has(x.rule)) || p.diagnosis[0];
    showcase = {
      before: p.top, after: await explainQ(show.q), clicked: show.q === focus,
      diagnosis: d ? { ...d, label: RULES[d.rule].label, fromTitle: d.from != null ? docs[d.from].title : null, toTitle: docs[d.to].title } : null,
    };
  }
  res.json({ point, showcase, fixedCount, focusFixed: useFocus && show?.q === focus, state: state() });
}).catch((e) => res.status(500).json({ error: e.message })));

app.post("/api/reset", (req, res) => locked(async () => { await reset(); res.json(state()); }).catch((e) => res.status(500).json({ error: e.message })));

// Parity: the same test questions answered through HydraDB and through the in-memory reference.
app.get("/api/parity", (req, res) => locked(async () => {
  const ref = new MemoryStore();
  for (const r of L.rules) await ref.addEdges(L.edgesOf(r).map(([a, b]) => [a, b, r]));
  let same = 0;
  const t0 = Date.now();
  for (const q of test) {
    const a = await L.run(q, L.params, L.rules, false, { neighbors: (ids) => L.serve.neighbors(ids, { fresh: true }) });
    const b = await L.run(q, L.params, L.rules, false, ref);
    if (a.join() === b.join()) same++;
  }
  res.json({ store: state().store, cache: "bypassed: every neighbour list read fresh from the store", questions: test.length, identicalTop5: same, rules: L.rules, edgesServe: await L.serve.edgeCount(), edgesInDb: serve.dbEdgeCount ? await serve.dbEdgeCount() : null, ms: Date.now() - t0 });
}).catch((e) => res.status(500).json({ error: e.message })));

app.listen(PORT, () => console.log(`[rewire] listening on :${PORT} (store=${STORE})`));
boot().catch((e) => { bootError = e.message; console.error("[rewire] BOOT FAILED:", e); });

async function waitFor(store) {
  let last;
  for (let i = 0; i < 90; i++) {
    try { await store.q("MATCH (n:Doc) RETURN n.id AS id LIMIT 1"); console.log(`[rewire] HydraDB answering after ${i}s`); return; }
    catch (e) { last = e; if (i % 10 === 0) console.log(`[rewire] waiting for HydraDB: ${e.message}`); await new Promise((r) => setTimeout(r, 1000)); }
  }
  throw new Error(`HydraDB did not answer in 90s: ${last?.message}`);
}
