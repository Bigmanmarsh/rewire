// Offline precompute so the 0.1-CPU live server does no heavy maths at boot:
//  data/sims.f32        question x paragraph cosine similarities (MiniLM), row = question index
//  data/rule-edges.json every alias rule's candidate MENTIONS edges, {rule: [[from, to], ...]}
import { readFile, writeFile } from "node:fs/promises";
import { Corpus, RULE_IDS, sims } from "../src/core.js";

const dataDir = new URL("../data/", import.meta.url);
const f32 = async (f) => { const b = await readFile(new URL(f, dataDir)); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };
const docs = JSON.parse(await readFile(new URL("corpus.json", dataDir), "utf8"));
const qs = JSON.parse(await readFile(new URL("questions.json", dataDir), "utf8"));
const corpus = new Corpus(docs, await f32("emb-docs.f32"));
const qEmb = await f32("emb-q.f32");

const all = new Float32Array(qs.length * corpus.N);
for (const q of qs) all.set(sims(corpus, qEmb.subarray(q.i * 384, (q.i + 1) * 384)), q.i * corpus.N);
await writeFile(new URL("sims.f32", dataDir), Buffer.from(all.buffer));

const edges = {};
for (const r of RULE_IDS) { edges[r] = corpus.edgesFor(r); console.log(`${r}: ${edges[r].length} edges`); }
await writeFile(new URL("rule-edges.json", dataDir), JSON.stringify(edges));
console.log("precompute done");
