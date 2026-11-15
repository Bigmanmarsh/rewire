// Build the shared corpus (paragraphs pooled across all questions, deduped by title)
// and embed every paragraph and question with MiniLM (all-MiniLM-L6-v2, 384 dims).
// Outputs: data/corpus.json, data/questions.json, data/emb-docs.f32, data/emb-q.f32
import { readFile, writeFile } from "node:fs/promises";
import { pipeline } from "@huggingface/transformers";

const dataDir = new URL("../data/", import.meta.url);
const raw = JSON.parse(await readFile(new URL("hotpot.json", dataDir), "utf8"));

const byTitle = new Map();
for (const q of raw) for (const c of q.context) if (!byTitle.has(c.title)) byTitle.set(c.title, c.text);
const corpus = [...byTitle].map(([title, text], id) => ({ id, title, text }));
const idOf = new Map(corpus.map((d) => [d.title, d.id]));

// Deterministic split: first 200 = train (the graph may learn from these), rest = held-out test.
const questions = raw.map((q, i) => ({
  i, id: q.id, question: q.question, answer: q.answer, type: q.type, level: q.level,
  gold: q.supporting.map((t) => idOf.get(t)),
  split: i < 200 ? "train" : "test",
}));

const embed = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { dtype: "fp32" });
async function embedAll(texts, label) {
  const out = new Float32Array(texts.length * 384);
  const B = 32;
  for (let s = 0; s < texts.length; s += B) {
    const t = await embed(texts.slice(s, s + B), { pooling: "mean", normalize: true });
    out.set(t.data, s * 384);
    if ((s / B) % 20 === 0) console.log(`${label} ${s}/${texts.length}`);
  }
  return out;
}

const t0 = Date.now();
const docEmb = await embedAll(corpus.map((d) => `${d.title}. ${d.text}`), "docs");
const qEmb = await embedAll(questions.map((q) => q.question), "questions");
await writeFile(new URL("corpus.json", dataDir), JSON.stringify(corpus));
await writeFile(new URL("questions.json", dataDir), JSON.stringify(questions));
await writeFile(new URL("emb-docs.f32", dataDir), Buffer.from(docEmb.buffer));
await writeFile(new URL("emb-q.f32", dataDir), Buffer.from(qEmb.buffer));
console.log(`done: ${corpus.length} docs, ${questions.length} questions in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
