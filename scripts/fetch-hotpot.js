// Fetch a slice of HotpotQA (distractor, validation) from the Hugging Face datasets-server.
// Writes data/hotpot.json: [{id, question, answer, type, level, supporting: [title...], context: [{title, text}]}]
import { writeFile } from "node:fs/promises";

const N = Number(process.env.N || 600);
const API = "https://datasets-server.huggingface.co/rows?dataset=hotpotqa%2Fhotpot_qa&config=distractor&split=validation";

async function page(offset, length) {
  for (let attempt = 0; attempt < 4; attempt++) {
    try {
      const res = await fetch(`${API}&offset=${offset}&length=${length}`, { signal: AbortSignal.timeout(30000) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return (await res.json()).rows.map((r) => r.row);
    } catch (e) {
      console.error(`offset ${offset} attempt ${attempt + 1} failed: ${e.message}`);
      await new Promise((r) => setTimeout(r, 2000 * (attempt + 1)));
    }
  }
  throw new Error(`could not fetch offset ${offset}`);
}

const out = [];
for (let offset = 0; offset < N; offset += 100) {
  for (const row of await page(offset, Math.min(100, N - offset))) {
    out.push({
      id: row.id,
      question: row.question,
      answer: row.answer,
      type: row.type,
      level: row.level,
      supporting: [...new Set(row.supporting_facts.title)],
      context: row.context.title.map((title, i) => ({ title, text: row.context.sentences[i].join("") })),
    });
  }
  console.log(`fetched ${out.length}`);
}
await writeFile(new URL("../data/hotpot.json", import.meta.url), JSON.stringify(out));
console.log(`wrote ${out.length} questions`);
