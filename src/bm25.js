// BM25 keyword index, used only for the stronger hybrid baseline (BM25 + MiniLM).
import { tokens } from "./core.js";

const STOP = new Set("the a an of and in on at to for by with from is was are were be been as or its it his her their this that which who what when where how did does do".split(" "));
const terms = (s) => tokens(s).filter((t) => !STOP.has(t));

export class BM25 {
  constructor(docs, k1 = 1.2, b = 0.75) {
    this.N = docs.length; this.k1 = k1; this.b = b;
    this.post = new Map();
    this.len = new Float32Array(this.N);
    docs.forEach((d, id) => {
      const tf = new Map();
      for (const t of terms(`${d.title} ${d.text}`)) tf.set(t, (tf.get(t) || 0) + 1);
      this.len[id] = [...tf.values()].reduce((x, y) => x + y, 0);
      for (const [t, c] of tf) { if (!this.post.has(t)) this.post.set(t, []); this.post.get(t).push([id, c]); }
    });
    this.avg = this.len.reduce((x, y) => x + y, 0) / this.N;
  }
  // scores normalised to [0, 1] by the query's best match
  score(q) {
    const s = new Float32Array(this.N);
    for (const t of new Set(terms(q))) {
      const p = this.post.get(t);
      if (!p) continue;
      const idf = Math.log(1 + (this.N - p.length + 0.5) / (p.length + 0.5));
      for (const [id, c] of p) s[id] += (idf * c * (this.k1 + 1)) / (c + this.k1 * (1 - this.b + (this.b * this.len[id]) / this.avg));
    }
    let m = 0; for (const x of s) if (x > m) m = x;
    if (m > 0) for (let i = 0; i < s.length; i++) s[i] /= m;
    return s;
  }
}
