// Rewire core: mention graph, seed-and-expand retrieval, and the learn-from-misses loop.
// Pure logic. Storage of edges goes through a GraphStore (memory or HydraDB), see stores/.

export const DIM = 384;

export function tokens(s) {
  return s.normalize("NFKD").replace(/\p{M}/gu, "").toLowerCase().match(/[\p{L}\p{N}]+/gu) || [];
}
const stripParen = (t) => t.replace(/\s*\([^)]*\)\s*/g, " ").trim();
const STOP = new Set("the a an of and in on at to for by with from is was are be as or its it his her their this that new".split(" "));

// ---- Alias rules: ways a paragraph can refer to another paragraph's title ----
// "exact" is always on. The rest are candidates the learner may adopt when misses show it needs them.
export const RULES = {
  exact: { label: "Exact title", desc: "the text contains the full title", alias: (t) => [tokens(t).join(" ")] },
  paren: { label: "Drop (qualifier)", desc: "'Ed Wood (film)' is referred to as 'Ed Wood'", alias: (t) => (t.includes("(") ? [tokens(stripParen(t)).join(" ")] : []) },
  the: { label: "Drop leading 'The'", desc: "'The Beatles' is referred to as 'Beatles'", alias: (t) => { const k = tokens(stripParen(t)); return k[0] === "the" && k.length > 1 ? [k.slice(1).join(" ")] : []; } },
  comma: { label: "Before the comma", desc: "'Paris, Texas' is referred to as 'Paris'", alias: (t) => (t.includes(",") ? [tokens(t.split(",")[0]).join(" ")] : []) },
  surname: {
    label: "Surname only", desc: "'Scott Derrickson' is referred to as 'Derrickson'",
    alias: (t) => { const w = stripParen(t).split(/\s+/); if (w.length < 2 || w.length > 3 || !w.every((x) => /^\p{Lu}/u.test(x))) return []; const k = tokens(w[w.length - 1]).join(" "); return k.length >= 4 && !STOP.has(k) ? [k] : []; },
  },
  firstlast: {
    label: "First + last name", desc: "'John Fitzgerald Kennedy' is referred to as 'John Kennedy'",
    alias: (t) => { const w = stripParen(t).split(/\s+/); return w.length === 3 && w.every((x) => /^\p{Lu}/u.test(x)) ? [tokens(`${w[0]} ${w[2]}`).join(" ")] : []; },
  },
  acronym: {
    label: "Acronym", desc: "'Federal Bureau of Investigation' is referred to as 'FBI'",
    alias: (t) => { const w = stripParen(t).split(/\s+/).filter((x) => /^\p{Lu}/u.test(x)); return w.length >= 3 ? [w.map((x) => x[0]).join("").toLowerCase()] : []; },
  },
  nopunct: { label: "Join split words", desc: "'Spider-Man' is referred to as 'Spiderman'", alias: (t) => { const k = tokens(stripParen(t)); return k.length > 1 && /[-.']/.test(t) ? [k.join("")] : []; } },
};
export const RULE_IDS = Object.keys(RULES);

// ---- Corpus index ----
export class Corpus {
  constructor(docs, docEmb) {
    this.docs = docs;
    this.emb = docEmb;
    this.N = docs.length;
    this.aliasCache = {};
  }
  get ngrams() { return (this._ng ??= this.docs.map((d) => ngramSet(tokens(d.text), 6))); }
  // alias key -> doc ids, for one rule; ambiguous or overly common aliases are dropped (noise guard)
  aliasMap(rule) {
    if (this.aliasCache[rule]) return this.aliasCache[rule];
    const m = new Map();
    for (const d of this.docs) for (const a of RULES[rule].alias(d.title)) {
      if (!a || a.length < 3 || STOP.has(a)) continue;
      if (!m.has(a)) m.set(a, []);
      m.get(a).push(d.id);
    }
    for (const [k, ids] of m) if (ids.length > 3) m.delete(k);
    return (this.aliasCache[rule] = m);
  }
  // all MENTIONS edges a rule implies: [[from, to], ...]
  edgesFor(rule) {
    const m = this.aliasMap(rule);
    const df = new Map();
    const out = [];
    for (const [k] of m) df.set(k, 0);
    for (let a = 0; a < this.N; a++) for (const g of this.ngrams[a]) if (df.has(g)) df.set(g, df.get(g) + 1);
    const tooCommon = new Set([...df].filter(([, c]) => c > this.N * 0.01).map(([k]) => k));
    for (let a = 0; a < this.N; a++) for (const g of this.ngrams[a]) {
      if (tooCommon.has(g)) continue;
      const ids = m.get(g);
      if (ids) for (const b of ids) if (b !== a) out.push([a, b]);
    }
    return out;
  }
  // which alias of doc `to` (under `rule`) appears in doc `from`'s text — the evidence for a link
  aliasIn(rule, from, to) {
    const g = ngramSet(tokens(this.docs[from].text), 6);
    return RULES[rule].alias(this.docs[to].title).find((a) => g.has(a)) || null;
  }
  // docs whose alias (under the active rules) appears in the question: Map(doc -> Set(rules))
  linkQuestion(text, rules) {
    const g = ngramSet(tokens(text), 6);
    const hits = new Map();
    for (const r of rules) {
      const m = this.aliasMap(r);
      for (const k of g) { const ids = m.get(k); if (ids) for (const x of ids) { if (!hits.has(x)) hits.set(x, new Set()); hits.get(x).add(r); } }
    }
    return hits;
  }
}

function ngramSet(tok, maxN) {
  const s = new Set();
  for (let i = 0; i < tok.length; i++) for (let n = 1; n <= maxN && i + n <= tok.length; n++) s.add(tok.slice(i, i + n).join(" "));
  return s;
}

export function sims(corpus, qv) {
  const s = new Float32Array(corpus.N);
  const E = corpus.emb;
  for (let d = 0; d < corpus.N; d++) { let x = 0; const o = d * DIM; for (let j = 0; j < DIM; j++) x += E[o + j] * qv[j]; s[d] = x; }
  return s;
}
export function topK(scores, k, ids) {
  const arr = ids ? [...ids] : Array.from({ length: scores.length }, (_, i) => i);
  return arr.sort((a, b) => scores[b] - scores[a]).slice(0, k);
}

// rw = learned trust per rule (0..1); an edge's weight is the highest trust among the rules that imply it.
export const DEFAULT_PARAMS = { seeds: 5, lam: 0.0, inW: 0.5, damp: 0.0, mu: 0.0, rw: { exact: 1 } };
const trust = (rw, rules) => { let w = 0; for (const r of rules) w = Math.max(w, rw[r] ?? 1); return w; };

// Seed-and-expand. `neighbors(ids)` returns {out: Map(id -> [[id, rules]]), in: ...} from the store.
export function prepare(corpus, qv) {
  const s = sims(corpus, qv);
  return { s, top: topK(s, 50) };
}

export async function retrieve({ corpus, qv, qtext, rules, params, neighbors, k = 5, explain = false, pre, linked }) {
  const { s, top } = pre || prepare(corpus, qv);
  linked = params.mu > 0 ? linked || corpus.linkQuestion(qtext, rules) : new Map();
  const rw = params.rw || {};
  const qbonus = new Map([...linked].map(([d, rs]) => [d, params.mu * trust(rw, rs)]));
  const seeds = new Set(top.slice(0, params.seeds));
  for (const [d, b] of qbonus) if (b > 0) seeds.add(d);
  const score = new Map();
  const why = new Map();
  for (const d of top) score.set(d, s[d]);
  for (const [d, b] of qbonus) score.set(d, (score.get(d) ?? s[d]) + b);
  if (params.lam > 0) {
    const nb = await neighbors([...seeds]);
    const deg = (id) => (nb.out.get(id)?.length || 0) + (nb.in.get(id)?.length || 0);
    for (const [dirMap, w, dir] of [[nb.out, 1, "out"], [nb.in, params.inW, "in"]]) {
      for (const [src, list] of dirMap) {
        const base = s[src] + (qbonus.get(src) || 0);
        const boost = (params.lam * w * base) / (1 + params.damp * Math.log1p(deg(src)));
        for (const [n, rs] of list) {
          const t = trust(rw, rs);
          if (t <= 0) continue;
          const v = s[n] + boost * t;
          if (v > (score.get(n) ?? -Infinity)) { score.set(n, v); if (explain) why.set(n, { via: src, dir, rules: [...rs] }); }
        }
      }
    }
  }
  const ranked = [...score.keys()].sort((a, b) => score.get(b) - score.get(a)).slice(0, k);
  return explain ? ranked.map((id) => ({ id, score: score.get(id), sim: s[id], via: why.get(id) || null, linked: (qbonus.get(id) || 0) > 0 })) : ranked;
}

export const hit = (ranked, gold) => gold.every((g) => ranked.includes(g));
