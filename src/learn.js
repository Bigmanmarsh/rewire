// The learning loop. Each round reads a fresh batch of TRAINING questions, finds the misses,
// diagnoses which missing link caused each one, adopts the alias rules that fix misses without
// breaking earlier hits, then re-tunes the traversal weights. Test questions are never read here.
import { RULES, RULE_IDS, prepare, retrieve, hit, topK, DEFAULT_PARAMS } from "./core.js";

const GRID = {
  lam: [0, 0.1, 0.2, 0.3, 0.45, 0.6],
  inW: [0, 0.5, 1],
  damp: [0, 0.3, 0.7, 1.5],
  mu: [0, 0.05, 0.1, 0.2],
  seeds: [2, 3, 5, 8],
};
const TRUST = [0.15, 0.3, 0.5, 0.75, 1];

export class Learner {
  // simsAll / ruleEdges are optional precomputed tables (see scripts/precompute.js)
  // store = scratch graph the learner trials rules on; serve = the graph live queries and the test
  // curve run against (HydraDB in production). Only adopted rules are committed to `serve`.
  constructor({ corpus, store, serve, questions, qEmb, simsAll, ruleEdges }) {
    Object.assign(this, { corpus, store, serve: serve || store, questions, simsAll });
    this.qv = (q) => qEmb?.subarray(q.i * 384, (q.i + 1) * 384);
    this.pre = new Map();
    this.ruleEdges = { ...(ruleEdges || {}) };
    this.ruleEdgeSet = {};
  }
  preFor(q) {
    if (!this.pre.has(q.i)) {
      const N = this.corpus.N;
      if (this.simsAll) { const s = this.simsAll.subarray(q.i * N, (q.i + 1) * N); this.pre.set(q.i, { s, top: topK(s, 50) }); }
      else this.pre.set(q.i, prepare(this.corpus, this.qv(q)));
    }
    return this.pre.get(q.i);
  }
  edgesOf(rule) {
    this.ruleEdges[rule] ??= this.corpus.edgesFor(rule);
    this.ruleEdgeSet[rule] ??= new Set(this.ruleEdges[rule].map(([a, b]) => a * 1e6 + b));
    return this.ruleEdges[rule];
  }

  async reset({ withGraph = true } = {}) {
    await this.store.clear();
    if (this.serve !== this.store) await this.serve.clear();
    this.rules = ["exact"];
    this.params = { ...DEFAULT_PARAMS, rw: { exact: 1 }, ...(withGraph ? { lam: 0.2, inW: 0.5, mu: 0.05 } : {}) };
    this.withGraph = withGraph;
    this.seen = [];
    this.history = [];
    if (withGraph) await this.commit("exact", true);
  }

  async commit(rule, scratchToo = false) {
    const rows = this.edgesOf(rule).map(([a, b]) => [a, b, rule]);
    const added = scratchToo ? await this.store.addEdges(rows) : 0;
    if (this.serve !== this.store) return Math.max(added, await this.serve.addEdges(rows));
    return added;
  }

  async run(q, params = this.params, rules = this.rules, explain = false, store = this.store) {
    return retrieve({
      corpus: this.corpus, qv: this.qv(q), qtext: q.question, rules, params, explain,
      pre: this.preFor(q), neighbors: (ids) => store.neighbors(ids),
    });
  }
  async score(qs, params = this.params, rules = this.rules) {
    let n = 0;
    for (const q of qs) if (hit(await this.run(q, params, rules), q.gold)) n++;
    return n;
  }
  async evaluate(qs, store = this.serve) {
    const by = {};
    let n = 0;
    for (const q of qs) {
      const ok = hit(await this.run(q, this.params, this.rules, false, store), q.gold);
      n += ok;
      by[q.type] ??= [0, 0]; by[q.type][0] += ok; by[q.type][1]++;
    }
    return { acc: n / qs.length, n, total: qs.length, byType: Object.fromEntries(Object.entries(by).map(([t, [h, c]]) => [t, h / c])) };
  }

  // Why was this question missed? Returns the candidate rules that would have created the missing link.
  diagnose(q) {
    const [A, B] = q.gold;
    const out = [];
    for (const r of RULE_IDS) {
      if (this.rules.includes(r)) continue;
      this.edgesOf(r);
      const set = this.ruleEdgeSet[r];
      const ab = set.has(A * 1e6 + B), ba = set.has(B * 1e6 + A);
      if (ab || ba) {
        const [from, to] = ab ? [A, B] : [B, A];
        out.push({ rule: r, kind: "pair", from, to, alias: this.corpus.aliasIn(r, from, to) });
        continue;
      }
      const lq = this.corpus.linkQuestion(q.question, [r]);
      const g = q.gold.find((x) => lq.has(x));
      if (g !== undefined) out.push({ rule: r, kind: "question", to: g, alias: RULES[r].alias(this.corpus.docs[g].title)[0] });
    }
    return out;
  }

  // Coordinate search over traversal weights and per-rule trust, on seen training questions only.
  async tune(qs) {
    if (!this.withGraph) return;
    let best = await this.score(qs);
    const keys = [...Object.keys(GRID), ...this.rules.filter((r) => r !== "exact").map((r) => "rw." + r)];
    for (let pass = 0; pass < 2; pass++) for (const key of keys) for (const v of key.startsWith("rw.") ? TRUST : (this.gridOverride?.[key] ?? GRID[key])) {
      const p = key.startsWith("rw.") ? { ...this.params, rw: { ...this.params.rw, [key.slice(3)]: v } } : { ...this.params, [key]: v };
      if (JSON.stringify(p) === JSON.stringify(this.params)) continue;
      const sc = await this.score(qs, p);
      if (sc > best) { best = sc; this.params = p; }
    }
    return best;
  }

  // One learning round over a batch of training questions. Adopts at most one new rule per round,
  // choosing the candidate (and trust level) that fixes the most seen training questions net.
  async round(batch) {
    this.seen.push(...batch);
    const log = { round: this.history.length + 1, batch: batch.length, seen: this.seen.length, adopted: [], rejected: [], misses: 0 };
    if (this.withGraph) {
      const support = {};
      for (const q of this.seen) {
        if (hit(await this.run(q), q.gold)) continue;
        log.misses++;
        for (const d of this.diagnose(q)) support[d.rule] = (support[d.rule] || 0) + 1;
      }
      const before = await this.score(this.seen);
      let pick = null;
      for (const rule of Object.keys(support).sort((a, b) => support[b] - support[a]).slice(0, 4)) {
        // dry run on the scratch graph: add the rule's edges, try trust levels, then take them out again
        await this.store.addEdges(this.edgesOf(rule).map(([a, b]) => [a, b, rule]));
        let bestT = null;
        for (const t of [1, 0.5, 0.25]) {
          const p = { ...this.params, rw: { ...this.params.rw, [rule]: t } };
          const gain = (await this.score(this.seen, p, [...this.rules, rule])) - before;
          if (!bestT || gain > bestT.gain) bestT = { rule, trust: t, gain };
        }
        await this.store.removeRule(rule);
        if (!pick || bestT.gain > pick.gain) { if (pick) log.rejected.push(this.entry(pick, support)); pick = bestT; }
        else log.rejected.push(this.entry(bestT, support));
      }
      if (pick && pick.gain > 0) {
        const added = await this.store.addEdges(this.edgesOf(pick.rule).map(([a, b]) => [a, b, pick.rule]));
        if (this.serve !== this.store) await this.commit(pick.rule);
        this.rules = [...this.rules, pick.rule];
        this.params = { ...this.params, rw: { ...this.params.rw, [pick.rule]: pick.trust } };
        log.adopted.push({ ...this.entry(pick, support), edges: added });
      } else if (pick) log.rejected.push(this.entry(pick, support));
    }
    log.trainHits = (await this.tune(this.seen)) ?? (await this.score(this.seen));
    log.params = JSON.parse(JSON.stringify(this.params));
    log.rules = [...this.rules];
    log.edges = await this.store.edgeCount();
    this.history.push(log);
    return log;
  }

  entry(p, support) { return { rule: p.rule, label: RULES[p.rule].label, support: support[p.rule], gain: p.gain, trust: p.trust }; }
}
