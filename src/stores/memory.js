// In-memory reference GraphStore. Same contract as the HydraDB store; used for fast offline
// benchmarking and as the parity reference that the HydraDB backend is checked against.
// Each edge carries the rule that created it; a from->to pair counts once however many rules imply it.
export class MemoryStore {
  constructor() { this.byRule = new Map(); this.reindex(); }
  async clear() { this.byRule = new Map(); this.reindex(); }
  reindex() {
    this.out = new Map(); this.in = new Map(); this.pairs = new Map();
    for (const [r, list] of this.byRule) for (const [a, b] of list) this.link(a, b, r);
  }
  // returns true when the from->to pair is new
  link(a, b, r) {
    const k = a * 1e6 + b;
    if (this.pairs.has(k)) { this.pairs.get(k).add(r); return false; }
    const rules = new Set([r]);
    this.pairs.set(k, rules);
    if (!this.out.has(a)) this.out.set(a, []);
    if (!this.in.has(b)) this.in.set(b, []);
    this.out.get(a).push([b, rules]); this.in.get(b).push([a, rules]);
    return true;
  }
  // edges: [[from, to, rule], ...]; returns how many new from->to pairs appeared
  async addEdges(edges) {
    let added = 0;
    for (const [a, b, r] of edges) {
      if (!this.byRule.has(r)) this.byRule.set(r, []);
      this.byRule.get(r).push([a, b]);
      if (this.link(a, b, r)) added++;
    }
    return added;
  }
  async removeRule(rule) { this.byRule.delete(rule); this.reindex(); }
  async neighbors(ids) {
    const out = new Map(), inn = new Map();
    for (const id of ids) {
      if (this.out.has(id)) out.set(id, this.out.get(id));
      if (this.in.has(id)) inn.set(id, this.in.get(id));
    }
    return { out, in: inn };
  }
  async edgeCount() { return this.pairs.size; }
}
