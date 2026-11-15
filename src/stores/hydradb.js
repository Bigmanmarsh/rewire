// HydraDB GraphStore over the open-source engine's HTTP query API.
// Graph model: (:Doc {id, title}) -[:MENTIONS {rule}]-> (:Doc). One edge per (from, to, rule).
// Engine notes we design around (HydraDB 0.4.0): only directed patterns run, so neighbours are two
// queries (out and in); results are paged (we follow next_cursor every time); UNWIND batches cap
// at 1024 rows.
const PAGE = 4096, BATCH = 1000;
const RULE_NO = { exact: 0, paren: 1, the: 2, comma: 3, surname: 4, firstlast: 5, acronym: 6, nopunct: 7 };
// one stable integer id per (from, to, rule) edge
export const edgeId = (a, b, r) => (RULE_NO[r] ?? 9) * 1e8 + a * 1e4 + b;

export class HydraStore {
  constructor({ url, token, cell = "cell-0", graph = "default", timeoutMs = 20000 }) {
    Object.assign(this, { url: url.replace(/\/$/, ""), token, cell, graph, timeoutMs });
    this.cache = new Map();
    this.pairs = new Map(); // pair -> Set(rule), client mirror so counts match the memory store
    this.queries = 0;
  }

  // Runs one statement and follows next_cursor so a large result is never silently cut short.
  async q(query, parameters = {}) {
    const all = [];
    let cursor;
    for (let page = 0; page < 100; page++) {
      this.queries++;
      const res = await fetch(`${this.url}/v1/graphs/${this.graph}/query`, {
        method: "POST",
        headers: { Authorization: `Bearer ${this.token}`, "Content-Type": "application/json" },
        body: JSON.stringify({ cell_id: this.cell, query, parameters, page_size: PAGE, ...(cursor != null ? { cursor } : {}) }),
        signal: AbortSignal.timeout(this.timeoutMs),
      });
      const text = await res.text();
      if (!res.ok) throw new Error(`HydraDB ${res.status}: ${text.slice(0, 300)}`);
      const body = JSON.parse(text);
      all.push(...rowsOf(body));
      if (body.next_cursor == null) return all;
      cursor = body.next_cursor;
    }
    throw new Error("HydraDB result exceeded 100 pages; refusing a partial answer");
  }

  async loadDocs(docs) {
    for (let i = 0; i < docs.length; i += BATCH) {
      const rows = docs.slice(i, i + BATCH).map((d) => ({ id: d.id, t: d.title }));
      await this.q("UNWIND $rows AS row MERGE (n {id: row.id}) SET n:Doc, n.title = row.t", { rows });
    }
  }

  async clear() {
    await this.q("MATCH (a:Doc)-[r:MENTIONS]->(b:Doc) DELETE r");
    this.pairs.clear(); this.cache.clear();
  }

  async addEdges(edges) {
    let added = 0;
    const fresh = [];
    for (const [a, b, r] of edges) {
      const k = a * 1e6 + b;
      if (!this.pairs.has(k)) { this.pairs.set(k, new Set()); added++; }
      const s = this.pairs.get(k);
      if (!s.has(r)) { s.add(r); fresh.push({ id: edgeId(a, b, r), a, b, r }); }
    }
    for (let i = 0; i < fresh.length; i += BATCH) {
      await this.q(
        // HydraDB 0.4.0: a batched relationship MERGE must key on {id: row.<field>}
        "UNWIND $rows AS row MATCH (a:Doc {id: row.a}), (b:Doc {id: row.b}) MERGE (a)-[e:MENTIONS {id: row.id}]->(b) SET e.rule = row.r",
        { rows: fresh.slice(i, i + BATCH) },
      );
    }
    this.cache.clear();
    return added;
  }

  async removeRule(rule) {
    await this.q("MATCH (a:Doc)-[r:MENTIONS]->(b:Doc) WHERE r.rule = $rule DELETE r", { rule });
    for (const [k, s] of this.pairs) { s.delete(rule); if (!s.size) this.pairs.delete(k); }
    this.cache.clear();
  }

  // Neighbour lists come from HydraDB; cached per graph version (cache is cleared on every write).
  async neighbors(ids) {
    const need = ids.filter((id) => !this.cache.has(id));
    if (need.length) {
      const got = new Map(need.map((id) => [id, { out: new Map(), in: new Map() }]));
      await this.fetchDir(need, "out", got);
      await this.fetchDir(need, "in", got);
      for (const [id, v] of got) this.cache.set(id, { out: [...v.out], in: [...v.in] });
    }
    const out = new Map(), inn = new Map();
    for (const id of ids) {
      const c = this.cache.get(id);
      if (c.out.length) out.set(id, c.out);
      if (c.in.length) inn.set(id, c.in);
    }
    return { out, in: inn };
  }

  async fetchDir(ids, dir, got) {
    const pat = dir === "out" ? "(a:Doc)-[r:MENTIONS]->(b:Doc)" : "(a:Doc)<-[r:MENTIONS]-(b:Doc)";
    for (let i = 0; i < ids.length; i += 200) {
      const rows = await this.q(`MATCH ${pat} WHERE a.id IN $ids RETURN a.id AS a, b.id AS b, r.rule AS rule`, { ids: ids.slice(i, i + 200) });
      for (const r of rows) {
        const m = got.get(r.a)?.[dir];
        if (!m) continue;
        if (!m.has(r.b)) m.set(r.b, new Set());
        m.get(r.b).add(r.rule);
      }
    }
  }

  async edgeCount() { return this.pairs.size; }
  // count() is not supported by the 0.4.0 query engine, so count the (paged) rows instead
  async dbEdgeCount() { return (await this.q("MATCH (a:Doc)-[r:MENTIONS]->(b:Doc) RETURN r.rule AS rule")).length; }

  // Explanation path for the UI: the actual 1-hop link HydraDB holds between two docs.
  async link(a, b) {
    return this.q("MATCH (x:Doc {id: $a})-[r:MENTIONS]->(y:Doc {id: $b}) RETURN r.rule AS rule, y.title AS title", { a, b });
  }
}

// HTTP reply: {columns: [name], rows: [[{type, value}]], next_cursor}. Values are tagged; unwrap them.
export function rowsOf(body) {
  const val = (v) => {
    if (!v || typeof v !== "object") return v;
    if (v.type === "list") return v.value.map(val);
    if (v.type === "record") return Object.fromEntries(Object.entries(v.value).map(([k, x]) => [k, val(x)]));
    return v.value ?? null;
  };
  return (body.rows || []).map((row) => Object.fromEntries(body.columns.map((c, i) => [c, val(row[i])])));
}
