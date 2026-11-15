// Load test against a running Rewire server: concurrent question lookups (each one traverses the
// graph in the server's store) and reports latency percentiles, throughput and errors.
// Usage: node scripts/loadtest.js https://your-app.onrender.com [requests=300] [concurrency=20]
const [base = "http://localhost:3000", N = "300", C = "20"] = process.argv.slice(2);
const total = Number(N), conc = Number(C);

const state = await (await fetch(`${base}/api/state`, { signal: AbortSignal.timeout(90000) })).json();
const first = state.train; // question indices >= train count are the held-out test split
const ids = Array.from({ length: total }, (_, k) => first + (k % state.test));

const lat = [];
let errors = 0, next = 0;
const t0 = Date.now();
async function worker() {
  while (next < ids.length) {
    const i = ids[next++];
    const s = Date.now();
    try {
      const r = await fetch(`${base}/api/q/${i}`, { signal: AbortSignal.timeout(60000) });
      if (!r.ok) throw new Error(r.status);
      await r.json();
      lat.push(Date.now() - s);
    } catch { errors++; }
  }
}
await Promise.all(Array.from({ length: conc }, worker));
const secs = (Date.now() - t0) / 1000;
lat.sort((a, b) => a - b);
const p = (q) => lat[Math.min(lat.length - 1, Math.floor(q * lat.length))];
const out = { base, store: state.store, requests: total, concurrency: conc, ok: lat.length, errors, seconds: +secs.toFixed(1), rps: +(lat.length / secs).toFixed(1), p50_ms: p(0.5), p95_ms: p(0.95), p99_ms: p(0.99), max_ms: lat.at(-1), when: new Date().toISOString() };
console.log(JSON.stringify(out, null, 1));
