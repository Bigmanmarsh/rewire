# Rewire

**Search that rewires its own graph every time it misses.** Built on the open-source [HydraDB](https://github.com/hydra-db/hydradb) engine for the HydraDB Hackathon, Track 1 (self-improving graph).

**Live demo:** https://rewire-iutk.onrender.com · **Video:** {{LOOM_URL}}

Vector search finds the paragraph that *sounds* like your question. Many real questions need a second paragraph, though: the one the first paragraph *points to*. Rewire stores every paragraph as a HydraDB node with `MENTIONS` links and follows them.

When it misses, it diagnoses which missing link caused the miss and learns a naming rule ("surname only", "drop the (qualifier)", "acronym"…). It adds that rule's links across the whole graph, and keeps the rule only if it fixes more past questions than it breaks.

## Results (measured, held-out)

Task: HotpotQA multi-hop questions over one shared pool of **5,911 paragraphs**. Each question needs **two** specific paragraphs. A question counts only if **both** are in the top 5. Everything is scored on **400 held-out questions** that are never used for learning or tuning. The 200 training questions are the only thing anything learns from, baselines included.

| Method | Held-out accuracy | Bridge | Comparison |
|---|---|---|---|
| Vector search (MiniLM `all-MiniLM-L6-v2`) | 49.5% | 40.6% | 85.0% |
| BM25 keyword search | 53.5% | 51.2% | 62.5% |
| **Hybrid BM25 + vector** (mix weight tuned on train), the strongest baseline | **64.8%** | 59.7% | 85.0% |
| Rewire, round 0 (exact-title links only, untuned) | 72.5% | 68.8% | 87.5% |
| Rewire, learning traversal weights only (no new rules) | 79.3% | n/a | n/a |
| **Rewire, full learning (10 rounds × 20 training questions)** | **83.3%** | 79.4% | 98.8% |
| Same learning loop with graph traversal switched off | 53.5% | n/a | n/a |

**Curve (held-out, per round):** 72.5 → 79.8 → 79.3 → 81.0 → 83.3 → 83.3 … 83.3%.
- Round 1 learns traversal weights.
- Round 3 adopts *Surname only* at trust 0.5 (+15,491 links).
- Round 4 adopts *Drop (qualifier)* at trust 1.
- Round 7 adopts *Before the comma*, with no held-out change.

**Not a one-off.** Five shuffled training orders end at 82.8%, 83.3%, 82.8%, 79.5% and 83.5%.

**Same numbers live on HydraDB.** On the deployed container the curve reproduces round by round (72.5 → 79.75 → 79.25 → 81.0 → 83.25%). `/api/parity` answers all 400 held-out questions through HydraDB and through the in-memory reference store. Result: **400/400 identical top-5 lists**, both at round 0 and after learning (rules `exact`, `surname`, `paren`). HydraDB stores one edge per (pair, rule): 2,880 + 16,765 + 2,520 = 22,165 edges covering 20,425 distinct paragraph pairs, which matches the reference exactly.

**Load (free Render instance, 0.1 CPU / 512 MB, HydraDB and the app in one container).** 300 question lookups at concurrency 20: **0 errors**, 11 requests/s, p50 1.7 s, p95 3.4 s, p99 4.3 s. Requests are serialised behind one lock so learning and reading never interleave, so latency here is mostly queueing.

Reproduce: `npm run bench` (offline, ~1 min) · `node scripts/loadtest.js <url> 300 20` · open `<url>/api/parity`.

## How it works

```
question ──MiniLM──▶ seed paragraphs ──HydraDB traversal──▶ linked paragraphs ──▶ top 5
                                         (:Doc)-[:MENTIONS {rule}]->(:Doc)
miss ──▶ diagnose missing link ──▶ candidate alias rule ──▶ trial on scratch graph
     ──▶ keep only if past questions improve ──▶ commit edges to HydraDB ──▶ re-tune traversal weights
```

1. **Corpus.** HotpotQA (distractor, validation): the first 600 questions. Their paragraphs are pooled and deduplicated into one shared index of 5,911 paragraphs, so every question searches all of them, not just its own 10.
2. **Graph.** Each paragraph becomes a `:Doc` node. A `-[:MENTIONS {rule}]->` edge is created when paragraph A's text contains an alias of paragraph B's title under an active rule. Round 0 has only the `exact` rule.
3. **Retrieval.** MiniLM (`all-MiniLM-L6-v2`) cosine similarity picks seed paragraphs. HydraDB returns the seeds' out- and in-neighbours (two directed Cypher queries, because the engine rejects undirected patterns). A neighbour is boosted by `λ · w_dir · sim(seed) / (1 + d · log(1+deg))`. Paragraphs named in the question get a bonus `μ`.
4. **Learning (Track 1).** Each round reads a new batch of 20 training questions. For every miss it checks which inactive rule would have created a link between the two needed paragraphs, or linked the question to one of them. The four best-supported candidate rules are tried on a scratch copy of the graph at trust 1, 0.5 and 0.25. At most one rule is adopted per round: the one that raises net hits on all training questions seen so far. Rules that don't help are rolled back. Adopted rules' edges are written to HydraDB. The traversal weights (`λ, w_in, d, μ, seeds`) and each rule's trust are then re-tuned by coordinate search on the seen training questions.
5. **Scoring.** Accuracy = *both* gold paragraphs in the top 5, on 400 held-out questions that are never used for learning.
6. **Two stores, one contract.** Rules are trialled on an in-memory scratch graph, and only adopted rules are written to HydraDB. Live answers and the held-out curve are traversed in HydraDB. `/api/parity` proves both stores agree.

## Why it needs a graph

- **It falls apart on a plain store.** Run the identical learning loop with traversal off: the learned rules can still match names in the question, but no links are followed. It reaches only 53.5%, against 83.3% with traversal.
- **The gain is on the questions vectors can't answer.** On *bridge* questions, where the second paragraph shares almost no words with the question, Rewire scores 79.4% against 59.7% for the best baseline.
- **The learning is about the graph's own structure.** Every adopted rule is a new family of edges, and every trust weight is a per-edge-type weight used during traversal. A vector index has no place to put either.

## Run it

```bash
npm install
npm run fetch        # HotpotQA rows from the Hugging Face datasets-server (~4 MB)
npm run embed        # MiniLM embeddings (~14 min on a laptop CPU; not stored in git)
npm run precompute   # similarity table + every rule's candidate edges
npm run bench        # prints the curve and writes data/bench.json
npm test
STORE=memory npm start   # http://localhost:3000, in-memory reference store
```

The `data/` folder ships precomputed (except the raw embeddings), so `npm run bench` and `npm start` work right after `npm install`. To run against HydraDB, build the Dockerfile: it starts the official `ghcr.io/hydra-db/hydradb:0.4.0` engine on `127.0.0.1` with a random per-boot token, then the app with `STORE=hydra`.

## Limits (what this is not)

| Limit | What it means |
|---|---|
| One benchmark | HotpotQA distractor-validation, 600 questions, with 5,911 pooled paragraphs. Not tested on other corpora yet. |
| Small rule space | The learner picks from 8 hand-written alias templates and their trust weights. It discovers which ones a corpus needs, but it does not invent new kinds of rules. |
| Plateau | Held-out accuracy stops rising after about 80 training questions (83.3%), because the useful rules have all been found. |
| Baselines | MiniLM, BM25 and a tuned hybrid. No cross-encoder reranker or larger embedding model was tried; those would raise every row, Rewire's seeds included. |
| Live demo is one shared session | All visitors share one learning state. *Reset demo* replays it from round 0. |
| Free hosting | Render free tier: sleeps after 15 min idle (the first load takes about 50 s) and the disk is wiped on restart, so the graph is rebuilt in HydraDB at boot. |
| HydraDB 0.4.0 dialect | No undirected patterns, no `count()`, and batched edge `MERGE` must key on an `id`. The store is written around these limits and fails loudly instead of truncating. |
| No free-text questions live | The live server uses precomputed MiniLM similarities for the 600 benchmark questions, which keeps the box under 512 MB. Free-text questions would need the model loaded at runtime. |

## Credits

HotpotQA (Yang et al., 2018), CC BY-SA 4.0 · MiniLM via `@huggingface/transformers` · HydraDB open-source engine. MIT licence for this code.
