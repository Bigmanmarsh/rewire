# Rewire

**Search that rewires its own graph every time it misses.** Built on the open-source [HydraDB](https://github.com/hydra-db/hydradb) engine for the HydraDB Hackathon, Track 1 (self-improving graph).

**Live demo:** {{LIVE_URL}} · **Video:** {{LOOM_URL}}

Vector search finds the paragraph that *sounds* like your question. Many real questions need a second paragraph, though: the one the first paragraph *points to*. Rewire stores every paragraph as a HydraDB node with `MENTIONS` links and follows them.

When it misses, it diagnoses which missing link caused the miss and learns a naming rule ("surname only", "drop the (qualifier)", "acronym"…). It adds that rule's links across the whole graph, and keeps the rule only if it fixes more past questions than it breaks.

## Results (measured, held-out)

{{RESULTS}}

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
4. **Learning (Track 1).** Each round reads a new batch of 40 training questions. For every miss it checks which inactive rule would have created a link between the two needed paragraphs, or linked the question to one of them. Candidate rules are tried most-supported first on a scratch copy of the graph. A rule is adopted only if it raises hits on all training questions seen so far; otherwise it is rolled back. Adopted rules' edges are written to HydraDB. The traversal weights (`λ, w_in, d, μ, seeds`) are then re-tuned by coordinate search on the seen training questions.
5. **Scoring.** Accuracy = *both* gold paragraphs in the top 5, on 400 held-out questions that are never used for learning.

## Why it needs a graph

{{WHY_GRAPH}}

## Run it

```bash
npm install
npm run fetch        # HotpotQA rows from the Hugging Face datasets-server (~4 MB)
npm run embed        # MiniLM embeddings (~25 min on a laptop CPU)
npm run precompute   # similarity table + every rule's candidate edges
npm run bench        # prints the curve and writes data/bench.json
npm test
STORE=memory npm start   # http://localhost:3000, in-memory reference store
```

The `data/` folder ships precomputed, so only `npm start` is needed. To run against HydraDB, build the Dockerfile: it starts the official `ghcr.io/hydra-db/hydradb:0.4.0` engine on `127.0.0.1` with a random per-boot token, then the app with `STORE=hydra`.

## Limits (what this is not)

{{LIMITS}}

## Credits

HotpotQA (Yang et al., 2018), CC BY-SA 4.0 · MiniLM via `@huggingface/transformers` · HydraDB open-source engine. MIT licence for this code.
