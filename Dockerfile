# One container: the open-source HydraDB engine + the Rewire Node app.
FROM node:20-bookworm-slim AS app
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev
COPY src ./src
COPY public ./public
COPY data/corpus.json data/questions.json data/sims.f32 data/rule-edges.json data/bench.json ./data/

FROM ghcr.io/hydra-db/hydradb:0.4.0
USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY --from=app /usr/local/bin/node /usr/local/bin/node
COPY --from=app /app /app
COPY --chmod=0755 start.sh /usr/local/bin/start.sh
ENV RUST_MIN_STACK=33554432 NODE_ENV=production
EXPOSE 10000
ENTRYPOINT ["/usr/local/bin/start.sh"]
