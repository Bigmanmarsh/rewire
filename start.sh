#!/bin/sh
# Start HydraDB (private, 127.0.0.1 only) then the Rewire app on $PORT. The HydraDB token is
# random per boot, readable only inside the container, and never printed.
set -eu
BASE=/tmp/hydra
# Start from an empty store every boot: the app rebuilds the graph itself, and a store left over
# from a crashed run can hold a stale cell lease ("cell cell-0 is not owned by this node").
rm -rf "$BASE/store" "$BASE/cache"
mkdir -p "$BASE/store" "$BASE/cache" "$BASE/secrets"
umask 077
head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' >"$BASE/secrets/auth-token"
umask 022

export CLOUD_PROVIDER=local LOCAL_PATH="$BASE/store"
export GRAPH_NAMESPACE=default GRAPH_ID=default GRAPH_CELL_ID=cell-0 GRAPH_CELLS=cell-0 GRAPH_NODE_ID=node-0
export GRAPH_BOLT_ADDR=127.0.0.1:7687 GRAPH_HTTP_ADDR=127.0.0.1:8443 GRAPH_ADMIN_ADDR=127.0.0.1:9090
export GRAPH_BOLT_NODE_ADDRESSES=node-0=127.0.0.1:7687 GRAPH_ADVERTISED_BOLT_ADDR=127.0.0.1:7687
export GRAPH_DATA_CACHE_DIR="$BASE/cache" GRAPH_AUTH_TOKEN_FILE="$BASE/secrets/auth-token" GRAPH_ALLOW_PLAINTEXT=true
# Sized for a 512 MB free instance shared with Node.
export GRAPH_SLATE_DB_CACHE_BYTES="${GRAPH_SLATE_DB_CACHE_BYTES:-100663296}"   # 96 MiB
export GRAPH_MAX_GRAPHBLAS_BYTES="${GRAPH_MAX_GRAPHBLAS_BYTES:-33554432}"      # 32 MiB
export GRAPH_DATA_CACHE_BYTES="${GRAPH_DATA_CACHE_BYTES:-268435456}"          # 256 MiB disk
export GRAPHBLAS_NTHREADS=1 OMP_NUM_THREADS=1 TOKIO_WORKER_THREADS=2 MALLOC_ARENA_MAX=2
export GRAPH_USAGE_REPORT_ENABLED=false
export RUST_MIN_STACK=33554432 HOME="$BASE"
cd "$BASE"

echo "[boot] starting HydraDB graph-node"
/usr/local/bin/graph-node &
HYDRA_PID=$!

i=0
until curl -sf -m 3 -o /dev/null http://127.0.0.1:9090/readyz; do
  i=$((i + 1)); [ "$i" -ge 180 ] && { echo "[boot] HydraDB not ready after 180s"; exit 1; }
  kill -0 "$HYDRA_PID" 2>/dev/null || { echo "[boot] HydraDB exited"; exit 1; }
  sleep 1
done
echo "[boot] HydraDB ready after ${i}s"

cd /app
STORE=hydra HYDRA_URL=http://127.0.0.1:8443 HYDRA_TOKEN_FILE="$BASE/secrets/auth-token" \
  node --max-old-space-size=200 src/server.js &
APP_PID=$!
# If either process dies, stop the container so Render restarts it.
while kill -0 "$HYDRA_PID" 2>/dev/null && kill -0 "$APP_PID" 2>/dev/null; do sleep 5; done
echo "[boot] a process exited; stopping"
kill "$HYDRA_PID" "$APP_PID" 2>/dev/null || true
exit 1
