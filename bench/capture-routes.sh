#!/usr/bin/env bash
# Capture per-token routing (EDGE0_TRACE=routes) and next-layer pre-gating accuracy (EDGE0_TRACE=predict) on the
# standard ShareGPT workload, for tools/plan_placement.py (TidalMoE tools/collect_routing.py, adapted: the trace is
# recorded inside the Edge0 MoE op, so vLLM's routed-experts capture is not needed). Runs ON the 5090 host.
#
#   bench/capture-routes.sh [label=routes] [concs=1,4,16] [extra serve.sh env...]
#
# Shards land in ~/.cache/edge0-vllm/edge0-vllm/traces/routes-<pid>-<t>-NNNNN.npz (container /root/.cache/...);
# the phase boundaries (shard file mtimes vs. the times logged here) separate c1 sequences from batched decode.
set -uo pipefail
LABEL=${1:-routes}; CONCS=${2:-1,4,16}; if [ $# -ge 2 ]; then shift 2; else shift $#; fi
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG=$REPO/logs/capture-$LABEL.log
TRACES=/home/andychen/.cache/edge0-vllm/edge0-vllm/traces
mkdir -p "$REPO/logs"
say() { echo "# $(date +%s) $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
docker stop edge0-vllm >/dev/null 2>&1; tmux kill-session -t edge0-vllm 2>/dev/null
for _ in $(seq 1 60); do docker ps -a --format '{{.Names}}' | grep -q '^edge0-vllm$' || break; sleep 2; done; sleep 2
say "start server with routes+predict trace"
env KV_BYTES= EDGE0_TRACE=routes,predict EDGE0_TRACE_FLUSH_S=5 EDGE0_TRACE_RING_TOKENS=131072 "$@" \
  "$REPO/serve.sh" >>"$LOG" 2>&1 || { say "serve.sh failed"; exit 1; }
t0=$SECONDS
until curl -sf localhost:8125/health >/dev/null 2>&1; do
  (( SECONDS - t0 > 900 )) && { say "start timed out"; exit 1; }
  sleep 5
done
say "healthy after $((SECONDS - t0)) s"
for c in ${CONCS//,/ }; do
  f=${CAP_FACTOR:-4}; [ "$c" = 1 ] && f=3
  say "phase c$c begin (seed ${SEED:-0})"
  SEED=${SEED:-0} OUT_LEN=512 FACTOR=$f BASE_URL=http://127.0.0.1:8125 \
    EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}' \
    bash "$REPO/bench/bench-openai.sh" "vllm-$LABEL-c$c" "$c" >>"$LOG" 2>&1
  sleep 6                                   # let the 5 s flush write the last shard of the phase
  say "phase c$c end"
done
ls -la --time-style=+%s "$TRACES" | tail -n +2 >>"$LOG"
say "CAPTURE_DONE"
