#!/usr/bin/env bash
# Probe one server configuration (runs ON the 5090): start it, record memory / KV sizing from the log, run the
# Strata correctness comparison and a quick c1 latency check, leave the server running.
#
#   bench/probe-config.sh <label> "<env for serve.sh>" [quick concs=1]
#   e.g. bench/probe-config.sh q8 "KV_BYTES= EDGE0_DENSE=q8 EDGE0_EXPERT_VRAM_GIB=9.5"
set -uo pipefail
LABEL=$1; ENVS=$2; CONCS=${3:-1}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=$REPO/logs/probe-$LABEL.log
say() { echo "# $(date +%H:%M:%S) $*" | tee -a "$OUT"; }
docker stop edge0-vllm >/dev/null 2>&1; tmux kill-session -t edge0-vllm 2>/dev/null
for _ in $(seq 1 60); do docker ps -a --format '{{.Names}}' | grep -q '^edge0-vllm$' || break; sleep 2; done; sleep 2
say "start: $ENVS"
# shellcheck disable=SC2086
env $ENVS "$REPO/serve.sh" >>"$OUT" 2>&1 || { say "serve.sh failed"; exit 1; }
SLOG=$(ls -t "$REPO"/logs/serve-*.log | head -1)
t0=$SECONDS
until curl -sf localhost:8125/health >/dev/null 2>&1; do
  if (( SECONDS - t0 > 60 )) && ! docker ps --format '{{.Names}}' | grep -q '^edge0-vllm$'; then
    say "container exited"; tail -40 "$SLOG" >>"$OUT"; exit 1
  fi
  (( SECONDS - t0 > 1200 )) && { say "start timed out"; exit 1; }
  sleep 5
done
say "healthy after $((SECONDS - t0)) s (log $SLOG)"
grep -E "edge0: (placement|built|after load|embed|drafter)|edge0 (adapt|prefetch)|Available KV|GPU KV cache size|kv cache group|attention block size|Model loading took|init engine|Mamba cache mode|Free memory on device|[Ss]peculat|[Dd]raft|Graph capturing" \
  "$SLOG" | sed 's/^.*\] //' | cut -c1-400 >>"$OUT"
say "correctness"
( cd "$REPO/correctness" && python3 -I compare_strata.py --out "$REPO/results/compare-$LABEL.json" ) \
  >"$REPO/results/compare-$LABEL.txt" 2>&1
echo "compare exit $?" >>"$OUT"
tail -6 "$REPO/results/compare-$LABEL.txt" >>"$OUT"
say "quick bench c$CONCS"
OUT_LEN=256 FACTOR=4 BASE_URL=http://127.0.0.1:8125 EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}' \
  bash "$REPO/bench/bench-openai.sh" "vllm-probe-$LABEL" "$CONCS" >>"$REPO/logs/bench-probe-$LABEL.log" 2>&1
python3 "$REPO/bench/summarize.py" "vllm-probe-$LABEL" | tee -a "$OUT"
grep -E "GPU KV cache usage" "$SLOG" | tail -1 | sed 's/^.*\] //' >>"$OUT"
say "PROBE_DONE"
