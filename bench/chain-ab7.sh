#!/usr/bin/env bash
# One-off chain (tune-experts agent, tmux edge0-chain3): after chain-ab456, GPU tests + rebuild (prefetch fix:
# persistent copy-list buffers), then A/B of the sticky next-layer prefetch on top of the kept changes
# (BEST + cold M block 64 + adaptive cache). B shrinks the plan to 12.1 GiB so total VRAM matches A.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG=$REPO/logs/chain-ab7.log
say() { echo "# $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
until grep -q CHAIN2_DONE "$REPO/logs/chain-ab456.log" 2>/dev/null; do sleep 20; done
docker stop edge0-vllm >/dev/null 2>&1; tmux kill-session -t edge0-vllm 2>/dev/null
for _ in $(seq 1 60); do docker ps -a --format '{{.Names}}' | grep -q '^edge0-vllm$' || break; sleep 2; done; sleep 2
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 -v /home/andychen/models:/models:ro \
  -v "$REPO":/workspace/edge0 -v /home/andychen/.cache/edge0-vllm:/root/.cache \
  -e PYTHONPATH=/workspace/edge0/plugin -e EDGE0_CKPT=/models/Edge0-145B-vllm-L4 --entrypoint bash edge0-vllm:dev \
  -c 'cd /workspace/edge0/plugin && python -m pytest -q tests 2>&1 | tail -12' >"$REPO/logs/gputests-4.log" 2>&1
tail -1 "$REPO/logs/gputests-4.log" | tee -a "$LOG"
if grep -q " failed" "$REPO/logs/gputests-4.log"; then say "GPU TESTS FAILED"; exit 1; fi
( cd "$REPO" && nice -n 19 docker build -q -f docker/Dockerfile -t edge0-vllm:dev . ) >>"$LOG" 2>&1
say "image rebuilt"
P=/workspace/edge0/plans
BASE="KV_BYTES= EDGE0_DENSE=q8 SSM_DTYPE=bfloat16 PREFIX_CACHING=0 MAX_SEQS=32 EDGE0_EXPERT_VRAM_GIB=12.8"
KEPT="EDGE0_COLD_BSM=64 EDGE0_BSM_MIN_TOKENS=64 EDGE0_ADAPT=1"
bash "$REPO/bench/ab.sh" prefetch2 1,8,32 "$BASE $KEPT EDGE0_PLACEMENT=plan:$P/plan-s1-12.8.json" \
  "$BASE $KEPT EDGE0_PLACEMENT=plan:$P/plan-s1-12.1.json EDGE0_PREFETCH=8" ABA
say "CHAIN3_DONE"
