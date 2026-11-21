#!/usr/bin/env bash
# One-off chain (tune-experts agent, tmux edge0-chain): GPU tests + image rebuild, then A/B #2 (concurrency: bf16
# GDN state, no prefix caching, 32 sequences, KV sized for them) and A/B #3 (placement plan from seed-1 routing).
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG=$REPO/logs/chain-ab23.log
say() { echo "# $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 -v /home/andychen/models:/models:ro \
  -v "$REPO":/workspace/edge0 -v /home/andychen/.cache/edge0-vllm:/root/.cache \
  -e PYTHONPATH=/workspace/edge0/plugin -e EDGE0_CKPT=/models/Edge0-145B-vllm-L4 --entrypoint bash edge0-vllm:dev \
  -c 'cd /workspace/edge0/plugin && python -m pytest -q tests 2>&1 | tail -12' >"$REPO/logs/gputests-3.log" 2>&1
tail -1 "$REPO/logs/gputests-3.log" | tee -a "$LOG"
if grep -q " failed" "$REPO/logs/gputests-3.log"; then say "GPU TESTS FAILED"; exit 1; fi
( cd "$REPO" && nice -n 19 docker build -q -f docker/Dockerfile -t edge0-vllm:dev . ) >>"$LOG" 2>&1
say "image rebuilt"
KV32="KV_BYTES= EDGE0_DENSE=q8 EDGE0_EXPERT_VRAM_GIB=12.8 SSM_DTYPE=bfloat16 PREFIX_CACHING=0 MAX_SEQS=32"
bash "$REPO/bench/ab.sh" kv32 1,8,32 "KV_BYTES= EDGE0_DENSE=q8 EDGE0_EXPERT_VRAM_GIB=14.2" "$KV32" ABA
say "A/B kv32 done"
bash "$REPO/bench/ab.sh" plan 1,8,32 "$KV32" "$KV32 EDGE0_PLACEMENT=plan:/workspace/edge0/plans/plan-s1-12.8.json" ABA
say "CHAIN_DONE"
