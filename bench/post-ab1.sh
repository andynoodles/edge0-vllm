#!/usr/bin/env bash
# One-off chain for the tune-experts agent (runs ON the 5090 in tmux edge0-chain): after A/B #1 (q8) finishes,
# 1. long-generation outputs on the fp32-state server left running by ab.sh (bring-up config),
# 2. routing capture with *other* ShareGPT prompts (seed 1) on the 32-sequence config (q8, bf16 GDN state, no prefix
#    caching), then the same long generations there (bf16 state), and the dtype check scoring,
# 3. GPU tests + image rebuild (sticky prefetch), 4. A/B #2 (concurrency).
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG=$REPO/logs/post-ab1.log
say() { echo "# $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
until grep -q AB_DONE "$REPO/logs/ab-q8.log" 2>/dev/null; do sleep 20; done
say "ab q8 done"
cd "$REPO/correctness" && python3 -I long_state_check.py gen --out "$REPO/results/long-fp32.json" >>"$LOG" 2>&1
say "long gen fp32 done"
KV32="EDGE0_DENSE=q8 EDGE0_EXPERT_VRAM_GIB=12.8 SSM_DTYPE=bfloat16 PREFIX_CACHING=0 MAX_SEQS=32"
# shellcheck disable=SC2086
SEED=1 CAP_FACTOR=2 bash "$REPO/bench/capture-routes.sh" routes-s1 1,8,32 $KV32
say "capture seed 1 done"
cd "$REPO/correctness" && python3 -I long_state_check.py gen --out "$REPO/results/long-bf16.json" >>"$LOG" 2>&1
python3 -I long_state_check.py score "$REPO/results/long-fp32.json" "$REPO/results/long-bf16.json" \
  >"$REPO/results/long-state.txt" 2>&1
cat "$REPO/results/long-state.txt" >>"$LOG"
say "long-state check done"
docker stop edge0-vllm >/dev/null 2>&1; tmux kill-session -t edge0-vllm 2>/dev/null
for _ in $(seq 1 60); do docker ps -a --format '{{.Names}}' | grep -q '^edge0-vllm$' || break; sleep 2; done; sleep 2
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 -v /home/andychen/models:/models:ro \
  -v "$REPO":/workspace/edge0 -v /home/andychen/.cache/edge0-vllm:/root/.cache \
  -e PYTHONPATH=/workspace/edge0/plugin -e EDGE0_CKPT=/models/Edge0-145B-vllm-L4 --entrypoint bash edge0-vllm:dev \
  -c 'cd /workspace/edge0/plugin && python -m pytest -q tests 2>&1 | tail -12' >"$REPO/logs/gputests-2.log" 2>&1
tail -2 "$REPO/logs/gputests-2.log" | tee -a "$LOG"
if grep -q " failed" "$REPO/logs/gputests-2.log"; then say "GPU TESTS FAILED: image not rebuilt, chain stops"; exit 1; fi
( cd "$REPO" && nice -n 19 docker build -q -f docker/Dockerfile -t edge0-vllm:dev . ) >>"$LOG" 2>&1
say "image rebuilt; A/B #2 (kv32)"
bash "$REPO/bench/ab.sh" kv32 1,8,32 "KV_BYTES= EDGE0_DENSE=q8 EDGE0_EXPERT_VRAM_GIB=14.2" "KV_BYTES= $KV32" ABA
say "CHAIN_DONE"
