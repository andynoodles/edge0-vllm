#!/usr/bin/env bash
# One-off chain for the tune-experts agent (runs ON the 5090 in tmux edge0-post): wait for the vllm-base sweep,
# stop the server, run the GPU unit tests, the expert-IO microbenchmarks, then the routing capture.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG=$REPO/logs/post-baseline.log
say() { echo "# $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
until grep -q BENCH_EXIT "$REPO/logs/bench-vllm-base.log" 2>/dev/null; do sleep 20; done
say "baseline done: $(grep BENCH_EXIT "$REPO/logs/bench-vllm-base.log")"
docker stop edge0-vllm >/dev/null 2>&1; tmux kill-session -t edge0-vllm 2>/dev/null; sleep 5
docker tag edge0-vllm:dev edge0-vllm:bringup-1535 2>/dev/null
say "gpu tests"
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 -v /home/andychen/models:/models:ro \
  -v "$REPO":/workspace/edge0 -v /home/andychen/.cache/edge0-vllm:/root/.cache \
  -e PYTHONPATH=/workspace/edge0/plugin -e EDGE0_CKPT=/models/Edge0-145B-vllm-L4 --entrypoint bash edge0-vllm:dev \
  -c 'cd /workspace/edge0/plugin && python -m pytest -q -x tests 2>&1 | tail -15' >"$REPO/logs/gputests.log" 2>&1
tail -3 "$REPO/logs/gputests.log" | tee -a "$LOG"
say "microbenchmarks"
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 -v "$REPO":/workspace/edge0 \
  -v /home/andychen/.cache/edge0-vllm:/root/.cache -e PYTHONPATH=/workspace/edge0/plugin --entrypoint python \
  edge0-vllm:dev /workspace/edge0/tools/bench_expert_io.py --json /workspace/edge0/logs/expert-io.json \
  >"$REPO/logs/expert-io.txt" 2>&1
tail -3 "$REPO/logs/expert-io.txt" | tee -a "$LOG"
say "routes capture"
bash "$REPO/bench/capture-routes.sh" routes 1,4,16 EDGE0_DEV=1
say "POST_DONE"
