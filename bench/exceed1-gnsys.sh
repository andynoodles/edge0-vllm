#!/usr/bin/env bash
# Agent exceed-1: graph-level nsys captures (accurate step period and launch cost, PROFILE.md "graph" set) of
# PRESET=single with and without EDGE0_PLE_DEFER=1, each after an 8-prompt warm pass (bench/gap-nsys.sh flow).
# Runs ON the 5090 host in tmux, holding the GPU lock.   bench/exceed1-gnsys.sh <name> "<serve env>" [...]
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
while [ $# -ge 2 ]; do
  NAME=$1; ENVS=$2; shift 2
  NSYS_OPTS="--trace=cuda,nvtx,osrt --cuda-graph-trace=graph --capture-range=cudaProfilerApi --capture-range-end=repeat --trace-fork-before-exec=true --sample=none --cpuctxsw=none --force-overwrite=true -o /workspace/edge0/profiles/$NAME/rep" \
    bash "$REPO/bench/gap-nsys.sh" "$NAME" "$ENVS"
  docker stop edge0-vllm >/dev/null 2>&1; tmux kill-session -t edge0-vllm 2>/dev/null; sleep 5
done
echo "EXCEED1_GNSYS_DONE"
