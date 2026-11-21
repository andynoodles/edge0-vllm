#!/usr/bin/env bash
# Nsight capture of the best c1 vLLM config with a warm adaptive cache (docs/SINGLE-STREAM-GAP.md). Runs ON the 5090
# host, in tmux, holding the GPU lock, Strata stopped.
#   bench/gap-nsys.sh <name> "<serve env>"
# 1) bench/nsys-capture.sh starts the server under nsys (node-level graph trace) and takes one c1 capture (cold
#    cache); 2) an 8-prompt c1 warm pass (512 tokens, ignore EOS); 3) a second c1 capture on the same server (REUSE=1),
#    then the server is stopped. Reports: profiles/<name>/, profiles/<name>w/; tools/nsys_breakdown.py steps on them.
set -uo pipefail
NAME=$1; ENVS=$2
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
KEEP=1 GPUM=0 bash "$REPO/bench/nsys-capture.sh" "$NAME" "$ENVS" 1:150:10 || exit 1
OUT_LEN=512 FACTOR=8 BASE_URL=http://127.0.0.1:8125 EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}' \
  bash "$REPO/bench/bench-openai.sh" "$NAME-warm" 1 >>"$REPO/logs/nsys-$NAME.log" 2>&1
python3 "$REPO/bench/summarize.py" "$NAME-warm" | tail -1 | tee -a "$REPO/logs/nsys-$NAME.log"
REUSE=1 GPUM=1 bash "$REPO/bench/nsys-capture.sh" "${NAME}w" "$ENVS" 1:150:10
echo "GAP_NSYS_DONE $NAME" | tee -a "$REPO/logs/nsys-$NAME.log"
