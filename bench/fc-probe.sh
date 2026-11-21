#!/usr/bin/env bash
# One probe server for EDGE0_TIER (agent fast-cache): start, one warm and one scored c1 pass (bench/fc-ab.sh with a
# single letter), then on the idle server the Strata comparison and the greedy text, and the tier lines of the log.
# Runs ON the 5090 host in tmux with the GPU lock.
#   bench/fc-probe.sh <label> "<serve env>"      (STOP=0 keeps the server; CORRECTNESS=0 / GREEDY=0 skip)
set -uo pipefail
LABEL=$1; ENVS=$2
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG=$REPO/logs/fcprobe-$LABEL.log
say() { echo "# $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
say "probe $LABEL: $ENVS"
WARM=${WARM:-8} PROMPTS=${PROMPTS:-8} C8=${C8:-0} KEEP=1 bash "$REPO/bench/fc-ab.sh" "$LABEL" "$ENVS" "$ENVS" A \
  >/dev/null 2>&1
tail -25 "$REPO/logs/fcab-$LABEL.log" >>"$LOG"
if curl -sf localhost:8125/health >/dev/null 2>&1; then
  if [ "${CORRECTNESS:-1}" = 1 ]; then
    ( cd "$REPO/correctness" && python3 -I compare_strata.py --out "$REPO/results/compare-$LABEL.json" ) \
      >"$REPO/results/compare-$LABEL.txt" 2>&1
    say "CORRECTNESS $LABEL exit $? $(grep '^SUMMARY' "$REPO/results/compare-$LABEL.txt" | cut -c1-260)"
  fi
  if [ "${LONG:-0}" != 0 ]; then         # LONG=<words>: one long-prompt request (prefill memory peak), then c1 again
    python3 -c "
import json, sys, time, urllib.request
n = int(sys.argv[1]); t0 = time.time()
body = {'model': 'edge0_145b', 'max_tokens': 64, 'temperature': 0,
        'messages': [{'role': 'user', 'content': ' '.join(f'w{i % 997}' for i in range(n)) + ' Summarize.'}],
        'chat_template_kwargs': {'enable_thinking': False}}
r = json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8125/v1/chat/completions',
    json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=900))
print('LONG ok prompt_tokens', r['usage']['prompt_tokens'], 'completion', r['usage']['completion_tokens'],
      'in %.1f s' % (time.time() - t0))" "$LONG" >>"$LOG" 2>&1 || say "LONG request FAILED"
    curl -sf localhost:8125/health >/dev/null && say "healthy after the long prompt" || say "SERVER DOWN after the long prompt"
  fi
  if [ "${GREEDY:-1}" = 1 ]; then
    ( cd "$REPO/correctness" && python3 -I greedy_same.py gen --out "$REPO/results/greedy-$LABEL.json" ) \
      >"$REPO/results/greedy-$LABEL.txt" 2>&1
    say "GREEDY $LABEL exit $? -> results/greedy-$LABEL.json"
  fi
fi
SLOG=$(ls -t "$REPO"/logs/serve-*.log | head -1)
grep -E "edge0 tier|Traceback|Error" "$SLOG" | sed 's/^.*\] //' | cut -c1-400 | tail -12 >>"$LOG"
if [ "${STOP:-1}" = 1 ]; then docker stop edge0-vllm >/dev/null 2>&1; tmux kill-session -t edge0-vllm 2>/dev/null; fi
say "FCPROBE_DONE $LABEL"
