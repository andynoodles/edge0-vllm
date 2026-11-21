#!/usr/bin/env bash
# One-off (tune-experts agent, tmux edge0-final): start the tuned default (./serve.sh, PRESET=tuned), check
# correctness on the idle server, then the full standard sweep (1024 output tokens) as label vllm-tuned. The server
# is left running.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG=$REPO/logs/final-tuned.log
say() { echo "# $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
docker stop edge0-vllm >/dev/null 2>&1; tmux kill-session -t edge0-vllm 2>/dev/null
for _ in $(seq 1 60); do docker ps -a --format '{{.Names}}' | grep -q '^edge0-vllm$' || break; sleep 2; done; sleep 2
say "start tuned default"
"$REPO/serve.sh" >>"$LOG" 2>&1 || { say "serve.sh failed"; exit 1; }
t0=$SECONDS
until curl -sf localhost:8125/health >/dev/null 2>&1; do
  (( SECONDS - t0 > 900 )) && { say "start timed out"; exit 1; }
  sleep 5
done
SLOG=$(ls -t "$REPO"/logs/serve-*.log | head -1)
say "healthy after $((SECONDS - t0)) s ($SLOG)"
grep -E "edge0: (settings|placement|built|after load)|edge0 adapt: adaptive|Available KV|GPU KV cache size|attention block" \
  "$SLOG" | sed 's/^.*\] //' | cut -c1-300 >>"$LOG"
( cd "$REPO/correctness" && python3 -I compare_strata.py --out "$REPO/results/compare-tuned.json" ) \
  >"$REPO/results/compare-tuned.txt" 2>&1
say "compare exit $?"
grep SUMMARY "$REPO/results/compare-tuned.txt" >>"$LOG"
BASE_URL=http://127.0.0.1:8125 OUT_LEN=1024 EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}' \
  bash "$REPO/bench/bench-openai.sh" vllm-tuned 1,2,4,8,16,32 >"$REPO/logs/bench-vllm-tuned.log" 2>&1
say "bench exit $?"
python3 "$REPO/bench/summarize.py" vllm-base vllm-tuned | tee -a "$LOG"
grep -E "Running:" "$SLOG" | sed 's/^.*Running/Running/' | sort | uniq -c | sort -rn | head -5 >>"$LOG"
say "FINAL_DONE"
