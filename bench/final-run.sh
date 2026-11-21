#!/usr/bin/env bash
# One-off (agent `final`, tmux edge0-final-run): the final default (PRESET=tuned, now with CPU cold experts) and the
# opt-in low-concurrency setting (+ DFlash2 k=3). Runs ON the 5090 host under the GPU lock.
#
#   bench/final-run.sh main      start the default, compare_strata + greedy text (no drafter), full standard sweep
#                                (1024 output tokens, c1..c32) as label vllm-final
#   bench/final-run.sh spec      start default + SPEC_METHOD=dflash2 SPEC_K=3, compare_strata + greedy text, greedy
#                                diff against main's, sweep c1..c8 (1024 output tokens) as label vllm-final-spec
#   bench/final-run.sh restore   start the default again and leave it running on :8125 (tmux edge0-vllm)
#   bench/final-run.sh all       main, spec, restore
# Log: logs/final-run.log; peak GPU memory per sweep point (GPUMEM), speculative counters (SPECSTAT).
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG=$REPO/logs/final-run.log
TMP=$(mktemp -d)
mkdir -p "$REPO/logs" "$REPO/results"
say() { echo "# $(date +%H:%M:%S) $*" | tee -a "$LOG"; }

stop_server() {
  docker stop edge0-vllm >/dev/null 2>&1
  tmux kill-session -t edge0-vllm 2>/dev/null
  for _ in $(seq 1 60); do docker ps -a --format '{{.Names}}' | grep -q '^edge0-vllm$' || break; sleep 2; done
  sleep 2
}

start_server() {   # $1 = env assignments (may be empty)
  stop_server
  say "start: PRESET=tuned ${1:-(defaults)}"
  # shellcheck disable=SC2086
  env $1 "$REPO/serve.sh" >>"$LOG" 2>&1 || return 1
  local t0=$SECONDS slog
  slog=$(ls -t "$REPO"/logs/serve-*.log | head -1)
  while (( SECONDS - t0 < 1200 )); do
    if curl -sf localhost:8125/health >/dev/null 2>&1; then
      say "healthy after $((SECONDS - t0)) s ($slog)"
      grep -E "edge0: (settings|placement|built|after load|drafter|cpu)|edge0 cpu|edge0 adapt: adaptive|Model loading took|Available KV|GPU KV cache size|attention block size" \
        "$slog" | sed 's/^.*\] //' | cut -c1-300 >>"$LOG"
      head -1 "$slog" | cut -c1-2000 >>"$LOG"
      return 0
    fi
    if (( SECONDS - t0 > 60 )) && ! docker ps --format '{{.Names}}' | grep -q '^edge0-vllm$'; then
      say "container exited during start"; grep -E "Error|error" "$slog" | tail -5 >>"$LOG"; return 1
    fi
    sleep 5
  done
  say "start timed out"; return 1
}

check() {   # $1 = label: compare_strata + greedy text on the idle server
  ( cd "$REPO/correctness" && python3 -I compare_strata.py --out "$REPO/results/compare-$1.json" ) \
    >"$REPO/results/compare-$1.txt" 2>&1
  echo "CORRECTNESS $1 exit $? $(grep '^SUMMARY' "$REPO/results/compare-$1.txt" | cut -c1-240)" | tee -a "$LOG"
  ( cd "$REPO/correctness" && python3 -I greedy_same.py gen --out "$REPO/results/greedy-$1.json" ) \
    >"$REPO/results/greedy-$1.txt" 2>&1
  echo "GREEDY $1 exit $? -> results/greedy-$1.json" | tee -a "$LOG"
}

sweep() {   # $1 = label, $2 = concurrencies (one vllm-bench call per concurrency, for per-point GPUMEM / SPECSTAT)
  echo "GPUMEM $1 idle $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) MiB" | tee -a "$LOG"
  for conc in ${2//,/ }; do
    python3 "$REPO/bench/specstat.py" snap >"$TMP/a.json"
    say "bench $1 conc $conc out 1024 prompts $((4 * conc))"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 250 >"$TMP/mem.txt" 2>/dev/null &
    local mpid=$!
    BASE_URL=http://127.0.0.1:8125 OUT_LEN=1024 FACTOR=4 EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}' \
      bash "$REPO/bench/bench-openai.sh" "$1" "$conc" >>"$REPO/logs/bench-$1.log" 2>&1 || say "bench $1 c$conc failed"
    kill "$mpid" 2>/dev/null; wait "$mpid" 2>/dev/null
    echo "GPUMEM $1 c$conc peak $(sort -n "$TMP/mem.txt" | tail -1) MiB" | tee -a "$LOG"
    python3 "$REPO/bench/specstat.py" snap >"$TMP/b.json"
    echo "SPECSTAT $1 c$conc $(python3 "$REPO/bench/specstat.py" diff "$TMP/a.json" "$TMP/b.json")" | tee -a "$LOG"
  done
  local slog
  slog=$(ls -t "$REPO"/logs/serve-*.log | head -1)
  grep -E "GPU KV cache usage" "$slog" | sed 's/^.*\] //' | awk -F'Running: ' '{print $2}' | sort -t' ' -k1,1n | tail -1 \
    | sed "s/^/max running: $1 /" | tee -a "$LOG"
  grep -E "edge0 cpu" "$slog" | tail -3 | sed 's/^.*\] //' | cut -c1-300 >>"$LOG"
}

phase_main() {
  start_server "" || { say "START FAILED (main)"; return 1; }
  check final-nospec
  sweep vllm-final 1,2,4,8,16,32
  python3 "$REPO/bench/summarize.py" strata-solo strata-batch8 vllm-base vllm-tuned vllm-final | tee -a "$LOG"
}

phase_spec() {
  start_server "SPEC_METHOD=dflash2 SPEC_K=3" || { say "START FAILED (spec)"; return 1; }
  check final-spec
  python3 -I "$REPO/correctness/greedy_same.py" diff "$REPO/results/greedy-final-nospec.json" \
    "$REPO/results/greedy-final-spec.json" >"$REPO/results/greedy-final-diff.txt" 2>&1
  echo "GREEDY_DIFF exit $? $(tail -3 "$REPO/results/greedy-final-diff.txt" | tr '\n' ' ' | cut -c1-300)" | tee -a "$LOG"
  sweep vllm-final-spec 1,2,4,8
  python3 "$REPO/bench/summarize.py" vllm-final vllm-final-spec | tee -a "$LOG"
}

case "${1:-all}" in
  main) phase_main ;;
  spec) phase_spec ;;
  restore) start_server "" ;;
  all) phase_main; phase_spec; start_server "" ;;
  *) echo "usage: $0 main|spec|restore|all" >&2; exit 2 ;;
esac
rm -rf "$TMP"
say "FINAL_RUN_DONE ${1:-all}"
