#!/usr/bin/env bash
# Agent `report` (final measurements for README.md). Runs ON the 5090 host in tmux edge0-report-*, holding the GPU
# lock, with no server running at the start.
#
#   bench/final-report.sh strata <label>   Strata at c1 (bench/gap-strata.sh ref, engine server direct, DETAILED=1):
#                                          4-prompt warm pass + 8 ShareGPT seed-0 prompts, replies end at EOS
#   bench/final-report.sh single1          PRESET=single from the image: c1 passes (bench/gap-vllm.sh final-v1: warm,
#                                          512 EOS-ignored, EOS), memory snapshot, compare_strata.py, greedy_same.py
#                                          gen, a 23k-token prompt, then the standard sweep as label vllm-final-single
#   bench/final-report.sh nospec           PRESET=single SPEC_METHOD=none: greedy_same.py gen, then the diff
#   bench/final-report.sh tuned            PRESET=tuned: memory snapshot, compare_strata.py, a 23k-token prompt, the
#                                          standard sweep as label vllm-final-tuned
#   bench/final-report.sh restore          PRESET=single again (bench/gap-vllm.sh final-v2, KEEP=1): the c1 passes,
#                                          and the server is left running on :8125 (tmux edge0-vllm)
#   bench/final-report.sh tables           summary tables of everything above
#   bench/final-report.sh all              strata final-s1, single1, nospec, tuned, strata final-s2, restore, tables
#
# Standard sweep: vllm-bench, ShareGPT seed 0, chat API, thinking off, temperature 0, 1024 output tokens with
# --ignore-eos, 4 x concurrency prompts, 2 warm-ups, c = 1, 2, 4, 8, 16, 32 (one vllm-bench call per point, with the
# peak GPU memory of the point (GPUMEM) and the speculative counters (SPECSTAT)). Log: logs/final-report.log.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG=$REPO/logs/final-report.log
TMP=$(mktemp -d)
mkdir -p "$REPO/logs" "$REPO/results/final"
say() { echo "# $(date +%H:%M:%S) $*" | tee -a "$LOG"; }

stop_server() {
  docker stop edge0-vllm >/dev/null 2>&1; tmux kill-session -t edge0-vllm 2>/dev/null
  for _ in $(seq 1 60); do docker ps -a --format '{{.Names}}' | grep -q '^edge0-vllm$' || break; sleep 2; done
  sleep 2
}

start_server() {   # $1 = env assignments
  stop_server
  say "start: $1"
  # shellcheck disable=SC2086
  env $1 "$REPO/serve.sh" >>"$LOG" 2>&1 || { say "serve.sh failed"; return 1; }
  local t0=$SECONDS slog
  slog=$(ls -t "$REPO"/logs/serve-*.log | head -1)
  until curl -sf localhost:8125/health >/dev/null 2>&1; do
    if (( SECONDS - t0 > 60 )) && ! docker ps --format '{{.Names}}' | grep -q '^edge0-vllm$'; then
      say "container exited during start ($slog)"; tail -30 "$slog" >>"$LOG"; return 1; fi
    (( SECONDS - t0 > 1200 )) && { say "start timed out"; return 1; }
    sleep 5
  done
  say "healthy after $((SECONDS - t0)) s ($slog)"
}

memsnap() {   # $1 = tag: GPU / host memory with the server up, and the plugin's memory lines
  local slog
  slog=$(ls -t "$REPO"/logs/serve-*.log | head -1)
  {
    echo "MEM $1 $(date -Is) serve log $slog"
    echo "MEM $1 nvidia-smi memory.used $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
    free -m | sed "s/^/MEM $1 free: /"
    grep -E '^(MemTotal|MemAvailable|Cached|Unevictable|Mlocked|AnonPages|Shmem):' /proc/meminfo | tr -s ' ' | sed "s/^/MEM $1 meminfo: /"
    grep -E "edge0: (settings|placement|built|after load|embed_tokens|lm_head|PLE layer|NCCL|MoE workspace)|edge0 tier: reactive|edge0 cpu-experts: [0-9]+ workers|Model loading took|Available KV|GPU KV cache size|Graph capturing finished|Maximum concurrency" \
      "$slog" | sed 's/^.*\] //' | cut -c1-400 | sed "s/^/MEM $1 log: /"
  } >>"$LOG" 2>&1
}

gate_compare() {   # $1 = tag
  ( cd "$REPO/correctness" && python3 -I compare_strata.py --out "$REPO/results/compare-$1.json" ) \
    >"$REPO/results/compare-$1.txt" 2>&1
  say "CORRECTNESS $1 exit $? $(grep '^SUMMARY' "$REPO/results/compare-$1.txt" | cut -c1-300)"
}

greedy_gen() {     # $1 = tag
  ( cd "$REPO/correctness" && python3 -I greedy_same.py gen --out "$REPO/results/greedy-$1.json" ) \
    >"$REPO/results/greedy-$1.txt" 2>&1
  say "GREEDY $1 exit $?"
}

long_prompt() {    # one ~23k-token prompt (6,000 words), 64 tokens out
  python3 -c "
import json, time, urllib.request
n = 6000; t0 = time.time()
body = {'model': 'edge0_145b', 'max_tokens': 64, 'temperature': 0,
        'messages': [{'role': 'user', 'content': ' '.join(f'w{i % 997}' for i in range(n)) + ' Summarize.'}],
        'chat_template_kwargs': {'enable_thinking': False}}
r = json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8125/v1/chat/completions',
    json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=900))
print('LONG ok prompt_tokens', r['usage']['prompt_tokens'], 'completion', r['usage']['completion_tokens'],
      'in %.1f s' % (time.time() - t0))" >>"$LOG" 2>&1 || say "LONG request FAILED"
  curl -sf localhost:8125/health >/dev/null && say "healthy after the long prompt" || say "SERVER DOWN after the long prompt"
}

sweep() {   # $1 = label, $2 = concurrencies
  local slog conc mpid
  slog=$(ls -t "$REPO"/logs/serve-*.log | head -1)
  echo "GPUMEM $1 idle $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) MiB" | tee -a "$LOG"
  for conc in ${2//,/ }; do
    python3 "$REPO/bench/specstat.py" snap >"$TMP/a.json"
    say "bench $1 conc $conc out 1024 prompts $((4 * conc)) (serve log line $(wc -l <"$slog"))"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 250 >"$TMP/mem.txt" 2>/dev/null &
    mpid=$!
    BASE_URL=http://127.0.0.1:8125 OUT_LEN=1024 FACTOR=4 EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}' \
      bash "$REPO/bench/bench-openai.sh" "$1" "$conc" >>"$REPO/logs/bench-$1.log" 2>&1 || say "bench $1 c$conc failed"
    kill "$mpid" 2>/dev/null; wait "$mpid" 2>/dev/null
    echo "GPUMEM $1 c$conc peak $(sort -n "$TMP/mem.txt" | tail -1) MiB" | tee -a "$LOG"
    python3 "$REPO/bench/specstat.py" snap >"$TMP/b.json"
    echo "SPECSTAT $1 c$conc $(python3 "$REPO/bench/specstat.py" diff "$TMP/a.json" "$TMP/b.json")" | tee -a "$LOG"
    python3 "$REPO/bench/summarize.py" "$1" | grep -E "^$1 +$conc " | tee -a "$LOG"
    curl -sf localhost:8125/health >/dev/null || { say "SERVER DOWN after $1 c$conc"; break; }
  done
  grep -E "GPU KV cache usage" "$slog" | sed 's/^.*Running: //' | awk '{print $1}' | sort -n | tail -1 \
    | sed "s/^/max running: $1 /" | tee -a "$LOG"
  grep -cE "Traceback|out of memory" "$slog" | sed "s/^/errors in serve log: /" | tee -a "$LOG"
  grep -E "edge0 cpu-experts: [0-9]+ jobs|edge0 tier: [0-9.]+s|edge0 adapt" "$slog" | tail -4 | sed 's/^.*\] //' | cut -c1-300 >>"$LOG"
}

phase_strata() {   # $1 = label
  stop_server
  say "strata $1"
  DETAILED=1 bash "$REPO/bench/gap-strata.sh" ref "$1" >>"$LOG" 2>&1
  say "strata $1 done: $(python3 "$REPO/bench/summarize.py" "$1" | tail -1)"
}

phase_single1() {
  stop_server
  say "single1: bench/gap-vllm.sh final-v1 PRESET=single (KEEP=1)"
  DETAILED=1 KEEP=1 bash "$REPO/bench/gap-vllm.sh" final-v1 "PRESET=single" >>"$LOG" 2>&1
  curl -sf localhost:8125/health >/dev/null 2>&1 || { say "single1: server not up after the c1 passes"; return 1; }
  memsnap single
  gate_compare final-single
  greedy_gen final-single
  long_prompt
  sweep vllm-final-single 1,2,4,8,16,32
  memsnap single-after-sweep
  stop_server
}

phase_nospec() {
  start_server "PRESET=single SPEC_METHOD=none" || return 1
  greedy_gen final-single-nospec
  stop_server
  ( cd "$REPO/correctness" && python3 -I greedy_same.py diff "$REPO/results/greedy-final-single-nospec.json" \
      "$REPO/results/greedy-final-single.json" ) >"$REPO/results/greedy-final-single-diff.txt" 2>&1
  say "GREEDY_DIFF exit $? $(tail -1 "$REPO/results/greedy-final-single-diff.txt")"
}

phase_tuned() {
  start_server "PRESET=tuned" || return 1
  memsnap tuned
  gate_compare final-tuned
  long_prompt
  sweep vllm-final-tuned 1,2,4,8,16,32
  memsnap tuned-after-sweep
  stop_server
}

phase_restore() {
  stop_server
  say "restore: bench/gap-vllm.sh final-v2 PRESET=single (KEEP=1: left running)"
  DETAILED=1 KEEP=1 bash "$REPO/bench/gap-vllm.sh" final-v2 "PRESET=single" >>"$LOG" 2>&1
  curl -sf localhost:8125/health >/dev/null 2>&1 && say "PRESET=single running on :8125" || say "RESTORE: server not up"
}

phase_tables() {
  python3 "$REPO/bench/summarize.py" strata-solo strata-batch8 vllm-base vllm-tuned vllm-final-tuned vllm-final-single \
    | tee -a "$LOG"
  python3 "$REPO/bench/summarize.py" final-s1 final-v1 final-v1-eos final-v2 final-v2-eos final-s2 | tee -a "$LOG"
  python3 "$REPO/tools/c1_compare.py" final-s1 final-s2 final-v1-eos final-v2-eos final-v1 final-v2 \
    --metrics "$REPO/results/gap" | tee -a "$LOG" "$REPO/results/final/c1-compare.txt"
}

case "${1:-}" in
  strata) phase_strata "${2:?label}" ;;
  single1) phase_single1 ;;
  nospec) phase_nospec ;;
  tuned) phase_tuned ;;
  restore) phase_restore ;;
  tables) phase_tables ;;
  all) phase_strata final-s1; phase_single1; phase_nospec; phase_tuned; phase_strata final-s2; phase_restore; phase_tables ;;
  *) echo "usage: $0 strata <label> | single1 | nospec | tuned | restore | tables | all" >&2; exit 2 ;;
esac
rm -rf "$TMP"
say "FINAL_REPORT_DONE ${1:-}"
