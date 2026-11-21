#!/usr/bin/env bash
# Run one bench pass per server configuration, in order (calibration sweeps; for decisions use bench/ab.sh's ABA).
# Runs ON the 5090 host, ideally in tmux.
#
#   bench/run-configs.sh <name> <concs> "<env 1>" ["<env 2>" ...]
#   e.g. bench/run-configs.sh mtpk 1,4,8 "SPEC_METHOD=mtp SPEC_K=1" "SPEC_METHOD=mtp SPEC_K=2"
#
# Config i is benchmarked as label <name>-<i>; per concurrency the speculative-decoding counters of the bench window
# (bench/specstat.py: acceptance length, per-position acceptance) and the server's KV sizing lines go to
# logs/runcfg-<name>.log. Env: OUT_LEN (512), FACTOR (2; FACTOR_C1 4), WAIT_S (1200). The last server keeps running.
# Also logged per concurrency: GPUMEM = peak nvidia-smi memory.used during the bench (MiB; the caching allocator keeps
# its high-water mark, so this is the step peak incl. non-torch memory) and scheduler preemptions (SPECSTAT).
# CORRECTNESS=1: compare_strata.py on the idle server; GREEDY=1: correctness/greedy_same.py gen (8 strata-ref prompts,
# 256 greedy tokens) -> results/greedy-<label>.json, to diff against the no-drafter server's file.
# STOP_AFTER=1: stop the last server at the end (releases the GPU).
set -uo pipefail
NAME=$1; CONCS=$2; shift 2
OUT_LEN=${OUT_LEN:-512}; FACTOR=${FACTOR:-2}; WAIT_S=${WAIT_S:-1200}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG=$REPO/logs/runcfg-$NAME.log
TMP=$(mktemp -d)
mkdir -p "$REPO/logs"
say() { echo "# $(date +%H:%M:%S) $*" | tee -a "$LOG"; }

stop_server() {
  docker stop edge0-vllm >/dev/null 2>&1
  tmux kill-session -t edge0-vllm 2>/dev/null
  for _ in $(seq 1 60); do docker ps -a --format '{{.Names}}' | grep -q '^edge0-vllm$' || break; sleep 2; done
  sleep 2
}

start_server() {
  stop_server
  say "start: $1"
  # shellcheck disable=SC2086
  env $1 "$REPO/serve.sh" >>"$LOG" 2>&1 || return 1
  local t0=$SECONDS slog
  slog=$(ls -t "$REPO"/logs/serve-*.log | head -1)
  while (( SECONDS - t0 < WAIT_S )); do
    if curl -sf localhost:8125/health >/dev/null 2>&1; then
      say "healthy after $((SECONDS - t0)) s ($slog)"
      grep -E "edge0: (placement|after load|drafter)|Model loading took|Available KV|GPU KV cache size|attention block size|Free memory on device" \
        "$slog" | sed 's/^.*\] //' | cut -c1-300 >>"$LOG"
      return 0
    fi
    if (( SECONDS - t0 > 60 )) && ! docker ps --format '{{.Names}}' | grep -q '^edge0-vllm$'; then
      say "container exited during start"; grep -E "Error|error" "$slog" | tail -5 >>"$LOG"; return 1
    fi
    sleep 5
  done
  say "start timed out"; return 1
}

labels=()
i=0
for envs in "$@"; do
  i=$((i + 1)); label="$NAME-$i"; labels+=("$label")
  start_server "$envs" || { say "START FAILED ($label)"; continue; }
  if [ "${CORRECTNESS:-0}" = 1 ]; then     # Strata reference comparison on the idle server (~15 s)
    ( cd "$REPO/correctness" && python3 -I compare_strata.py --out "$REPO/results/compare-$label.json" ) \
      >"$REPO/results/compare-$label.txt" 2>&1
    echo "CORRECTNESS $label exit $? $(grep '^SUMMARY' "$REPO/results/compare-$label.txt" | cut -c1-220)" | tee -a "$LOG"
  fi
  if [ "${GREEDY:-0}" = 1 ]; then          # greedy text for the with/without-drafter identity gate (~40 s)
    ( cd "$REPO/correctness" && python3 -I greedy_same.py gen --out "$REPO/results/greedy-$label.json" ) \
      >"$REPO/results/greedy-$label.txt" 2>&1
    echo "GREEDY $label exit $? -> results/greedy-$label.json" | tee -a "$LOG"
  fi
  echo "GPUMEM $label idle $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) MiB" | tee -a "$LOG"
  for conc in ${CONCS//,/ }; do
    f=$FACTOR; [ "$conc" = 1 ] && f=${FACTOR_C1:-4}
    ol=$OUT_LEN
    v="FACTOR_C$conc"; [ -n "${!v:-}" ] && f=${!v}          # per-concurrency overrides, e.g. FACTOR_C32=1
    v="OUT_LEN_C$conc"; [ -n "${!v:-}" ] && ol=${!v}        # e.g. OUT_LEN_C32=1024
    python3 "$REPO/bench/specstat.py" snap >"$TMP/a.json"
    say "bench $label conc $conc out $ol prompts $((f * conc))"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 250 >"$TMP/mem.txt" 2>/dev/null &
    mpid=$!
    OUT_LEN=$ol FACTOR=$f BASE_URL=http://127.0.0.1:8125 \
      EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}' \
      bash "$REPO/bench/bench-openai.sh" "$label" "$conc" >>"$LOG" 2>&1 || say "bench $label c$conc failed"
    kill "$mpid" 2>/dev/null; wait "$mpid" 2>/dev/null
    echo "GPUMEM $label c$conc peak $(sort -n "$TMP/mem.txt" | tail -1) MiB" | tee -a "$LOG"
    python3 "$REPO/bench/specstat.py" snap >"$TMP/b.json"
    echo "SPECSTAT $label c$conc $(python3 "$REPO/bench/specstat.py" diff "$TMP/a.json" "$TMP/b.json")" | tee -a "$LOG"
  done
  slog=$(ls -t "$REPO"/logs/serve-*.log | head -1)
  grep -E "GPU KV cache usage" "$slog" | sed 's/^.*\] //' | awk -F'Running: ' '{print $2}' | sort -t' ' -k1,1n | tail -1 \
    | sed "s/^/max running: $label /" >>"$LOG"
done
python3 "$REPO/bench/summarize.py" "${labels[@]}" | tee -a "$LOG"
[ "${STOP_AFTER:-0}" = 1 ] && stop_server
rm -rf "$TMP"
say "RUNCFG_DONE"
