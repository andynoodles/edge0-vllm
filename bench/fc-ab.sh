#!/usr/bin/env bash
# Interleaved A/B for the reactive expert tier / single-stream preset (TUNING-LOG "Single-stream", agent fast-cache).
# Runs ON the 5090 host in tmux, holding the GPU lock, with Strata stopped.
#
#   bench/fc-ab.sh <name> "<env A>" "<env B>" [order=ABBA]
#
# Per letter change: a fresh server, then one warm pass (WARM prompts at c1, not scored: fills the adaptive caches,
# PROFILE.md 2 caveat a). Per letter: one scored c1 pass (PROMPTS prompts, OUT_LEN tokens, EOS ignored) and, for the
# first A and the first B, a c8 pass (C8_PROMPTS; C8=0 skips). Per pass: acceptance (bench/specstat.py) and the serve
# log's line range (edge0 tier / trace bw / cpu-experts lines of the pass go to the A/B log). Results:
# ~/work/edge0-bench/results/<name>-<letter><i>[-c8]; summary at the end. The last server is stopped (KEEP=1 keeps it).
set -uo pipefail
NAME=$1; ENVA=$2; ENVB=$3; ORDER=${4:-ABBA}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BENCH=${BENCH:-$HOME/work/edge0-bench}
OUT_LEN=${OUT_LEN:-512}; PROMPTS=${PROMPTS:-8}; WARM=${WARM:-8}; C8=${C8:-1}; C8_PROMPTS=${C8_PROMPTS:-16}
WAIT_S=${WAIT_S:-1200}
LOG=$REPO/logs/fcab-$NAME.log
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
  SLOG=$(ls -t "$REPO"/logs/serve-*.log | head -1)
  local t0=$SECONDS
  while (( SECONDS - t0 < WAIT_S )); do
    if curl -sf localhost:8125/health >/dev/null 2>&1; then
      say "healthy after $((SECONDS - t0)) s ($SLOG)"
      grep -E "edge0: (placement|after load)|edge0 tier: reactive|Model loading took|Available KV|GPU KV cache size|graph pool|Free memory on device|reserved .* KV" \
        "$SLOG" | sed 's/^.*\] //' | cut -c1-400 >>"$LOG"
      return 0
    fi
    if (( SECONDS - t0 > 60 )) && ! docker ps --format '{{.Names}}' | grep -q '^edge0-vllm$'; then
      say "container exited during start"; grep -E "Error|error" "$SLOG" | tail -5 >>"$LOG"; return 1
    fi
    sleep 5
  done
  say "start timed out"; return 1
}

pass() {   # $1 label, $2 concurrency, $3 prompts
  local l0 l1
  python3 "$REPO/bench/specstat.py" snap >"$TMP/a.json"
  l0=$(wc -l <"$SLOG")
  say "bench $1 c$2 prompts $3 out $OUT_LEN"
  mkdir -p "$BENCH/results/$1"
  "$HOME/.local/bin/vllm-bench" --backend openai-chat --base-url http://127.0.0.1:8125 --model edge0_145b \
    --tokenizer "$BENCH/tokenizer-qwen38" \
    --dataset-name sharegpt --dataset-path "$BENCH/datasets/ShareGPT_V3_unfiltered_cleaned_split.json" \
    --sharegpt-output-len "$OUT_LEN" --ignore-eos --temperature 0 --seed 0 \
    --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' \
    --max-concurrency "$2" --num-prompts "$3" --num-warmups 2 \
    --percentile-metrics ttft,tpot,itl,e2el --save-result --result-dir "$BENCH/results/$1" --label "$1" \
    --disable-tqdm >>"$LOG" 2>&1 || say "bench $1 failed"
  sleep 11      # one more tier / trace / cpu-experts log line after the last request
  l1=$(wc -l <"$SLOG")
  python3 "$REPO/bench/specstat.py" snap >"$TMP/b.json"
  echo "SPECSTAT $1 $(python3 "$REPO/bench/specstat.py" diff "$TMP/a.json" "$TMP/b.json")" | tee -a "$LOG"
  echo "SLOG $1 $SLOG $l0 $l1" | tee -a "$LOG"
  sed -n "$((l0 + 1)),${l1}p" "$SLOG" | grep -E "edge0 tier:|edge0 trace bw|edge0 cpu-experts: [0-9]+ jobs|edge0 adapt: epoch|edge0 ple-defer" \
    | sed 's/^.*\] //' | cut -c1-420 | sed "s/^/  [$1] /" >>"$LOG"
  python3 "$REPO/bench/summarize.py" "$1" | tail -1 | tee -a "$LOG"
}

prev=""; c8a=0; c8b=0; labels=(); labels8=()
for ((k = 0; k < ${#ORDER}; k++)); do
  c=${ORDER:$k:1}
  envs=$ENVA; [ "$c" = B ] && envs=$ENVB
  if [ "$c" != "$prev" ]; then
    start_server "$envs" || { say "START FAILED ($c)"; stop_server; exit 1; }
    prev=$c
    [ "$WARM" != 0 ] && pass "$NAME-$c$((k + 1))-warm" 1 "$WARM"
  fi
  label="$NAME-$c$((k + 1))"; labels+=("$label")
  pass "$label" 1 "$PROMPTS"
  if [ "$C8" = 1 ]; then
    if [ "$c" = A ] && [ $c8a = 0 ]; then c8a=1; pass "$label-c8" 8 "$C8_PROMPTS"; labels8+=("$label-c8"); fi
    if [ "$c" = B ] && [ $c8b = 0 ]; then c8b=1; pass "$label-c8" 8 "$C8_PROMPTS"; labels8+=("$label-c8"); fi
  fi
done
python3 "$REPO/bench/summarize.py" "${labels[@]}" "${labels8[@]}" | tee -a "$LOG"
[ "${KEEP:-0}" = 1 ] || stop_server
rm -rf "$TMP"
say "FCAB_DONE $NAME"
