#!/usr/bin/env bash
# vLLM at concurrency 1 for docs/SINGLE-STREAM-GAP.md (same load as bench/gap-strata.sh). Runs ON the 5090 host, in
# tmux, while holding the GPU lock and with Strata stopped.
#
#   bench/gap-vllm.sh <label> "<serve env>"
#   e.g. bench/gap-vllm.sh df3cpu "EDGE0_DEV=1 SPEC_METHOD=dflash2 SPEC_K=3 EDGE0_CPU_EXPERTS=1 EDGE0_TRACE=bw"
#
# Passes (vllm-bench ShareGPT seed 0, chat API, thinking off, temperature 0, concurrency 1, 2 warm-ups each):
#   <label>-warm  WARM_PROMPTS (8) prompts, 512 tokens, ignore EOS: warms the adaptive cache (PROFILE.md 2 caveat a)
#   <label>       PROMPTS (8) prompts, 512 tokens, ignore EOS: the A/B protocol of TUNING-LOG
#   <label>-eos   PROMPTS prompts, max 1024 tokens, stops at EOS: the reply lengths Strata produces
# Per pass: bench/specstat.py window (acceptance length), the serve log's line range (EDGE0_TRACE=bw and
# cpu-experts lines of the pass: tools/gap_parse.py vllm). The server is stopped at the end (KEEP=1 keeps it).
set -uo pipefail
LABEL=$1; ENVS=$2
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BENCH=${BENCH:-$HOME/work/edge0-bench}
OUT=$REPO/results/gap
LOG=$REPO/logs/gap-vllm-$LABEL.log
PROMPTS=${PROMPTS:-8}; WARM_PROMPTS=${WARM_PROMPTS:-8}; WAIT_S=${WAIT_S:-1200}
mkdir -p "$OUT" "$REPO/logs"
TMP=$(mktemp -d)
say() { echo "# $(date +%H:%M:%S) $*" | tee -a "$LOG"; }

stop_server() {
  docker stop edge0-vllm >/dev/null 2>&1
  tmux kill-session -t edge0-vllm 2>/dev/null
  for _ in $(seq 1 60); do docker ps -a --format '{{.Names}}' | grep -q '^edge0-vllm$' || break; sleep 2; done
  sleep 2
}

stop_server
say "start: $ENVS"
# shellcheck disable=SC2086
env $ENVS "$REPO/serve.sh" >>"$LOG" 2>&1 || { say "START FAILED"; exit 1; }
SLOG=$(ls -t "$REPO"/logs/serve-*.log | head -1)
t0=$SECONDS
until curl -sf localhost:8125/health >/dev/null 2>&1; do
  if (( SECONDS - t0 > WAIT_S )); then say "start timed out"; stop_server; exit 1; fi
  if (( SECONDS - t0 > 60 )) && ! docker ps --format '{{.Names}}' | grep -q '^edge0-vllm$'; then
    say "container exited during start ($SLOG)"; exit 1
  fi
  sleep 5
done
say "healthy after $((SECONDS - t0)) s ($SLOG)"
grep -E "edge0: (placement|after load|drafter)|edge0 cpu-experts: [0-9]+ workers|Model loading took|Available KV|GPU KV cache size" \
  "$SLOG" | sed 's/^.*\] //' | cut -c1-300 >>"$LOG"

pass() {   # $1 = label, $2 = prompts, $3 = out len, $4 = 1: ignore EOS
  local l0 l1 eos=()
  [ "$4" = 1 ] && eos=(--ignore-eos)
  [ "${DETAILED:-0}" = 1 ] && eos+=(--save-detailed)    # per-request ttfts / itls / output_lens (tools/c1_compare.py)
  python3 "$REPO/bench/specstat.py" snap >"$TMP/a.json"
  l0=$(wc -l <"$SLOG")
  say "bench $1 c1 prompts $2 out $3 ignore_eos $4 (serve log line $l0)"
  mkdir -p "$BENCH/results/$1"
  "$HOME/.local/bin/vllm-bench" --backend openai-chat --base-url http://127.0.0.1:8125 --model edge0_145b \
    --tokenizer "$BENCH/tokenizer-qwen38" \
    --dataset-name sharegpt --dataset-path "$BENCH/datasets/ShareGPT_V3_unfiltered_cleaned_split.json" \
    --sharegpt-output-len "$3" "${eos[@]}" --temperature 0 --seed 0 \
    --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' \
    --max-concurrency 1 --num-prompts "$2" --num-warmups 2 \
    --percentile-metrics ttft,tpot,itl,e2el --save-result --result-dir "$BENCH/results/$1" --label "$1" \
    --disable-tqdm >>"$LOG" 2>&1 || say "bench $1 failed"
  sleep 12      # one more EDGE0_TRACE / cpu-experts flush after the last request
  l1=$(wc -l <"$SLOG")
  python3 "$REPO/bench/specstat.py" snap >"$TMP/b.json"
  echo "SPECSTAT $1 $(python3 "$REPO/bench/specstat.py" diff "$TMP/a.json" "$TMP/b.json")" | tee -a "$LOG"
  echo "SLOG $1 $SLOG $l0 $l1" | tee -a "$LOG"
  python3 "$REPO/bench/summarize.py" "$1" | tail -1 | tee -a "$LOG"
}

[ "$WARM_PROMPTS" != 0 ] && pass "$LABEL-warm" "$WARM_PROMPTS" 512 1
pass "$LABEL" "$PROMPTS" 512 1
pass "$LABEL-eos" "$PROMPTS" 1024 0
[ "${KEEP:-0}" = 1 ] || stop_server
rm -rf "$TMP"
say "GAP_VLLM_DONE $LABEL"
