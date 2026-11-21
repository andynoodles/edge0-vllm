#!/usr/bin/env bash
# Interleaved A/B driver (TidalMoE protocol: A B B A, noise ~±2.5%). Runs ON the 5090 host, ideally in tmux.
#
#   bench/ab.sh <name> <concs> "<env A>" "<env B>" [order=ABBA]
#   e.g. bench/ab.sh q8 1,8,32 "KV_BYTES=" "KV_BYTES= EDGE0_DENSE=q8 EDGE0_EXPERT_VRAM_GIB=14.5"
#
# <env X>: space-separated VAR=VALUE assignments handed to serve.sh (no spaces inside values). Each letter of the
# order is one bench pass (results/<name>-<letter><pass#>); a server is (re)started only when the letter changes,
# so ABBA = 3 starts. Env: OUT_LEN (512), FACTOR (prompts = FACTOR x conc, default 2; FACTOR_C1 for conc 1,
# default 4), WAIT_S (start timeout, 900), WARM_PROMPTS (0; N = an unrecorded c1 warm pass after each server start).
# After the passes: bench/summarize.py over all passes. The last server is left running.
set -uo pipefail
NAME=$1; CONCS=$2; ENVA=$3; ENVB=$4; ORDER=${5:-ABBA}
OUT_LEN=${OUT_LEN:-512}; FACTOR=${FACTOR:-2}; WAIT_S=${WAIT_S:-900}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG=$REPO/logs/ab-$NAME.log
mkdir -p "$REPO/logs"
say() { echo "# $(date +%H:%M:%S) $*" | tee -a "$LOG"; }

stop_server() {   # the container runs with --rm: wait until docker has removed it, or the next start conflicts
  docker stop edge0-vllm >/dev/null 2>&1
  tmux kill-session -t edge0-vllm 2>/dev/null
  for _ in $(seq 1 60); do docker ps -a --format '{{.Names}}' | grep -q '^edge0-vllm$' || break; sleep 2; done
  sleep 2
}

start_server() {   # $1 = env assignments
  stop_server
  say "start: $1"
  # shellcheck disable=SC2086
  env $1 "$REPO/serve.sh" >>"$LOG" 2>&1 || return 1
  local t0=$SECONDS
  while (( SECONDS - t0 < WAIT_S )); do
    curl -sf localhost:8125/health >/dev/null 2>&1 && { say "healthy after $((SECONDS - t0)) s"; return 0; }
    if (( SECONDS - t0 > 60 )) && ! docker ps --format '{{.Names}}' | grep -q '^edge0-vllm$'; then
      say "container exited during start"; return 1
    fi
    sleep 5
  done
  say "start timed out"; return 1
}

prev=""
for ((k = 0; k < ${#ORDER}; k++)); do
  c=${ORDER:$k:1}
  envs=$ENVA; [ "$c" = B ] && envs=$ENVB
  if [ "$c" != "$prev" ]; then
    start_server "$envs" || { say "START FAILED ($c)"; exit 1; }
    prev=$c
    if [ "${WARM_PROMPTS:-0}" -gt 0 ]; then   # WARM_PROMPTS=N: an unrecorded c1 pass after each start (adaptive cache)
      say "warm $NAME-$c$((k + 1)) c1 prompts $WARM_PROMPTS"
      OUT_LEN=$OUT_LEN FACTOR=$WARM_PROMPTS BASE_URL=http://127.0.0.1:8125 \
        EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}' \
        bash "$REPO/bench/bench-openai.sh" "$NAME-warm$((k + 1))" 1 >>"$LOG" 2>&1 || say "warm pass failed"
    fi
  fi
  label="$NAME-$c$((k + 1))"
  for conc in ${CONCS//,/ }; do
    f=$FACTOR; [ "$conc" = 1 ] && f=${FACTOR_C1:-4}
    say "bench $label conc $conc out $OUT_LEN prompts $((f * conc))"
    OUT_LEN=$OUT_LEN FACTOR=$f BASE_URL=http://127.0.0.1:8125 \
      EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}' \
      bash "$REPO/bench/bench-openai.sh" "$label" "$conc" >>"$LOG" 2>&1 || say "bench $label c$conc failed"
  done
done
labels=()
for ((k = 0; k < ${#ORDER}; k++)); do labels+=("$NAME-${ORDER:$k:1}$((k + 1))"); done
python3 "$REPO/bench/summarize.py" "${labels[@]}" | tee -a "$LOG"
say "AB_DONE"
