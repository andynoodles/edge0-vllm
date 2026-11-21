#!/usr/bin/env bash
# Strata at concurrency 1 for docs/SINGLE-STREAM-GAP.md. Runs ON the 5090 host, in tmux, while holding the GPU lock
# and with vLLM stopped (Strata needs the whole GPU and ~52 GB of RAM).
#
#   bench/gap-strata.sh ref <label>
#       ~/work/edge0-serve.sh exactly as the reference runs it (edge0-infer serve on :8124, tmux edge0-serve); copies
#       the engine config that edge0-infer generated to results/gap/strata-config.json; warm pass + measured pass;
#       saves Strata's GET /metrics (per-request history: hit rate, PCIe share, drafts, decode ms) after each pass.
#       DIRECT=1 (default): vllm-bench talks to the engine's own server (serve/server.py on the port edge0-infer
#       printed), not to edge0-infer's relay on :8124, which vllm-bench mis-reads (TUNING-LOG "Single-stream" C1);
#       DIRECT=0 = through the relay. DETAILED=1: per-request data in the result JSON (tools/c1_compare.py).
#   bench/gap-strata.sh log <label> "<VAR=VALUE ...>" [extra engine args ...]
#       serve/server.py run directly with that config plus a "log" file, so the engine's own stderr is kept
#       (edge0-infer sends it to /dev/null): decode timing per request (STRATA_DECODE_TIMING=1), GPU stages per window
#       (STRATA_VERIFY_PROFILE=1), expert tiers, adaptive tier rounds. Same engine, same arguments.
#   bench/gap-strata.sh floor <label> [extra engine args ...]
#       the engine alone with --gpu-only-full (Strata's "true per-token GPU floor": all 48 layers + LM head, no pool)
#
# Bench: vllm-bench ShareGPT seed 0, chat API, thinking off (reasoning_effort none), temperature 0, OUT_LEN (1024, as the strata-solo reference)
# max tokens with --ignore-eos (Strata ignores it, so replies end at EOS), PROMPTS (8) at concurrency 1, 2 warm-ups.
# WARM_PROMPTS (4; 0 = none) are run first as label <label>-warm. Strata is stopped at the end (KEEP=1 keeps it).
set -uo pipefail
MODE=$1; LABEL=$2; shift 2
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SDIR=${SDIR:-$HOME/work/edge0/edge0-strata}
OUT=$REPO/results/gap
LOG=$REPO/logs/gap-strata-$LABEL.log
OUT_LEN=${OUT_LEN:-1024}; PROMPTS=${PROMPTS:-8}; WARM_PROMPTS=${WARM_PROMPTS:-4}; WAIT_S=${WAIT_S:-1500}
mkdir -p "$OUT" "$REPO/logs"
say() { echo "# $(date +%H:%M:%S) $*" | tee -a "$LOG"; }

stop_strata() {
  tmux kill-session -t edge0-serve 2>/dev/null
  tmux kill-session -t edge0-gap-strata 2>/dev/null
  pkill -f "^$SDIR/.venv/bin/python $SDIR/serve/server.py" 2>/dev/null
  pkill -f "^$HOME/work/edge0/edge0-infer/build/cpu/bin/edge0-infer serve" 2>/dev/null
  for _ in $(seq 1 60); do pgrep -f "^$SDIR/build/strata" >/dev/null || break; sleep 2; done
  pkill -9 -f "^$SDIR/build/strata" 2>/dev/null
  sleep 3
}

wait_ready() {   # $1 = url
  local t0=$SECONDS
  while (( SECONDS - t0 < WAIT_S )); do
    curl -sf -m 5 "$1/v1/models" >/dev/null 2>&1 && { say "ready after $((SECONDS - t0)) s"; return 0; }
    sleep 5
  done
  say "start timed out"; return 1
}

URL=http://127.0.0.1:8124
bench() {   # $1 = label, $2 = prompts
  say "bench $1 c1 prompts $2 out $OUT_LEN ($URL)"
  OUT_LEN=$OUT_LEN FACTOR=$2 BASE_URL=$URL EXTRA_BODY='{"reasoning_effort":"none"}' \
    bash "$REPO/bench/bench-openai.sh" "$1" 1 >>"$LOG" 2>&1 || say "bench $1 failed"
  curl -sf -m 20 "$URL/metrics" >"$OUT/$1.metrics.json" 2>/dev/null ||
    curl -s -m 20 http://127.0.0.1:8124/metrics >"$OUT/$1.metrics.json" 2>/dev/null
  python3 "$REPO/bench/summarize.py" "$1" | tail -1 | tee -a "$LOG"
}

passes() {
  [ "$WARM_PROMPTS" != 0 ] && bench "$LABEL-warm" "$WARM_PROMPTS"
  bench "$LABEL" "$PROMPTS"
}

case "$MODE" in
  ref)
    stop_strata
    say "start: ~/work/edge0-serve.sh"
    tmux new-session -d -s edge0-serve "bash $HOME/work/edge0-serve.sh 2>&1 | tee $REPO/logs/gap-strata-$LABEL.serve.log"
    wait_ready http://127.0.0.1:8124 || { stop_strata; exit 1; }
    cfg=$(pgrep -af "serve/server.py" | grep -o -- '--config [^ ]*' | head -1 | cut -d' ' -f2)
    [ -n "$cfg" ] && cp "$cfg" "$OUT/strata-config.json" && say "engine config: $cfg -> $OUT/strata-config.json"
    if [ "${DIRECT:-1}" = 1 ]; then    # bench the engine's own server (serve/server.py, HTTP/1.0, one connection per
      # request) that edge0-infer started, not edge0-infer's relay on :8124: vllm-bench reuses the relay's HTTP/1.1
      # keep-alive connection and mis-reads its streams (TUNING-LOG "Single-stream" C1: a request recorded with 1
      # token while the engine generated 609, the next one reading the previous one's tail)
      p=$(grep -o 'ready: http://127.0.0.1:[0-9]*' "$REPO/logs/gap-strata-$LABEL.serve.log" | tail -1 | grep -o '[0-9]*$')
      [ -n "$p" ] && URL=http://127.0.0.1:$p && say "DIRECT: benchmarking the engine server at $URL"
    fi
    passes
    ;;
  log)
    ENVS=$1; shift
    [ -f "$OUT/strata-config.json" ] || { say "run 'ref' first (no $OUT/strata-config.json)"; exit 1; }
    stop_strata
    ELOG=$REPO/logs/gap-strata-$LABEL.engine.log
    : >"$ELOG"
    python3 - "$OUT/strata-config.json" "$OUT/strata-config-$LABEL.json" "$ELOG" "$ENVS" "$@" <<'EOF'
import json, sys
src, dst, log, envs, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
c = json.load(open(src))
c["log"] = log
for kv in envs.split():
    k, v = kv.split("=", 1)
    c.setdefault("env", {})[k] = v
c["args"] = list(c["args"]) + extra
json.dump(c, open(dst, "w"), indent=2)
EOF
    say "start: server.py direct, env [$ENVS], extra args [$*], engine log $ELOG"
    # shellcheck disable=SC2086
    tmux new-session -d -s edge0-gap-strata "cd $SDIR && env $ENVS $SDIR/.venv/bin/python $SDIR/serve/server.py \
      --engine strata --config $OUT/strata-config-$LABEL.json --host 127.0.0.1 --port 8124 \
      2>&1 | tee $REPO/logs/gap-strata-$LABEL.serve.log"
    wait_ready http://127.0.0.1:8124 || { stop_strata; exit 1; }
    passes
    ;;
  floor)
    [ -f "$OUT/strata-config.json" ] || { say "run 'ref' first"; exit 1; }
    stop_strata
    read -r -a ARGS < <(python3 -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["args"]))' "$OUT/strata-config.json")
    say "floor: $SDIR/build/strata ${ARGS[*]} $* --gpu-only-full"
    ( cd "$SDIR" && eval "$(python3 -c 'import json,sys,shlex; print(" ".join("export %s=%s" % (k, shlex.quote(v)) for k, v in json.load(open(sys.argv[1])).get("env", {}).items()))' "$OUT/strata-config.json")" \
      && CUDA_VISIBLE_DEVICES=0 timeout 900 "$SDIR/build/strata" "${ARGS[@]}" "$@" --gpu-only-full ) >>"$LOG" 2>&1
    say "floor exit $?"; grep -A3 "GPU floor" "$LOG" | tail -4
    ;;
  *) echo "mode: ref | log | floor" >&2; exit 1 ;;
esac
[ "${KEEP:-0}" = 1 ] || stop_strata
say "GAP_STRATA_DONE $MODE $LABEL"
