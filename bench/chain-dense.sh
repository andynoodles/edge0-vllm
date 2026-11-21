#!/usr/bin/env bash
# Single-stream dense-path experiments (TUNING-LOG "Single-stream", agent dense-c1). Runs ON the 5090 host in tmux,
# holding the GPU lock, with no server running.
#
#   bench/chain-dense.sh kbench            # q8gemv GPU tests + microbenchmark vs b12x (no server)
#   bench/chain-dense.sh probe <label> "<env>"   # bench/probe-config.sh (correctness + quick c1), then stop
#   bench/chain-dense.sh ab <name> <concs> "<env A>" "<env B>"   # bench/ab.sh ABBA with FACTOR_C1=8, then stop
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG=$REPO/logs/chain-dense.log
say() { echo "# $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
stop_server() {
  docker stop edge0-vllm >/dev/null 2>&1; tmux kill-session -t edge0-vllm 2>/dev/null
  for _ in $(seq 1 60); do docker ps -a --format '{{.Names}}' | grep -q '^edge0-vllm$' || break; sleep 2; done
}
case "$1" in
  kbench)
    say "kbench start"
    docker run --rm --gpus all --ipc=host -v "$REPO:/workspace/edge0" -v /home/andychen/.cache/edge0-vllm:/root/.cache \
      -e PYTHONPATH=/workspace/edge0/plugin --entrypoint bash edge0-vllm:dev -c \
      "cd /workspace/edge0/plugin && python -m pytest -q tests/test_q8gemv.py 2>&1 | tail -15 &&
       python3 /workspace/edge0/tools/q8gemv_bench.py --m 1,2,4,8 --mask --shared --shapes D,U,qkvz,gdn_out,sh_gu,sh_down,lm_head ${KB_ARGS:---cfg 16:1,32:2,64:1,64:2,16:4} &&
       python3 /workspace/edge0/tools/q8gemv_bench.py --m 1,4 --shapes D,U,sh_gu --nopdl" 2>&1 | tee -a "$LOG"
    say "kbench done" ;;
  probe)
    bash "$REPO/bench/probe-config.sh" "$2" "$3"; tail -30 "$REPO/logs/probe-$2.log" >>"$LOG"; stop_server
    say "probe $2 done" ;;
  ab)
    FACTOR_C1=${FACTOR_C1:-8} bash "$REPO/bench/ab.sh" "$2" "$3" "$4" "$5" "${6:-ABBA}"
    tail -25 "$REPO/logs/ab-$2.log" >>"$LOG"; stop_server
    say "ab $2 done" ;;
  gate)   # gate <label> "<env>" [strata=1]: start, (compare_strata), greedy_same gen (256 tok), quick c1, stop
    stop_server
    say "gate $2 start: $3"
    # shellcheck disable=SC2086
    env $3 "$REPO/serve.sh" >>"$LOG" 2>&1 || { say "serve.sh failed"; exit 1; }
    SLOG=$(ls -t "$REPO"/logs/serve-*.log | head -1); t0=$SECONDS
    until curl -sf localhost:8125/health >/dev/null 2>&1; do
      if (( SECONDS - t0 > 60 )) && ! docker ps --format '{{.Names}}' | grep -q '^edge0-vllm$'; then
        say "container exited"; tail -40 "$SLOG" >>"$LOG"; exit 1; fi
      (( SECONDS - t0 > 1500 )) && { say "start timed out"; stop_server; exit 1; }
      sleep 5
    done
    say "healthy after $((SECONDS - t0)) s ($SLOG)"
    grep -E "edge0: (Q8_0 linears|latent D|lm_head|after load)|GPU KV cache size|Traceback|Error" "$SLOG" | \
      sed 's/^.*\] //' | cut -c1-300 >>"$LOG"
    if [ "${4:-1}" = 1 ]; then
      ( cd "$REPO/correctness" && python3 -I compare_strata.py --out "$REPO/results/compare-$2.json" ) \
        >"$REPO/results/compare-$2.txt" 2>&1
      tail -6 "$REPO/results/compare-$2.txt" >>"$LOG"
    fi
    ( cd "$REPO/correctness" && python3 -I greedy_same.py gen --out "$REPO/results/greedy-$2.json" ) >>"$LOG" 2>&1
    OUT_LEN=512 FACTOR=8 BASE_URL=http://127.0.0.1:8125 EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}' \
      bash "$REPO/bench/bench-openai.sh" "dense-gate-$2" 1 >>"$REPO/logs/bench-dense-gate-$2.log" 2>&1
    python3 "$REPO/bench/summarize.py" "dense-gate-$2" | tee -a "$LOG"
    grep -E "Traceback|Error|timeout" "$SLOG" | head -5 >>"$LOG"
    stop_server
    say "gate $2 done" ;;
  *) echo "usage: $0 kbench | probe <label> <env> | ab <name> <concs> <envA> <envB> | gate <label> <env> [strata]" >&2
     exit 1 ;;
esac
