#!/usr/bin/env bash
# Agent consolidate (TUNING-LOG "Single-stream" C1): lock in the c1 parity configuration. Runs ON the 5090 host in
# tmux edge0-consolidate-*, holding the GPU lock, with no server running.
#
#   bench/consolidate.sh build   tag the current image edge0-vllm:pre-consolidate, rebuild edge0-vllm:dev from this
#                                tree (nice), check the installed plugin matches the tree
#   bench/consolidate.sh gate    the image's PRESET=single (no EDGE0_DEV): compare_strata.py + greedy_same.py gen
#                                (DFlash2 drafter), then PRESET=single SPEC_METHOD=none: greedy_same.py gen; diff
#   bench/consolidate.sh cmp     fair c1 comparison, interleaved: Strata (~/work/edge0-serve.sh), vLLM PRESET=single,
#                                vLLM PRESET=single (+ one c8 pass), Strata. bench/gap-strata.sh ref / gap-vllm.sh
#                                (same 8 ShareGPT seed-0 prompts, thinking off; EOS-stopping pass for both engines)
# Log: logs/consolidate.log. Labels: cmp-s1, cmp-v1(-warm|-eos), cmp-v2(-warm|-eos|-c8), cmp-s2 (cons-s1 = the
# first Strata pass through edge0-infer's relay, mis-read by vllm-bench; DIRECT=1 benches the engine's own server).
# Every pass is saved with --save-detailed; tools/c1_compare.py prints TTFT to the first content token and decode TPOT.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BENCH=${BENCH:-$HOME/work/edge0-bench}
LOG=$REPO/logs/consolidate.log
mkdir -p "$REPO/logs" "$REPO/results"
say() { echo "# $(date +%H:%M:%S) $*" | tee -a "$LOG"; }

stop_server() {
  docker stop edge0-vllm >/dev/null 2>&1; tmux kill-session -t edge0-vllm 2>/dev/null
  for _ in $(seq 1 60); do docker ps -a --format '{{.Names}}' | grep -q '^edge0-vllm$' || break; sleep 2; done
  sleep 2
}

start_server() {   # $1 = env
  stop_server
  say "start: $1"
  # shellcheck disable=SC2086
  env $1 "$REPO/serve.sh" >>"$LOG" 2>&1 || { say "serve.sh failed"; return 1; }
  SLOG=$(ls -t "$REPO"/logs/serve-*.log | head -1); local t0=$SECONDS
  until curl -sf localhost:8125/health >/dev/null 2>&1; do
    if (( SECONDS - t0 > 60 )) && ! docker ps --format '{{.Names}}' | grep -q '^edge0-vllm$'; then
      say "container exited ($SLOG)"; tail -40 "$SLOG" >>"$LOG"; return 1; fi
    (( SECONDS - t0 > 1500 )) && { say "start timed out"; stop_server; return 1; }
    sleep 5
  done
  say "healthy after $((SECONDS - t0)) s ($SLOG)"
  grep -E "PYTHONPATH|edge0: (placement|after load|latent D|lm_head|shared)|edge0 tier: reactive|edge0 cpu-experts: [0-9]+ workers|cpu-experts.*(extension|prebuilt)|GPU KV cache size|Traceback" \
    "$SLOG" | sed 's/^.*\] //' | cut -c1-300 >>"$LOG"
}

case "${1:-}" in
  build)
    say "build: tag old image, rebuild edge0-vllm:dev"
    docker image inspect edge0-vllm:pre-consolidate >/dev/null 2>&1 || docker tag edge0-vllm:dev edge0-vllm:pre-consolidate
    ( cd "$REPO" && nice -n 19 docker build -f docker/Dockerfile -t edge0-vllm:dev . ) >>"$LOG" 2>&1
    say "build exit $?"
    docker run --rm --entrypoint bash edge0-vllm:dev -c '
      d=$(python3 -c "import edge0_vllm, os; print(os.path.dirname(edge0_vllm.__file__))")
      echo "installed plugin: $d"; ls $d | tr "\n" " "; echo
      grep -c "EDGE0_DU_MASK\|EDGE0_SHARED_FUSE\|EDGE0_LMHEAD\|EDGE0_TIER" $d/config.py $d/*.py | grep -v ":0"
      python3 -c "from edge0_vllm import cpu_experts as c; import os; p=os.path.join(\"/opt/edge0/ext\", c.ext_key()); print(\"cpu-experts prebuilt:\", p, os.listdir(p))"
      cd /opt/edge0/plugin && find edge0_vllm -name "*.py" -o -name "*.cu" -o -name "*.cpp" -o -name "*.h" | sort | xargs md5sum' \
      >"$REPO/results/consolidate-image.txt" 2>&1
    ( cd "$REPO/plugin" && find edge0_vllm -name "*.py" -o -name "*.cu" -o -name "*.cpp" -o -name "*.h" | xargs md5sum | LC_ALL=C sort -k2 ) \
      >"$REPO/results/consolidate-tree.md5"
    if diff <(grep -E '^[0-9a-f]{32}  edge0_vllm/' "$REPO/results/consolidate-image.txt" | LC_ALL=C sort -k2) "$REPO/results/consolidate-tree.md5" >/dev/null; then
      say "image plugin == tree plugin ($(wc -l <"$REPO/results/consolidate-tree.md5") files)"
    else
      say "IMAGE PLUGIN DIFFERS FROM TREE"; diff <(grep -E '^[0-9a-f]{32}  edge0_vllm/' "$REPO/results/consolidate-image.txt" | LC_ALL=C sort -k2) \
        "$REPO/results/consolidate-tree.md5" | head -20 >>"$LOG"
    fi
    head -6 "$REPO/results/consolidate-image.txt" | cut -c1-300 >>"$LOG"
    say "BUILD_DONE" ;;
  gate)
    if start_server "PRESET=single"; then
      ( cd "$REPO/correctness" && python3 -I compare_strata.py --out "$REPO/results/compare-cons-single.json" ) \
        >"$REPO/results/compare-cons-single.txt" 2>&1
      say "CORRECTNESS exit $? $(grep '^SUMMARY' "$REPO/results/compare-cons-single.txt" | cut -c1-300)"
      ( cd "$REPO/correctness" && python3 -I greedy_same.py gen --out "$REPO/results/greedy-cons-single.json" ) >>"$LOG" 2>&1
      say "GREEDY spec exit $?"
      grep -E "Traceback|Error" "$SLOG" | head -5 >>"$LOG"
    fi
    if start_server "PRESET=single SPEC_METHOD=none"; then
      ( cd "$REPO/correctness" && python3 -I greedy_same.py gen --out "$REPO/results/greedy-cons-single-nospec.json" ) >>"$LOG" 2>&1
      say "GREEDY nospec exit $?"
    fi
    stop_server
    ( cd "$REPO/correctness" && python3 -I greedy_same.py diff "$REPO/results/greedy-cons-single-nospec.json" \
        "$REPO/results/greedy-cons-single.json" ) >"$REPO/results/greedy-cons-single-diff.txt" 2>&1
    say "GREEDY_DIFF exit $? $(tail -1 "$REPO/results/greedy-cons-single-diff.txt")"
    say "GATE_DONE" ;;
  cmp)
    stop_server
    DIRECT=1 DETAILED=1 bash "$REPO/bench/gap-strata.sh" ref cmp-s1 >>"$LOG" 2>&1; say "S1 done"
    DETAILED=1 bash "$REPO/bench/gap-vllm.sh" cmp-v1 "PRESET=single" >>"$LOG" 2>&1; say "V1 done"
    KEEP=1 DETAILED=1 bash "$REPO/bench/gap-vllm.sh" cmp-v2 "PRESET=single" >>"$LOG" 2>&1; say "V2 done"
    if curl -sf localhost:8125/health >/dev/null 2>&1; then
      TMP=$(mktemp -d)
      python3 "$REPO/bench/specstat.py" snap >"$TMP/a.json"
      say "bench cmp-v2-c8 c8 prompts 16 out 512 ignore_eos"
      mkdir -p "$BENCH/results/cmp-v2-c8"
      "$HOME/.local/bin/vllm-bench" --backend openai-chat --base-url http://127.0.0.1:8125 --model edge0_145b \
        --tokenizer "$BENCH/tokenizer-qwen38" \
        --dataset-name sharegpt --dataset-path "$BENCH/datasets/ShareGPT_V3_unfiltered_cleaned_split.json" \
        --sharegpt-output-len 512 --ignore-eos --temperature 0 --seed 0 --save-detailed \
        --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' \
        --max-concurrency 8 --num-prompts 16 --num-warmups 2 \
        --percentile-metrics ttft,tpot,itl,e2el --save-result --result-dir "$BENCH/results/cmp-v2-c8" \
        --label cmp-v2-c8 --disable-tqdm >>"$LOG" 2>&1 || say "bench cmp-v2-c8 failed"
      python3 "$REPO/bench/specstat.py" snap >"$TMP/b.json"
      echo "SPECSTAT cmp-v2-c8 $(python3 "$REPO/bench/specstat.py" diff "$TMP/a.json" "$TMP/b.json")" | tee -a "$LOG"
      python3 "$REPO/bench/summarize.py" cmp-v2-c8 | tail -1 | tee -a "$LOG"
      rm -rf "$TMP"
    else
      say "V2 server not up for the c8 pass"
    fi
    stop_server
    DIRECT=1 DETAILED=1 bash "$REPO/bench/gap-strata.sh" ref cmp-s2 >>"$LOG" 2>&1; say "S2 done"
    python3 "$REPO/bench/summarize.py" cmp-s1 cmp-v1 cmp-v1-eos cmp-v2 cmp-v2-eos cmp-s2 cmp-v2-c8 | tee -a "$LOG"
    python3 "$REPO/tools/c1_compare.py" cmp-s1 cmp-s2 cmp-v1-eos cmp-v2-eos cmp-v1 cmp-v2 cmp-v2-c8 --metrics "$REPO/results/gap" | tee -a "$LOG"
    say "CMP_DONE" ;;
  *) echo "usage: $0 build | gate | cmp" >&2; exit 1 ;;
esac
