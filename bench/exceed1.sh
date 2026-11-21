#!/usr/bin/env bash
# Agent exceed-1 (TUNING-LOG "Single-stream" R1): lock in round 1. Runs ON the 5090 host in tmux edge0-exceed1-*,
# holding the GPU lock, with no server running.
#
#   bench/exceed1.sh build   tag the current image edge0-vllm:pre-exceed1, rebuild edge0-vllm:dev from this tree
#                            (nice; prebuilds the cpu-experts and ple-defer extensions), check plugin == tree
#   bench/exceed1.sh gate    the image's PRESET=single (no EDGE0_DEV): compare_strata.py, greedy_same.py gen (DFlash2
#                            drafter), one 23k-token prompt (prefill memory peak); then PRESET=single SPEC_METHOD=none:
#                            greedy_same.py gen; diff
#   bench/exceed1.sh cmp     vLLM PRESET=single twice (bench/gap-vllm.sh: warm 512 + 512 ignore-EOS + EOS pass, the
#                            same 8 ShareGPT seed-0 prompts as Strata's C1 passes), one c8 pass on the second server
# Log: logs/exceed1.log. Labels: r1-gate, r1-v1(-warm|-eos), r1-v2(-warm|-eos|-c8).
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BENCH=${BENCH:-$HOME/work/edge0-bench}
LOG=$REPO/logs/exceed1.log
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
  grep -E "PYTHONPATH|edge0: (placement|after load|latent D|lm_head|shared|PLE rows deferred|EDGE0_PLE_DEFER)|edge0 tier: reactive|edge0 cpu-experts: [0-9]+ workers|cpu-experts.*(extension|prebuilt)|GPU KV cache size|Traceback" \
    "$SLOG" | sed 's/^.*\] //' | cut -c1-300 >>"$LOG"
}

long_prompt() {    # one 23k-token prompt (LONG=6000 words, as G2 / D2): the prefill memory peak of the preset
  python3 -c "
import json, sys, time, urllib.request
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

case "${1:-}" in
  build)
    say "build: tag old image edge0-vllm:pre-exceed1, rebuild edge0-vllm:dev"
    docker image inspect edge0-vllm:pre-exceed1 >/dev/null 2>&1 || docker tag edge0-vllm:dev edge0-vllm:pre-exceed1
    ( cd "$REPO" && nice -n 19 docker build -f docker/Dockerfile -t edge0-vllm:dev . ) >>"$LOG" 2>&1
    say "build exit $?"
    docker run --rm --entrypoint bash edge0-vllm:dev -c '
      d=$(python3 -c "import edge0_vllm, os; print(os.path.dirname(edge0_vllm.__file__))")
      echo "installed plugin: $d"
      python3 -c "from edge0_vllm import cpu_experts as c, ple_defer as p; import os
for m, so in ((c, \"edge0_cpu_experts.so\"), (p, \"edge0_ple.so\")):
    q = os.path.join(\"/opt/edge0/ext\", m.ext_key()); print(m.__name__, \"prebuilt:\", q, os.path.isfile(os.path.join(q, so)))"
      cd /opt/edge0/plugin && find edge0_vllm -name "*.py" -o -name "*.cu" -o -name "*.cpp" -o -name "*.h" | sort | xargs md5sum' \
      >"$REPO/results/exceed1-image.txt" 2>&1
    ( cd "$REPO/plugin" && find edge0_vllm -name "*.py" -o -name "*.cu" -o -name "*.cpp" -o -name "*.h" | xargs md5sum | LC_ALL=C sort -k2 ) \
      >"$REPO/results/exceed1-tree.md5"
    if diff <(grep -E '^[0-9a-f]{32}  edge0_vllm/' "$REPO/results/exceed1-image.txt" | LC_ALL=C sort -k2) "$REPO/results/exceed1-tree.md5" >/dev/null; then
      say "image plugin == tree plugin ($(wc -l <"$REPO/results/exceed1-tree.md5") files)"
    else
      say "IMAGE PLUGIN DIFFERS FROM TREE"
    fi
    head -4 "$REPO/results/exceed1-image.txt" | cut -c1-300 >>"$LOG"
    say "BUILD_DONE" ;;
  gate)
    if start_server "PRESET=single"; then
      ( cd "$REPO/correctness" && python3 -I compare_strata.py --out "$REPO/results/compare-r1-single.json" ) \
        >"$REPO/results/compare-r1-single.txt" 2>&1
      say "CORRECTNESS exit $? $(grep '^SUMMARY' "$REPO/results/compare-r1-single.txt" | cut -c1-300)"
      ( cd "$REPO/correctness" && python3 -I greedy_same.py gen --out "$REPO/results/greedy-r1-single.json" ) >>"$LOG" 2>&1
      say "GREEDY spec exit $?"
      long_prompt
      grep -E "edge0 ple-defer" "$SLOG" | tail -3 | sed 's/^.*\] //' | cut -c1-300 >>"$LOG"
      grep -E "Traceback|Error" "$SLOG" | head -5 >>"$LOG"
    fi
    if start_server "PRESET=single SPEC_METHOD=none"; then
      ( cd "$REPO/correctness" && python3 -I greedy_same.py gen --out "$REPO/results/greedy-r1-single-nospec.json" ) >>"$LOG" 2>&1
      say "GREEDY nospec exit $?"
    fi
    stop_server
    ( cd "$REPO/correctness" && python3 -I greedy_same.py diff "$REPO/results/greedy-r1-single-nospec.json" \
        "$REPO/results/greedy-r1-single.json" ) >"$REPO/results/greedy-r1-single-diff.txt" 2>&1
    say "GREEDY_DIFF exit $? $(tail -1 "$REPO/results/greedy-r1-single-diff.txt")"
    say "GATE_DONE" ;;
  cmp)
    stop_server
    DETAILED=1 bash "$REPO/bench/gap-vllm.sh" r1-v1 "PRESET=single" >>"$LOG" 2>&1; say "V1 done"
    KEEP=1 DETAILED=1 bash "$REPO/bench/gap-vllm.sh" r1-v2 "PRESET=single" >>"$LOG" 2>&1; say "V2 done"
    if curl -sf localhost:8125/health >/dev/null 2>&1; then
      TMP=$(mktemp -d)
      python3 "$REPO/bench/specstat.py" snap >"$TMP/a.json"
      say "bench r1-v2-c8 c8 prompts 16 out 512 ignore_eos"
      mkdir -p "$BENCH/results/r1-v2-c8"
      "$HOME/.local/bin/vllm-bench" --backend openai-chat --base-url http://127.0.0.1:8125 --model edge0_145b \
        --tokenizer "$BENCH/tokenizer-qwen38" \
        --dataset-name sharegpt --dataset-path "$BENCH/datasets/ShareGPT_V3_unfiltered_cleaned_split.json" \
        --sharegpt-output-len 512 --ignore-eos --temperature 0 --seed 0 --save-detailed \
        --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' \
        --max-concurrency 8 --num-prompts 16 --num-warmups 2 \
        --percentile-metrics ttft,tpot,itl,e2el --save-result --result-dir "$BENCH/results/r1-v2-c8" \
        --label r1-v2-c8 --disable-tqdm >>"$LOG" 2>&1 || say "bench r1-v2-c8 failed"
      python3 "$REPO/bench/specstat.py" snap >"$TMP/b.json"
      echo "SPECSTAT r1-v2-c8 $(python3 "$REPO/bench/specstat.py" diff "$TMP/a.json" "$TMP/b.json")" | tee -a "$LOG"
      rm -rf "$TMP"
    else
      say "V2 server not up for the c8 pass"
    fi
    stop_server
    python3 "$REPO/bench/summarize.py" cmp-s1 cmp-s2 r1-v1 r1-v1-eos r1-v2 r1-v2-eos r1-v2-c8 | tee -a "$LOG"
    python3 "$REPO/tools/c1_compare.py" cmp-s1 cmp-s2 r1-v1-eos r1-v2-eos r1-v1 r1-v2 r1-v2-c8 --metrics "$REPO/results/gap" | tee -a "$LOG"
    say "CMP_DONE" ;;
  *) echo "usage: $0 build | gate | cmp" >&2; exit 1 ;;
esac
