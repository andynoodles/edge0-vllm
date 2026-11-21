#!/usr/bin/env bash
# Agent exceed-2 (TUNING-LOG "Single-stream" R2). Runs ON the 5090 host in tmux edge0-exceed2-*, holding the GPU lock,
# with no server running.
#
#   bench/exceed2.sh prof    profile of the current PRESET=single at c1 after an 8-prompt warm pass: node-level nsys
#                            (EDGE0_NVTX=1 EDGE0_TRACE=bw; kernel split, cold counts) -> profiles/r2single{,w}/, then
#                            graph-level nsys (EDGE0_NVTX=1; true step period, idle) -> profiles/g2single{,w}/
#   bench/exceed2.sh probe <label> "<env>" [...]   one warm + one scored c1 pass per server (bench/fc-ab.sh, letter A)
#   bench/exceed2.sh vram <label> "<env>"          c1 + c8 passes, then a 23k-token prompt, GPU memory sampled
#   bench/exceed2.sh build | gate [env] | cmp [env]   rebuild the image (old one kept as edge0-vllm:pre-exceed2); the
#                            correctness gate of the image's PRESET=single; the fair c1 comparison (vLLM twice + c8)
# Log: logs/exceed2.log.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG=$REPO/logs/exceed2.log
mkdir -p "$REPO/logs" "$REPO/results"
say() { echo "# $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
stop_server() {
  docker stop edge0-vllm >/dev/null 2>&1; tmux kill-session -t edge0-vllm 2>/dev/null
  for _ in $(seq 1 60); do docker ps -a --format '{{.Names}}' | grep -q '^edge0-vllm$' || break; sleep 2; done
  sleep 2
}

case "${1:-}" in
  prof)
    stop_server
    say "prof: node-level r2single"
    bash "$REPO/bench/gap-nsys.sh" r2single "PRESET=single EDGE0_NVTX=1 EDGE0_TRACE=bw" >>"$LOG" 2>&1
    SLOG=$(ls -t "$REPO"/logs/serve-*.log | head -1)
    grep -E "edge0 tier:|edge0 trace bw|edge0 cpu-experts: [0-9]+ jobs|edge0 ple-defer" "$SLOG" | sed 's/^.*\] //' \
      | cut -c1-420 | tail -30 >>"$LOG"
    stop_server
    say "prof: graph-level g2single"
    NSYS_OPTS="--trace=cuda,nvtx,osrt --cuda-graph-trace=graph --capture-range=cudaProfilerApi --capture-range-end=repeat --trace-fork-before-exec=true --sample=none --cpuctxsw=none --force-overwrite=true -o /workspace/edge0/profiles/g2single/rep" \
      bash "$REPO/bench/gap-nsys.sh" g2single "PRESET=single EDGE0_NVTX=1" >>"$LOG" 2>&1
    SLOG=$(ls -t "$REPO"/logs/serve-*.log | head -1)
    grep -E "edge0 tier:|edge0 cpu-experts: [0-9]+ jobs|edge0 ple-defer" "$SLOG" | sed 's/^.*\] //' \
      | cut -c1-420 | tail -30 >>"$LOG"
    stop_server
    say "PROF_DONE" ;;
  probe)   # bench/exceed2.sh probe <label> "<env>" [<label> "<env>" ...]: one warm + one scored c1 pass each
    shift
    while [ $# -ge 2 ]; do
      L=$1; E=$2; shift 2
      say "probe $L: $E"
      WARM=8 PROMPTS=8 C8=${C8:-0} bash "$REPO/bench/fc-ab.sh" "$L" "$E" "$E" A >/dev/null 2>&1
      grep -E "^\| |tok/s|SPECSTAT|^  \[" "$REPO/logs/fcab-$L.log" | grep -v -- "-warm" | cut -c1-420 | tail -14 >>"$LOG"
      say "probe $L done"
    done
    say "PROBES_DONE" ;;
  vram)    # bench/exceed2.sh vram <label> "<env>": warm + scored c1 + c8 pass, then one 23k-token prompt, with the
           # GPU's memory.used sampled every 0.2 s (peak per phase); the server is stopped at the end
    L=$2; E=$3
    say "vram probe $L: $E"
    SMP=$REPO/logs/exceed2-mem-$L.txt
    ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.2; done ) >"$SMP" 2>/dev/null &
    spid=$!
    WARM=8 PROMPTS=8 C8=1 KEEP=1 bash "$REPO/bench/fc-ab.sh" "$L" "$E" "$E" A >/dev/null 2>&1
    grep -E "^\| |tok/s|SPECSTAT|^  \[" "$REPO/logs/fcab-$L.log" | grep -v -- "-warm" | cut -c1-420 | tail -16 >>"$LOG"
    python3 "$REPO/bench/summarize.py" "$L-A1" "$L-A1-c8" >>"$LOG"
    say "peak memory.used after the c1 / c8 passes: $(sort -n "$SMP" | tail -1) MiB"
    n0=$(wc -l <"$SMP")
    if curl -sf localhost:8125/health >/dev/null 2>&1; then
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
      say "peak memory.used during the long prompt: $(tail -n +"$n0" "$SMP" | sort -n | tail -1) MiB"
    else
      say "server not up after the passes"
    fi
    SLOG=$(ls -t "$REPO"/logs/serve-*.log | head -1)
    grep -E "edge0: after load|GPU KV cache size|Graph capturing finished|OutOfMemory|out of memory|Traceback" "$SLOG" \
      | sed 's/^.*\] //' | cut -c1-260 | head -8 >>"$LOG"
    kill $spid 2>/dev/null
    stop_server
    say "VRAM_DONE $L" ;;
  build)   # tag the current image edge0-vllm:pre-exceed2, rebuild edge0-vllm:dev from this tree, check plugin == tree
    say "build: tag old image edge0-vllm:pre-exceed2, rebuild edge0-vllm:dev"
    docker image inspect edge0-vllm:pre-exceed2 >/dev/null 2>&1 || docker tag edge0-vllm:dev edge0-vllm:pre-exceed2
    ( cd "$REPO" && nice -n 19 docker build -f docker/Dockerfile -t edge0-vllm:dev . ) >>"$LOG" 2>&1
    say "build exit $?"
    docker run --rm --entrypoint bash edge0-vllm:dev -c '
      python3 -c "from edge0_vllm import cpu_experts as c, ple_defer as p; import os
for m, so in ((c, \"edge0_cpu_experts.so\"), (p, \"edge0_ple.so\")):
    q = os.path.join(\"/opt/edge0/ext\", m.ext_key()); print(m.__name__, \"prebuilt:\", q, os.path.isfile(os.path.join(q, so)))"
      cd /opt/edge0/plugin && find edge0_vllm -name "*.py" -o -name "*.cu" -o -name "*.cpp" -o -name "*.h" | sort | xargs md5sum' \
      >"$REPO/results/exceed2-image.txt" 2>&1
    ( cd "$REPO/plugin" && find edge0_vllm -name "*.py" -o -name "*.cu" -o -name "*.cpp" -o -name "*.h" | xargs md5sum | LC_ALL=C sort -k2 ) \
      >"$REPO/results/exceed2-tree.md5"
    if diff <(grep -E '^[0-9a-f]{32}  edge0_vllm/' "$REPO/results/exceed2-image.txt" | LC_ALL=C sort -k2) "$REPO/results/exceed2-tree.md5" >/dev/null; then
      say "image plugin == tree plugin ($(wc -l <"$REPO/results/exceed2-tree.md5") files)"
    else
      say "IMAGE PLUGIN DIFFERS FROM TREE"
    fi
    grep -E "prebuilt" "$REPO/results/exceed2-image.txt" | cut -c1-200 >>"$LOG"
    say "image $(docker image inspect -f '{{.Id}}' edge0-vllm:dev | cut -c8-19)"
    say "BUILD_DONE" ;;
  gate)    # the image's PRESET=single (no EDGE0_DEV): compare_strata.py, greedy_same.py gen, one 23k-token prompt; then
           # PRESET=single SPEC_METHOD=none: greedy_same.py gen; diff. Extra env for both servers: $2
    X=${2:-}
    for spec in "PRESET=single $X|r2-single" "PRESET=single SPEC_METHOD=none $X|r2-single-nospec"; do
      E=${spec%%|*}; G=${spec##*|}
      stop_server
      say "gate start: $E"
      # shellcheck disable=SC2086
      env $E "$REPO/serve.sh" >>"$LOG" 2>&1 || { say "serve.sh failed"; continue; }
      SLOG=$(ls -t "$REPO"/logs/serve-*.log | head -1); t0=$SECONDS; ok=1
      until curl -sf localhost:8125/health >/dev/null 2>&1; do
        if (( SECONDS - t0 > 60 )) && ! docker ps --format '{{.Names}}' | grep -q '^edge0-vllm$'; then
          say "container exited ($SLOG)"; tail -30 "$SLOG" >>"$LOG"; ok=0; break; fi
        sleep 5
      done
      [ $ok = 1 ] || continue
      say "healthy after $((SECONDS - t0)) s ($SLOG)"
      grep -E "PYTHONPATH|edge0: (placement|after load)|edge0 cpu-experts: [0-9]+ workers|GPU KV cache size|Traceback" \
        "$SLOG" | sed 's/^.*\] //' | cut -c1-260 >>"$LOG"
      if [ "$G" = r2-single ]; then
        ( cd "$REPO/correctness" && python3 -I compare_strata.py --out "$REPO/results/compare-$G.json" ) \
          >"$REPO/results/compare-$G.txt" 2>&1
        say "CORRECTNESS exit $? $(grep '^SUMMARY' "$REPO/results/compare-$G.txt" | cut -c1-300)"
      fi
      ( cd "$REPO/correctness" && python3 -I greedy_same.py gen --out "$REPO/results/greedy-$G.json" ) >>"$LOG" 2>&1
      say "GREEDY $G exit $?"
      if [ "$G" = r2-single ]; then
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
      fi
      grep -E "Traceback|timeouts [1-9]" "$SLOG" | head -5 >>"$LOG"
    done
    stop_server
    ( cd "$REPO/correctness" && python3 -I greedy_same.py diff "$REPO/results/greedy-r2-single-nospec.json" \
        "$REPO/results/greedy-r2-single.json" ) >"$REPO/results/greedy-r2-single-diff.txt" 2>&1
    say "GREEDY_DIFF exit $? $(tail -1 "$REPO/results/greedy-r2-single-diff.txt")"
    say "GATE_DONE" ;;
  cmp)     # vLLM PRESET=single twice (bench/gap-vllm.sh: warm 512 + 512 ignore-EOS + EOS pass, the same 8 ShareGPT
           # seed-0 prompts as Strata's C1 passes), one c8 pass on the second server. Extra env: $2
    X=${2:-}
    stop_server
    DETAILED=1 bash "$REPO/bench/gap-vllm.sh" r2-v1 "PRESET=single $X" >>"$LOG" 2>&1; say "V1 done"
    KEEP=1 DETAILED=1 bash "$REPO/bench/gap-vllm.sh" r2-v2 "PRESET=single $X" >>"$LOG" 2>&1; say "V2 done"
    if curl -sf localhost:8125/health >/dev/null 2>&1; then
      BENCH=${BENCH:-$HOME/work/edge0-bench}
      TMP=$(mktemp -d)
      python3 "$REPO/bench/specstat.py" snap >"$TMP/a.json"
      say "bench r2-v2-c8 c8 prompts 16 out 512 ignore_eos"
      mkdir -p "$BENCH/results/r2-v2-c8"
      "$HOME/.local/bin/vllm-bench" --backend openai-chat --base-url http://127.0.0.1:8125 --model edge0_145b \
        --tokenizer "$BENCH/tokenizer-qwen38" \
        --dataset-name sharegpt --dataset-path "$BENCH/datasets/ShareGPT_V3_unfiltered_cleaned_split.json" \
        --sharegpt-output-len 512 --ignore-eos --temperature 0 --seed 0 --save-detailed \
        --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' \
        --max-concurrency 8 --num-prompts 16 --num-warmups 2 \
        --percentile-metrics ttft,tpot,itl,e2el --save-result --result-dir "$BENCH/results/r2-v2-c8" \
        --label r2-v2-c8 --disable-tqdm >>"$LOG" 2>&1 || say "bench r2-v2-c8 failed"
      python3 "$REPO/bench/specstat.py" snap >"$TMP/b.json"
      echo "SPECSTAT r2-v2-c8 $(python3 "$REPO/bench/specstat.py" diff "$TMP/a.json" "$TMP/b.json")" | tee -a "$LOG"
      rm -rf "$TMP"
    else
      say "V2 server not up for the c8 pass"
    fi
    stop_server
    python3 "$REPO/bench/summarize.py" cmp-s1 cmp-s2 r2-v1 r2-v1-eos r2-v2 r2-v2-eos r2-v2-c8 | tee -a "$LOG"
    python3 "$REPO/tools/c1_compare.py" cmp-s1 cmp-s2 r2-v1-eos r2-v2-eos r2-v1 r2-v2 r2-v2-c8 --metrics "$REPO/results/gap" | tee -a "$LOG"
    say "CMP_DONE" ;;
  *) echo "usage: $0 prof | probe <label> <env> ... | vram <label> <env> | build | gate [env] | cmp [env]" >&2; exit 1 ;;
esac
