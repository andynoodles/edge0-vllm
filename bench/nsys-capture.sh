#!/usr/bin/env bash
# Nsight Systems captures of steady-state decode under vllm-bench load (docs/PROFILE.md). Runs ON the 5090 host,
# in tmux, while holding the GPU lock.
#
#   bench/nsys-capture.sh <name> "<serve env>" <conc:iters:secs> [...]
#   e.g. bench/nsys-capture.sh nospec "EDGE0_DEV=1 EDGE0_NVTX=1" 1:150:8 8:60:12 32:30:20
#   (secs = time to wait after /start_profile before /stop_profile; must cover <iters> engine steps)
#
# Starts ./serve.sh (with <serve env>) under 'nsys profile' (NSYS_OPTS, default: cuda,nvtx,osrt trace, CUDA graphs
# traced per node, capture range = cudaProfilerApi, repeat) and vLLM's CUDA profiler hooks
# (--profiler-config {"profiler":"cuda"}: /start_profile -> cudaProfilerStart in the engine, each execute_model
# wrapped in an NVTX range "execute_context_N(T)_generation_M(T)"). Per <conc:iters>: start vllm-bench (ShareGPT,
# chat API, thinking off, OUT_LEN=512 tokens, ignore EOS, prompts = 2 x conc), wait until conc requests run and none
# wait, then SETTLE_S more seconds (prefills done), POST /start_profile {"max_iterations": iters}, wait for the
# capture to end, sample GPU metrics on the host for GPUM_S seconds (PCIe RX/TX, SM activity; sudo nsys,
# GPUM=0 to skip), then stop the bench. Reports: profiles/<name>/rep.<k>.nsys-rep (k = capture order), host GPU
# metrics: profiles/<name>/gpum-c<conc>.nsys-rep; log: logs/nsys-<name>.log. The server is stopped at the end
# (KEEP=1 keeps it).
set -uo pipefail
NAME=$1; ENVS=$2; shift 2
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=$REPO/profiles/$NAME
LOG=$REPO/logs/nsys-$NAME.log
SERVE_ARGS=${SERVE_ARGS:-}; OUT_LEN=${OUT_LEN:-512}; SETTLE_S=${SETTLE_S:-4}; WAIT_S=${WAIT_S:-1200}; GPUM=${GPUM:-1}; GPUM_S=${GPUM_S:-4}
NSYS_HOST=${NSYS_HOST:-/usr/local/bin/nsys}
NSYS_IMPORT=${NSYS_IMPORT:-/opt/nvidia/nsight-systems/2025.3.2/host-linux-x64/QdstrmImporter}
NSYS_OPTS=${NSYS_OPTS:-"--trace=cuda,nvtx,osrt --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=repeat --trace-fork-before-exec=true --sample=none --cpuctxsw=none --force-overwrite=true -o /workspace/edge0/profiles/$NAME/rep"}
mkdir -p "$OUT" "$REPO/logs"
say() { echo "# $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
metric() { curl -s localhost:8125/metrics | awk -v m="$1" '$1 ~ "^"m"[{ ]" || $1 == m {s += $2} END {print s + 0}'; }

stop_server() {
  docker stop -t 60 edge0-vllm >/dev/null 2>&1
  tmux kill-session -t edge0-vllm 2>/dev/null
  for _ in $(seq 1 60); do docker ps -a --format '{{.Names}}' | grep -q '^edge0-vllm$' || break; sleep 2; done
  sleep 2
}

if [ "${REUSE:-0}" = 1 ] && curl -sf localhost:8125/health >/dev/null 2>&1; then
  say "reusing the running server (REUSE=1)"     # must have been started by this script (nsys + profiler config)
else
  stop_server
  say "start: $ENVS NSYS=$NSYS_OPTS"
  # shellcheck disable=SC2086
  env $ENVS NSYS="$NSYS_OPTS" "$REPO/serve.sh" --profiler-config '{"profiler":"cuda"}' $SERVE_ARGS >>"$LOG" 2>&1 \
    || { say "START FAILED"; exit 1; }
fi
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

k=0
for spec in "$@"; do
  IFS=: read -r conc iters minwait <<<"$spec"; minwait=${minwait:-10}
  k=$((k + 1))
  label="nsys-$NAME-c$conc"
  say "bench $label conc $conc out $OUT_LEN prompts $((2 * conc)), capture $iters iterations"
  OUT_LEN=$OUT_LEN FACTOR=2 BASE_URL=http://127.0.0.1:8125 \
    EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}' \
    setsid bash "$REPO/bench/bench-openai.sh" "$label" "$conc" >>"$LOG" 2>&1 &
  bpid=$!
  t1=$SECONDS
  while :; do
    run=$(metric 'vllm:num_requests_running'); wq=$(metric 'vllm:num_requests_waiting')
    [ "${run%.*}" -ge "$conc" ] && [ "${wq%.*}" -eq 0 ] && break
    if (( SECONDS - t1 > 600 )); then say "no steady state after 600 s (running $run waiting $wq)"; break; fi
    sleep 0.5
  done
  sleep "$SETTLE_S"
  # a session that ended by max_iterations is still "running" for the API server (409 on the next start)
  curl -s -X POST localhost:8125/stop_profile >/dev/null 2>&1; sleep 1
  gen0=$(metric 'vllm:generation_tokens_total'); ts0=$(date +%s.%N)
  nstop0=$(grep -c "Profiler stopped successfully" "$SLOG")
  say "steady: running $(metric 'vllm:num_requests_running') waiting $(metric 'vllm:num_requests_waiting'); start_profile"
  curl -s -X POST localhost:8125/start_profile -H 'Content-Type: application/json' \
    -d "{\"max_iterations\": $iters}" >>"$LOG" 2>&1
  # The engine stops the CUDA profiler after <iters> iterations (its log line is not printed every time, so wait
  # MINWAIT seconds = the 3rd field of the spec, then /stop_profile to be sure). nsys (2025.3, repeat mode) leaves each
  # range's raw stream as /tmp/nsys-report-*.qdstrm in the container and converts only at session end: copy the
  # newest one out once its size is stable and import it on the host (QdstrmImporter) in the background.
  sleep "$minwait"
  curl -s -X POST localhost:8125/stop_profile >/dev/null 2>&1
  sleep 3
  q=""; prev=-1
  for _ in $(seq 1 120); do
    q=$(docker exec edge0-vllm bash -c 'ls -t /tmp/nsys-report-*.qdstrm 2>/dev/null | head -1')
    [ -n "$q" ] || { sleep 1; continue; }
    sz=$(docker exec edge0-vllm stat -c %s "$q")
    [ "$sz" = "$prev" ] && break
    prev=$sz; sleep 3
  done
  dst="$OUT/c$conc-k$k"
  if [ -n "$q" ] && docker cp "edge0-vllm:$q" "$dst.qdstrm" >/dev/null 2>&1; then
    docker exec edge0-vllm rm -f "$q"
    ( "$NSYS_IMPORT" -i "$dst.qdstrm" >/dev/null 2>&1 && rm -f "$dst.qdstrm" \
      && "$NSYS_HOST" export --type sqlite -o "$dst.sqlite" "$dst.nsys-rep" >/dev/null 2>&1 ) &
    say "capture $k: $q ($prev B) -> $dst.qdstrm (importing)"
  else
    say "capture $k: no qdstrm found"
  fi
  gen1=$(metric 'vllm:generation_tokens_total'); ts1=$(date +%s.%N)
  say "capture $k done: $(ls "$OUT" | tr '\n' ' ')"
  say "tokens during capture window+write: $(awk -v a="$gen0" -v b="$gen1" -v s="$ts0" -v e="$ts1" 'BEGIN{printf "%d tokens in %.1f s = %.1f tok/s", b-a, e-s, (b-a)/(e-s)}')"
  curl -s -X POST localhost:8125/stop_profile >/dev/null 2>&1
  if [ "$GPUM" = 1 ]; then
    gen0=$(metric 'vllm:generation_tokens_total'); ts0=$(date +%s.%N)
    sudo -n "$NSYS_HOST" profile --gpu-metrics-devices=0 --gpu-metrics-set=gb20x --gpu-metrics-frequency=10000 \
      --trace=none --sample=none --cpuctxsw=none --force-overwrite=true -o "$OUT/gpum-c$conc" -d "$GPUM_S" \
      sleep "$GPUM_S" >>"$LOG" 2>&1 || say "gpu metrics failed"
    gen1=$(metric 'vllm:generation_tokens_total'); ts1=$(date +%s.%N)
    say "gpu metrics c$conc: $(awk -v a="$gen0" -v b="$gen1" -v s="$ts0" -v e="$ts1" 'BEGIN{printf "%d tokens in %.1f s = %.1f tok/s (running, no capture)", b-a, e-s, (b-a)/(e-s)}')"
  fi
  kill -- -"$bpid" 2>/dev/null     # setsid: the bench and its children form one process group
  wait "$bpid" 2>/dev/null
  # let the engine drain the aborted requests
  for _ in $(seq 1 60); do [ "$(metric 'vllm:num_requests_running')" = 0 ] && break; sleep 1; done
done
[ "${KEEP:-0}" = 1 ] || stop_server
say "waiting for the host-side imports"
wait
sudo -n chown -R "$(id -u):$(id -g)" "$OUT" 2>/dev/null
say "NSYS_DONE $(ls "$OUT" | tr '\n' ' ')"
