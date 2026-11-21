#!/usr/bin/env bash
# edge0/convert/run_convert.sh - run the converter / checker / unit tests inside the vLLM base image, CPU only.
#
#   run_convert.sh convert --out /models/Edge0-145B-vllm-L4 --num-layers 4 --no-mtp     # args go to convert_checkpoint.py
#   run_convert.sh check   --ckpt /models/Edge0-145B-vllm --src /models/Edge0-145B-NVFP4 --level full
#   run_convert.sh test                                                                  # test_convert.py
#
# Paths are container paths: /models = $MODELS_ROOT (default ~/models), /workspace/edge0 = this repository,
# /tokref = $TOKREF, the Qwen3.8 HF tokenizer (read-only), /bench = $BENCH (read-only: corpus, strata-ref.json;
# default ~/work/edge0-bench).
# `convert` always gets --src /models/Edge0-145B-NVFP4 --hf-tokenizer /tokref unless you pass them.
#
# Environment knobs:
#   CPUS=2 MEM=3g        docker --cpus / --memory (+ --memory-swap = MEM). The memory cgroup also caps this job's page
#                        cache, which protects other residents of the box. After GPU_FREE: CPUS=12 MEM=24g.
#   LOWPRIO=1            run under `nice -n 19 ionice -c3` inside the container (default 1)
#   TMUX_SESSION=name    run detached in that tmux session, log to ~/work/edge0-vllm/logs/<name>-<time>.log;
#                        the log ends with a line EXIT=<code>
#   IMAGE=nvcr.io/nvidia/vllm:26.11-py3
set -euo pipefail
MODE=${1:?usage: run_convert.sh convert|check|test [args...]}; shift
IMAGE=${IMAGE:-nvcr.io/nvidia/vllm:26.11-py3}
CPUS=${CPUS:-2}
MEM=${MEM:-3g}
LOWPRIO=${LOWPRIO:-1}
WORK=${WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MODELS_ROOT=${MODELS_ROOT:-$HOME/models}
CACHE_DIR=${CACHE_DIR:-$HOME/.cache/edge0-vllm}
BENCH=${BENCH:-$HOME/work/edge0-bench}
TOKREF=${TOKREF:-$BENCH/tokenizer-qwen38}
mkdir -p "$WORK/logs" "$CACHE_DIR"

case "$MODE" in
  convert)
    ARGS=("$@")
    [[ " $* " == *" --src "* ]] || ARGS=(--src /models/Edge0-145B-NVFP4 "${ARGS[@]}")
    [[ " $* " == *" --hf-tokenizer "* ]] || ARGS+=(--hf-tokenizer /tokref)
    CMD=(python3 -u /workspace/edge0/convert/convert_checkpoint.py "${ARGS[@]}") ;;
  check)
    ARGS=("$@")
    [[ " $* " == *" --hf-ref "* ]] || ARGS+=(--hf-ref /tokref)
    CMD=(python3 -u /workspace/edge0/convert/check_ckpt.py "${ARGS[@]}") ;;
  test)
    CMD=(python3 -u /workspace/edge0/convert/test_convert.py "$@") ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
PRE=()
[[ "$LOWPRIO" == 1 ]] && PRE=(nice -n 19 ionice -c3)

DOCKER=(docker run --rm --name "edge0-convert-$MODE-$$" --network none --cpus "$CPUS" --memory "$MEM" --memory-swap "$MEM"
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  -e OMP_NUM_THREADS="${CPUS%.*}" -e OPENBLAS_NUM_THREADS="${CPUS%.*}" -e MKL_NUM_THREADS="${CPUS%.*}"
  --ulimit memlock=-1 --ulimit stack=67108864
  -v "$MODELS_ROOT":/models -v "$WORK":/workspace/edge0 -v "$CACHE_DIR":/root/.cache
  -v "$TOKREF":/tokref:ro -v "$BENCH":/bench:ro
  --entrypoint "${PRE[0]:-python3}" "$IMAGE")
if [[ ${#PRE[@]} -gt 0 ]]; then FULL=("${DOCKER[@]}" "${PRE[@]:1}" "${CMD[@]}"); else FULL=("${DOCKER[@]}" "${CMD[@]:1}"); fi

if [[ -n "${TMUX_SESSION:-}" ]]; then
  STAMP=$(date +%Y%m%d-%H%M%S)
  LOG="$WORK/logs/$TMUX_SESSION-$STAMP.log"
  RUN="$WORK/logs/$TMUX_SESSION-$STAMP.sh"
  { echo '#!/usr/bin/env bash'; printf '%q ' "${FULL[@]}"; echo "> $(printf %q "$LOG") 2>&1"
    echo "echo EXIT=\$? >> $(printf %q "$LOG")"; } > "$RUN"
  chmod +x "$RUN"
  tmux new-session -d -s "$TMUX_SESSION" "$RUN"
  echo "started tmux session $TMUX_SESSION, log $LOG"
else
  exec "${FULL[@]}"
fi
