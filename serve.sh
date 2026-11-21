#!/usr/bin/env bash
# Edge0-145B-NVFP4 on vLLM (edge0-vllm plugin), port 8125. Runs on the GPU host (tuned on one RTX 5090 32 GB).
#
#   MODEL_DIR=/nvme/models/Edge0-145B-NVFP4-vLLM ./serve.sh [extra 'vllm serve' args...]
#                                             start container $NAME in tmux session $TMUX_SESSION (log: logs/serve-*.log)
#   DRY_RUN=1 ./serve.sh                      print the docker command instead of running it (works anytime)
#   curl -s localhost:8125/health             ready when this returns 200
#   docker stop edge0-vllm; tmux kill-session -t edge0-vllm    stop
#
# Paths (host side; every one can be overridden):
#   MODEL_DIR=$MODELS_ROOT/Edge0-145B-NVFP4-vLLM   the checkpoint (hf download andynoodles/Edge0-145B-NVFP4-vLLM);
#                       falls back to $MODELS_ROOT/Edge0-145B-vllm (the converter's default --out) when only that exists
#   MODELS_ROOT=$HOME/models   mounted at /models when MODEL_DIR / DFLASH_DIR are under it; a dir outside it is mounted
#                       read-only on its own at /mnt/edge0/<name>
#   DFLASH_DIR=$MODELS_ROOT/Qwen3.8-Flash-Next-DFlash2   drafter for SPEC_METHOD=dflash2 (PRESET=single)
#                       (hf download incoai/Qwen3.8-Flash-Next-DFlash2)
#   PLE_DIR=            extra dir mounted at its own path; default: the directory an absolute symlink
#                       MODEL_DIR/ple-fp8.gguf points to (the 51 GB PLE table moved to another NVMe)
#   WORK_DIR=<this repo> (mounted at /workspace/edge0)   CACHE_DIR=$HOME/.cache/edge0-vllm (mounted at /root/.cache)
#   GPU_FREE=           if set, refuse to start until this file exists (a lock file when another engine shares the GPU)
#
# Switches (defaults = PRESET=tuned, see below; the values listed here are the bring-up ones, DESIGN.md 2.10 / 3.1):
#   EAGER=0             1 = --enforce-eager (first runs / debugging; also required by EDGE0_MOE_IMPL=torch)
#   MAX_MODEL_LEN=32768 MAX_SEQS=16 MAX_BATCHED=4096 GPU_UTIL=0.96
#   KV_BYTES=3500000000 --kv-cache-memory-bytes (GDN/PLE state + QSA KV); empty = size the KV cache from GPU_UTIL
#   KV_DTYPE=auto       auto (bf16) | fp8_e4m3 (lever L1)
#   SSM_DTYPE=float32   --mamba-ssm-cache-dtype (bfloat16 = lever L1, check KL first)
#   SPEC=               --speculative-config JSON, raw (wins over the SPEC_* builder below)
#   SPEC_METHOD=        speculative decoding (TUNING-LOG.md section 8): none | mtp (Edge0's own MTP layer, from the
#                       checkpoint's mtp.* shard) | dflash2 (/models/Qwen3.8-Flash-Next-DFlash2, block drafter fed by the
#                       completed outputs of layers 5,15,25,35,45; Edge0ForCausalLM exposes them)
#   SPEC_K=3            num_speculative_tokens = the maximum draft length (it also sizes the per-request GDN state:
#                       1 + K state pages per request, ~90 MiB per K per request, so K costs KV memory at every batch
#                       size). Best measured: SPEC_METHOD=dflash2 SPEC_K=3 (+ EDGE0_CPU_EXPERTS=1): +65% at c1, +6%
#                       at c8, -19% at c32 (only 6 requests fit), so for <= ~6 concurrent requests only; k=5/7/15 are
#                       slower (TUNING-LOG.md "Draft length sweep")
#   SPEC_SCHED=         num_speculative_tokens_per_batch_size JSON, draft length by running requests, e.g.
#                       [[1,4,3],[5,16,2],[17,32,1]] (no spaces)
#   SPEC_CONF=          draft_confidence_threshold (mtp): stop drafting below this top-1 probability
#   SPEC_VOCAB=         draft_token_map (mtp): container path of a JSON id list, e.g. <container model dir>/draft_vocab.json
#   SPEC_EXTRA=         extra JSON members appended verbatim, e.g. '"draft_lm_head_quantization":"fp8"'
#   SPEC_MAX_BATCHED=2048 SPEC_GPU_UTIL=0.94   with speculative decoding on, the defaults of MAX_BATCHED / GPU_UTIL
#                       (an explicit MAX_BATCHED / GPU_UTIL still wins). vLLM's memory profile runs with skip_attn, so
#                       the GDN/QSA prefill working set of a full MAX_BATCHED chunk (and the drafter's) is not in the
#                       KV sizing; with a drafter the real peak ran 1.8 GiB past GPU_UTIL's budget and OOMed at
#                       c32 (TUNING-LOG.md "Draft length sweep", S1). 2048 halves that overshoot and frees ~0.9 GiB of KV.
#   PREFIX_CACHING=1    0 = --no-enable-prefix-caching: GDN state then uses mamba_cache_mode 'none' (1 state page per
#                       request instead of 'align''s 2), i.e. more concurrent requests per GiB of KV (TUNING-LOG.md)
#   VLLM_OFFLOAD_GB=0   >0: vLLM's own UVA weight offload of dense params (--offload-backend uva --cpu-offload-gb N),
#                       the stock fallback for VRAM pressure; the routed experts always use the plugin's host pool
#   EDGE0_DEV=0         1 = bind-mounted plugin source shadows the installed copy (PYTHONPATH), no image rebuild
#   IMAGE=edge0-vllm:dev NAME=edge0-vllm TMUX_SESSION=edge0-vllm PORT=8125 SERVED_NAME=edge0_145b
#   TORCH_DIST_USE_NCCL2=0  torch 2.15's default "nccl2" process-group backend reserves ~3.4 GiB of VRAM at distributed
#                       init even for world_size 1 (vLLM only applies this workaround on integrated GPUs); 0 = stock
#                       ProcessGroupNCCL, which leaves that VRAM for hot experts (BRINGUP-LOG.md)
#   VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128  QSA indexer prefill logits workspace, allocated per prefill call at its
#                       worst case (vLLM default 512 MiB; a fragmented 512 MiB request OOMed at concurrency 16).
#                       128 MiB still covers 4096 rows x 8192 compressed columns (= MAX_BATCHED 4096 at MAX_MODEL_LEN
#                       32768) in one chunk; raise it with longer contexts
# Plugin knobs, passed through to the container (edge0/plugin/edge0_vllm/config.py, DESIGN.md 2.9):
#   EDGE0_EXPERT_VRAM_GIB=9.5  VRAM for hot expert slots (H = GiB*2^30/1,935,360); 0 = fallback F0 (all experts in host
#                              memory: 44.3 GiB pinned -> also set EDGE0_EMBED_HOST=0)
#   EDGE0_PLACEMENT=profile    profile (top-H of profile.bin) | identity | plan:/workspace/edge0/<plan.json>
#   EDGE0_DENSE=bf16           bf16 (dequantize Q8_0 at load) | q8 (b12x native, phase 2)
#   EDGE0_COLD_ALLOC=host_register  host_register | host_alloc | torch_pinned (vLLM's UVA-offload allocator)
#   EDGE0_SPLIT=1              hot and cold Marlin calls on two streams (0 = sequential, A/B)
#   EDGE0_MOE_IMPL=marlin      marlin | torch (F1 reference for the first EDGE0_REF_LAYERS layers; needs EAGER=1)
#   EDGE0_EMBED_HOST=1         embed_tokens in pinned host memory
#   EDGE0_TRACE=0              bw (hot/cold distinct experts + cold bytes per step, every EDGE0_TRACE_FLUSH_S s) | routes
#                              (per-token routing + call boundaries -> tools/plan_placement.py) | predict (next-layer
#                              pre-gating accuracy); comma-separated
#   EDGE0_ADAPT=0              1 = adaptive hot/cold swaps (TidalMoE z-test; EDGE0_ADAPT_EPOCH_S/SWAPS/HALFLIFE/MARGIN/
#                              Z/MIN_STEPS/MAX_TOKENS, EDGE0_ADAPT_SAVE=<plan path>); plugin/edge0_vllm/adaptive.py
#   EDGE0_TIER=0               1 = request-reactive expert tier (Strata's policy, decided per step without a host sync,
#                              copies on the copy engines; supersedes EDGE0_ADAPT): EDGE0_TIER_FREE=4 (free VRAM slots
#                              per layer, taken from the hot budget) _SWAPS=48 (per step) _DECAY=0.915 _GAIN=1.5
#                              _MIN_USE=2 _LOG_S=10 _COPY=cudart|torch|batch; plugin/edge0_vllm/tier.py.
#                              EDGE0_TIER_EARLY=1: decide right after the forward (at the next step's PLE staging,
#                              before its stream sync), copy during the LM head / sampler / drafter, flip before the
#                              next launch: a swap is resident one step earlier (TUNING-LOG "Single-stream" R3).
#                              EDGE0_TIER_SHADOW (default = EDGE0_TIER_EARLY): a swapped-in expert keeps its host slot,
#                              so its eviction needs no D2H write-back; EDGE0_TIER_SPARE=0 more spare host slots / layer
#   EDGE0_DFLASH_SHARE_VOCAB=0 1 = build the DFlash2 drafter without its own BF16 embed_tokens / lm_head (vLLM replaces
#                              both with the target's after loading anyway): no 2.36 GiB load transient
#   EDGE0_ROUTE_FUSE=0         1 = the routing tail of a decode-sized MoE call (padding reroute + tier count) in one
#                              kernel instead of six (kernels.route_post, bit-identical; TUNING-LOG "Single-stream" R2)
#   EDGE0_PREFETCH=0          S > 0: S VRAM staging slots per layer, next layer's predicted cold experts copied during
#                              this layer (EDGE0_PREFETCH_MIN_P, EDGE0_PREFETCH_CTAS); costs S x 47 x 1.85 MB extra VRAM
#   EDGE0_HOT_BSM / EDGE0_COLD_BSM=0, EDGE0_COLD_BLOCKS_PER_SM=-1   per-tier Marlin M-block / CTA cap (0/-1 = vLLM's)
#   EDGE0_PLE_IO=vllm          vllm (serial preadv) | threads (EDGE0_PLE_THREADS readers)
#   EDGE0_PLE_DEFER=0          (single: 1) 1 = PLE rows off the critical path: a C++ reader thread reads the step's rows
#                              while the GPU runs the launch and layer 0; the PLE layer waits for them inside the graph
#                              (plugin/edge0_vllm/ple_defer.py; TUNING-LOG "Single-stream" R1: +4.2% at c1, +5.4% at c8).
#                              EDGE0_PLE_DEFER_WAKE=doorbell|event, _SYNC=1 (0 = no stream sync: host-bound, slower),
#                              _TIMEOUT_MS=2000, _CPU=-1, EDGE0_PLE_LOG_S=30
#   Single-stream dense path (TUNING-LOG "Single-stream" D1; plugin/edge0_vllm/q8gemv.py, lmhead_q8.py; all need
#   EDGE0_DENSE=q8; calls of <= EDGE0_Q8_GEMV_MAX_M=8 tokens, larger ones keep b12x):
#   EDGE0_DU_MASK=0            (single: 1) 1 = latent D / U only over the routing groups the step's tokens use (Triton Q8 GEMV)
#   EDGE0_SHARED_FUSE=0        (single: 1) 1 = shared expert + sigmoid gate + residual add in 4 kernels instead of ~10
#   EDGE0_LMHEAD=bf16          (single: q8) q8 = LM head re-quantized to Q8_0 (its GGUF format) and served by b12x: half the bytes for
#                              the target's logits and the DFlash2 drafter (shares the head), frees 0.6 GiB for KV
#   EDGE0_Q8_GEMV=0            1 = every small-M Q8_0 linear through the Triton GEMV (measured neutral; not used)
#   EDGE0_MOE_CHUNK_TOKENS=0   tokens per Marlin call (0 = MAX_BATCHED)
#   EDGE0_CPU_EXPERTS=0        (tuned: 1, single: 1) 1 = Strata-style CPU computation of part of every decode call's cold experts, in parallel
#                              with the GPU's PCIe reads (docs/CPU-EXPERTS.md; TUNING-LOG "CPU cold experts"): knobs
#                              EDGE0_CPU_THREADS=8 EDGE0_CPU_CORES (default physical cores 2..)
#                              EDGE0_CPU_FRAC=0.55 (share of a call's cold experts; auto = balance
#                              CPU and GPU time) EDGE0_CPU_MIN_TOKENS=2 / _MAX_TOKENS=256 (call sizes) _MIN_COLD=4
#                              _MAX_ROUTES=8 _SPIN_MS=200 _TIMEOUT_MS=2000 _LOG_S=30
#   EDGE0_NVTX=0               1 = NVTX ranges (MoE op parts, PLE staging, runner/scheduler phases); 2 = also one range
#                              per module forward (eager calibration). plugin/edge0_vllm/nvtx.py, docs/PROFILE.md
#   NSYS=                      run 'vllm serve' under Nsight Systems: the value is the 'nsys profile' option string, e.g.
#                              NSYS="--trace=cuda,nvtx,osrt --cuda-graph-trace=node --capture-range=cudaProfilerApi
#                              --capture-range-end=repeat -o /workspace/edge0/profiles/run" plus serve.sh args
#                              --profiler-config '{"profiler":"cuda"}' (then POST /start_profile, /stop_profile);
#                              the host's NSYS_DIR=/opt/nvidia/nsight-systems/2025.3.2 is mounted read-only
#                              (bench/nsys-capture.sh drives this)
#   EDGE0_VERIFY_LOAD=0        1 = compare every loaded Q8_0 / narrowed F32 param with a numpy decode of the checkpoint
#                              (edge0_vllm/loadcheck.py; fails the load on a mismatch). Also checks that every Q8_0
#                              tensor went through the dequant loader and folds vLLM's 449 expected "Attempted to load
#                              weight ... dtype" warnings into summary lines. Off (0): no change to the load path.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

GPU_FREE=${GPU_FREE:-}
NAME=${NAME:-edge0-vllm}
TMUX_SESSION=${TMUX_SESSION:-edge0-vllm}
IMAGE=${IMAGE:-edge0-vllm:dev}
PORT=${PORT:-8125}
SERVED_NAME=${SERVED_NAME:-edge0_145b}
MODELS_ROOT=${MODELS_ROOT:-$HOME/models}
MODELS_ROOT=${MODELS_ROOT%/}
if [ -z "${MODEL_DIR:-}" ]; then
  MODEL_DIR=$MODELS_ROOT/Edge0-145B-NVFP4-vLLM
  [ ! -d "$MODEL_DIR" ] && [ -d "$MODELS_ROOT/Edge0-145B-vllm" ] && MODEL_DIR=$MODELS_ROOT/Edge0-145B-vllm
fi
MODEL_DIR=${MODEL_DIR%/}
WORK_DIR=${WORK_DIR:-$REPO}
CACHE_DIR=${CACHE_DIR:-$HOME/.cache/edge0-vllm}
# The PLE table may live on another disk: the checkpoint dir then holds an absolute symlink ple-fp8.gguf ->
# $PLE_DIR/ple-fp8.gguf (bench/plemove.sh), so PLE_DIR is mounted at the same path for the symlink to resolve inside
# the container.
if [ -z "${PLE_DIR+x}" ]; then
  PLE_DIR=
  PLE_LINK=$(readlink "$MODEL_DIR/ple-fp8.gguf" 2>/dev/null || true)
  case "$PLE_LINK" in /*) PLE_DIR=$(dirname "$PLE_LINK") ;; esac
fi
PLE_MOUNT=()
[ -n "$PLE_DIR" ] && [ -d "$PLE_DIR" ] && PLE_MOUNT=(-v "$PLE_DIR:$PLE_DIR:ro")

# Host dir -> container path: under MODELS_ROOT -> /models/<rel> (MODELS_ROOT is mounted once); elsewhere -> its own
# read-only mount at /mnt/edge0/<basename>.
MODEL_MOUNTS=()
ROOT_MOUNT=0
to_ctr() {
  local d=${1%/}
  case "$d" in
    "$MODELS_ROOT"/*) CTR="/models/${d#"$MODELS_ROOT"/}"; ROOT_MOUNT=1 ;;
    *) CTR="/mnt/edge0/$(basename "$d")"; MODEL_MOUNTS+=(-v "$d:$CTR:ro") ;;
  esac
}
to_ctr "$MODEL_DIR"; CTR_MODEL=$CTR

# ---- plugin knobs (DESIGN.md 2.9) -------------------------------------------------------------------------------
# PRESET=tuned (default): docs/TUNING-LOG.md section 5 plus CPU cold experts (TUNING-LOG "Final configuration"; no
# speculative decoding: add SPEC_METHOD=dflash2 SPEC_K=3 for <= ~6 concurrent requests, +65% at c1 but -19% at c32,
# only 6 requests fit). PRESET=bringup: the bring-up command (BRINGUP-LOG.md, run
# with KV_BYTES= as there). PRESET=single: concurrency 1 (TUNING-LOG "Single-stream"): DFlash2 k=3, CPU cold
# experts, the reactive expert tier (EDGE0_TIER), the small-M dense knobs (EDGE0_DU_MASK, EDGE0_SHARED_FUSE,
# EDGE0_LMHEAD=q8), PLE rows read off the critical path (EDGE0_PLE_DEFER), KV for a few requests and the freed VRAM
# given to hot experts (16.2 GiB, plan-s1-16.2), and the tier deciding right after the forward with its copies in the
# drafter window and shadow host copies of swapped-in experts (EDGE0_TIER_EARLY, _COPY=batch, _SPARE=64: +5.7 GB of
# pinned host memory; TUNING-LOG "Single-stream" R3).
# Any variable set explicitly still wins over the preset.
PRESET=${PRESET:-tuned}
case "$PRESET" in
  tuned)
    : "${EDGE0_EXPERT_VRAM_GIB=12.8}" "${EDGE0_PLACEMENT:=plan:/workspace/edge0/plans/plan-s1-12.8.json}"
    : "${EDGE0_DENSE:=q8}" "${EDGE0_COLD_BSM:=64}" "${EDGE0_BSM_MIN_TOKENS:=64}" "${EDGE0_ADAPT:=1}"
    : "${EDGE0_CPU_EXPERTS:=1}" "${EDGE0_CPU_THREADS:=8}" "${EDGE0_CPU_FRAC:=0.55}"
    : "${MAX_SEQS:=32}" "${KV_BYTES=}" "${SSM_DTYPE:=bfloat16}" "${PREFIX_CACHING:=0}" ;;
  single)    # 16.2 GiB = 15.2 + the 0.6 GiB the Q8 LM head frees (G2 / C1) + 0.5 GiB of headroom (R1; a 23k-token
             # prompt still fits)
    : "${EDGE0_EXPERT_VRAM_GIB=16.2}" "${EDGE0_PLACEMENT:=plan:/workspace/edge0/plans/plan-s1-16.2.json}"
    : "${EDGE0_DENSE:=q8}" "${EDGE0_COLD_BSM:=64}" "${EDGE0_BSM_MIN_TOKENS:=64}" "${EDGE0_TIER:=1}"
    : "${EDGE0_DU_MASK:=1}" "${EDGE0_SHARED_FUSE:=1}" "${EDGE0_LMHEAD:=q8}" "${EDGE0_PLE_DEFER:=1}"
    : "${EDGE0_TIER_EARLY:=1}" "${EDGE0_TIER_COPY:=batch}" "${EDGE0_TIER_SPARE:=64}"    # R3: early tier + shadows
    : "${EDGE0_CPU_EXPERTS:=1}" "${EDGE0_CPU_THREADS:=8}" "${EDGE0_CPU_FRAC:=0.55}"
    : "${SPEC_METHOD:=dflash2}" "${SPEC_K:=3}" "${MAX_BATCHED:=1024}"
    : "${MAX_SEQS:=4}" "${KV_BYTES=1879048192}" "${SSM_DTYPE:=bfloat16}" "${PREFIX_CACHING:=0}" ;;
  bringup)
    : "${EDGE0_EXPERT_VRAM_GIB=9.5}" "${EDGE0_PLACEMENT:=profile}" "${EDGE0_DENSE:=bf16}"
    : "${MAX_SEQS:=16}" "${KV_BYTES=3500000000}" "${SSM_DTYPE:=float32}" "${PREFIX_CACHING:=1}" ;;
  *) echo "PRESET=$PRESET: expected tuned | single | bringup" >&2; exit 1 ;;
esac
export EDGE0_EXPERT_VRAM_GIB EDGE0_PLACEMENT EDGE0_DENSE
[ -n "${EDGE0_COLD_BSM:-}" ] && export EDGE0_COLD_BSM EDGE0_BSM_MIN_TOKENS
[ -n "${EDGE0_ADAPT:-}" ] && export EDGE0_ADAPT
[ -n "${EDGE0_TIER:-}" ] && export EDGE0_TIER
[ -n "${EDGE0_DU_MASK:-}" ] && export EDGE0_DU_MASK
[ -n "${EDGE0_SHARED_FUSE:-}" ] && export EDGE0_SHARED_FUSE
[ -n "${EDGE0_LMHEAD:-}" ] && export EDGE0_LMHEAD
[ -n "${EDGE0_PLE_DEFER:-}" ] && export EDGE0_PLE_DEFER
[ -n "${EDGE0_TIER_EARLY:-}" ] && export EDGE0_TIER_EARLY
[ -n "${EDGE0_TIER_COPY:-}" ] && export EDGE0_TIER_COPY
[ -n "${EDGE0_TIER_SPARE:-}" ] && export EDGE0_TIER_SPARE
[ -n "${EDGE0_CPU_EXPERTS:-}" ] && export EDGE0_CPU_EXPERTS EDGE0_CPU_THREADS EDGE0_CPU_FRAC
export EDGE0_COLD_ALLOC=${EDGE0_COLD_ALLOC:-host_register}
export EDGE0_SPLIT=${EDGE0_SPLIT:-1}
export EDGE0_MOE_IMPL=${EDGE0_MOE_IMPL:-marlin}
export EDGE0_EMBED_HOST=${EDGE0_EMBED_HOST:-1}
export EDGE0_TRACE=${EDGE0_TRACE:-0}
export EDGE0_PLE_IO=${EDGE0_PLE_IO:-vllm}
EDGE0_ENV=()
for v in $(compgen -e | grep -E '^EDGE0_' | grep -v '^EDGE0_DEV$' || true); do EDGE0_ENV+=(-e "$v=${!v}"); done

# ---- vLLM flags (bring-up command, DESIGN.md 2.10) --------------------------------------------------------------
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_SEQS=${MAX_SEQS:-16}
PREFIX_CACHING=${PREFIX_CACHING:-1}
SPEC_ON=0
[ -n "${SPEC:-}" ] || [ "${SPEC_METHOD:-none}" != none ] && SPEC_ON=1
if [ "$SPEC_ON" = 1 ]; then    # speculative decoding: smaller prefill chunks and more headroom (TUNING-LOG "Draft length sweep" S1)
  MAX_BATCHED=${MAX_BATCHED:-${SPEC_MAX_BATCHED:-2048}}
  GPU_UTIL=${GPU_UTIL:-${SPEC_GPU_UTIL:-0.94}}
fi
MAX_BATCHED=${MAX_BATCHED:-4096}
GPU_UTIL=${GPU_UTIL:-0.96}
KV_BYTES=${KV_BYTES-3500000000}
KV_DTYPE=${KV_DTYPE:-auto}
SSM_DTYPE=${SSM_DTYPE:-float32}
SPEC=${SPEC:-}
SPEC_METHOD=${SPEC_METHOD:-none}
SPEC_K=${SPEC_K:-3}
DFLASH_DIR=${DFLASH_DIR:-$MODELS_ROOT/Qwen3.8-Flash-Next-DFlash2}
if [ -z "$SPEC" ] && [ "$SPEC_METHOD" != none ]; then
  case "$SPEC_METHOD" in
    mtp) SPEC="{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_K" ;;
    dflash2)
      to_ctr "$DFLASH_DIR"
      SPEC="{\"method\":\"dflash\",\"model\":\"$CTR\",\"num_speculative_tokens\":$SPEC_K" ;;
    *) echo "SPEC_METHOD=$SPEC_METHOD: expected none | mtp | dflash2" >&2; exit 1 ;;
  esac
  [ -n "${SPEC_SCHED:-}" ] && SPEC+=",\"num_speculative_tokens_per_batch_size\":$SPEC_SCHED"
  [ -n "${SPEC_CONF:-}" ] && SPEC+=",\"draft_confidence_threshold\":$SPEC_CONF"
  [ -n "${SPEC_VOCAB:-}" ] && SPEC+=",\"draft_token_map\":\"$SPEC_VOCAB\""
  [ -n "${SPEC_EXTRA:-}" ] && SPEC+=",$SPEC_EXTRA"
  SPEC+="}"
fi
VLLM_OFFLOAD_GB=${VLLM_OFFLOAD_GB:-0}
ENGRAM='{"cpu_offload": false, "host_file_gather": true}'

ARGS=(
  vllm serve "$CTR_MODEL"
  --served-model-name "$SERVED_NAME" --host 0.0.0.0 --port "$PORT"
  --quantization edge0 --dtype bfloat16
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$MAX_BATCHED"
  --gpu-memory-utilization "$GPU_UTIL"
  --kv-cache-dtype "$KV_DTYPE" --mamba-ssm-cache-dtype "$SSM_DTYPE"
  --engram-config "$ENGRAM"
  --generation-config auto
)
[ -n "$KV_BYTES" ] && ARGS+=(--kv-cache-memory-bytes "$KV_BYTES")
[ "${EAGER:-0}" = 1 ] && ARGS+=(--enforce-eager)
[ -n "$SPEC" ] && ARGS+=(--speculative-config "$SPEC")
[ "$PREFIX_CACHING" = 0 ] && ARGS+=(--no-enable-prefix-caching)
[ "$VLLM_OFFLOAD_GB" != 0 ] && ARGS+=(--offload-backend uva --cpu-offload-gb "$VLLM_OFFLOAD_GB")
if [ "${EDGE0_MOE_IMPL}" = torch ] && [ "${EAGER:-0}" != 1 ]; then
  echo "EDGE0_MOE_IMPL=torch needs EAGER=1 (the reference path syncs with the host)" >&2; exit 1
fi

DEV_ENV=()
[ "${EDGE0_DEV:-0}" = 1 ] && DEV_ENV=(-e PYTHONPATH=/workspace/edge0/plugin)
NSYS_DIR=${NSYS_DIR:-/opt/nvidia/nsight-systems/2025.3.2}
if [ -n "${NSYS:-}" ]; then           # Nsight Systems wrapper (docs/PROFILE.md); the image has no nsys of its own
  read -r -a NSYS_OPTS <<<"$NSYS"
  DEV_ENV+=(-v "$NSYS_DIR:$NSYS_DIR:ro")
  ARGS=("$NSYS_DIR/bin/nsys" profile "${NSYS_OPTS[@]}" "${ARGS[@]}")
fi

[ "$ROOT_MOUNT" = 1 ] && MODEL_MOUNTS=(-v "$MODELS_ROOT:/models" "${MODEL_MOUNTS[@]}")

DOCKER=(docker run --rm --name "$NAME" --gpus all --ipc=host --network host
  --ulimit memlock=-1 --ulimit stack=67108864
  "${MODEL_MOUNTS[@]}" -v "$WORK_DIR:/workspace/edge0" -v "$CACHE_DIR:/root/.cache" "${PLE_MOUNT[@]}"
  -e HF_HUB_OFFLINE=1 -e VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-7200}
  -e TORCH_DIST_USE_NCCL2=${TORCH_DIST_USE_NCCL2:-0}
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-128}
  "${DEV_ENV[@]}" "${EDGE0_ENV[@]}"
  "$IMAGE" "${ARGS[@]}" "$@")

if [ "${DRY_RUN:-0}" = 1 ]; then printf '%q ' "${DOCKER[@]}"; echo; exit 0; fi

if [ -n "$GPU_FREE" ] && [ ! -e "$GPU_FREE" ]; then
  echo "refusing to start: GPU_FREE=$GPU_FREE does not exist (another engine holds the GPU)" >&2
  exit 1
fi
if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "MODEL_DIR=$MODEL_DIR has no config.json: hf download andynoodles/Edge0-145B-NVFP4-vLLM --local-dir <dir>," \
       "then MODEL_DIR=<dir> ./serve.sh" >&2
  exit 1
fi
if [ "$SPEC_METHOD" = dflash2 ] && [ ! -f "$DFLASH_DIR/config.json" ]; then
  echo "DFLASH_DIR=$DFLASH_DIR has no config.json (PRESET=single uses the DFlash2 drafter):" \
       "hf download incoai/Qwen3.8-Flash-Next-DFlash2 --local-dir <dir>, then DFLASH_DIR=<dir>; or SPEC_METHOD=none" >&2
  exit 1
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "image $IMAGE missing: cd $WORK_DIR && docker build -f docker/Dockerfile -t $IMAGE ." >&2; exit 1
fi
if ss -ltn "( sport = :$PORT )" 2>/dev/null | grep -q LISTEN; then
  echo "port $PORT is already in use" >&2; exit 1
fi
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "tmux session $TMUX_SESSION exists; stop it first: tmux kill-session -t $TMUX_SESSION" >&2; exit 1
fi
mkdir -p "$CACHE_DIR" "$WORK_DIR/logs"
docker rm -f "$NAME" >/dev/null 2>&1 || true
LOG="$WORK_DIR/logs/serve-$(date +%Y%m%d-%H%M%S).log"
{ echo "# $(date -Is) $(printf '%q ' "${DOCKER[@]}")"; } > "$LOG"
tmux new-session -d -s "$TMUX_SESSION" "$(printf '%q ' "${DOCKER[@]}") 2>&1 | tee -a $(printf '%q' "$LOG")"
echo "started $NAME in tmux session $TMUX_SESSION on :$PORT (model $CTR_MODEL, hot experts ${EDGE0_EXPERT_VRAM_GIB} GiB," \
     "placement $EDGE0_PLACEMENT); log $LOG"
