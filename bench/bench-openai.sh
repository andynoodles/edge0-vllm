#!/usr/bin/env bash
# Throughput sweep against any OpenAI-compatible server on the 5090 box, using the same workload as
# TidalMoE's bench/bench.sh: vllm-bench, chat API, ShareGPT prompts, 1024 output tokens (ignore EOS),
# temperature 0, 4 x concurrency prompts per point, 2 warm-up requests.
#
#   BASE_URL=http://127.0.0.1:8124 MODEL=edge0_145b EXTRA_BODY='{"reasoning_effort":"none"}' \
#     bench-openai.sh <label> [concurrencies]
#
# Edge0's own tokenizer/tokenizer.json is GGUF metadata, not a HF tokenizer; its vocabulary is identical to
# Qwen/Qwen3.8-Flash-Next's (plus [PADnnnnnn] fillers up to 248320), so the Qwen tokenizer is used here.
set -euo pipefail
LABEL=$1; CONCS=${2:-1,2,4,8,16,32}
BENCH=${BENCH:-$HOME/work/edge0-bench}
BASE_URL=${BASE_URL:-http://127.0.0.1:8124}
MODEL=${MODEL:-edge0_145b}
EXTRA_BODY=${EXTRA_BODY:-'{"reasoning_effort":"none"}'}
OUT_LEN=${OUT_LEN:-1024}
FACTOR=${FACTOR:-4}            # prompts per point = FACTOR x concurrency (A/B runs: see ab.sh)
SEED=${SEED:-0}                # sampling seed of the ShareGPT prompts (0 = the standard benchmark set)
OPT=()
[ "${DETAILED:-0}" = 1 ] && OPT+=(--save-detailed)     # per-request ttfts / itls / latencies / output_lens in the JSON
mkdir -p "$BENCH/results/$LABEL"
exec "$HOME/.local/bin/vllm-bench" "${OPT[@]}" --backend openai-chat --base-url "$BASE_URL" --model "$MODEL" \
  --tokenizer "$BENCH/tokenizer-qwen38" \
  --dataset-name sharegpt --dataset-path "$BENCH/datasets/ShareGPT_V3_unfiltered_cleaned_split.json" \
  --sharegpt-output-len "$OUT_LEN" --ignore-eos --temperature 0 --seed "$SEED" --extra-body "$EXTRA_BODY" \
  --sweep-max-concurrency "$CONCS" --sweep-num-prompts-factor "$FACTOR" --num-warmups 2 \
  --percentile-metrics ttft,tpot,itl,e2el --save-result --result-dir "$BENCH/results/$LABEL" --label "$LABEL" \
  --disable-tqdm
