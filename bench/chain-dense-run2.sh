#!/usr/bin/env bash
# dense-c1 run 2 (TUNING-LOG "Single-stream" D1): B+ = B with the Q8 head on the Triton GEMV for <= 16-row calls (the
# DFlash2 drafter's head call) and the QSA q/k/v rows as Q8_0 (EDGE0_QSA_Q8=1). ON the 5090, in tmux, GPU lock held.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
C=$REPO/bench/chain-dense.sh
LOG=$REPO/logs/chain-dense.log
SPEC="EDGE0_DEV=1 SPEC_METHOD=dflash2 SPEC_K=3 EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55"
NOSPEC="EDGE0_DEV=1 EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55"
KNOBS="EDGE0_DU_MASK=1 EDGE0_SHARED_FUSE=1 EDGE0_LMHEAD=q8 EDGE0_QSA_Q8=1"
B16="EDGE0_EXPERT_VRAM_GIB=16.0 EDGE0_PLACEMENT=plan:/workspace/edge0/plans/plan-s1-16.0.json"
case "${1:-all}" in
  headbench)
    docker run --rm --gpus all --ipc=host -v "$REPO:/workspace/edge0" -v /home/andychen/.cache/edge0-vllm:/root/.cache \
      -e PYTHONPATH=/workspace/edge0/plugin --entrypoint python3 edge0-vllm:dev \
      /workspace/edge0/tools/q8gemv_bench.py --m 1,4,16 --shapes lm_head,qsa_qkv 2>&1 | grep -v "not prepared" | tee -a "$LOG" ;;
  gates)
    bash "$C" gate dense-nospec-B2 "$NOSPEC $KNOBS" 1
    bash "$C" gate dense-spec-B2 "$SPEC $KNOBS" 0
    python3 -I "$REPO/correctness/greedy_same.py" diff "$REPO/results/greedy-dense-nospec-B2.json" \
      "$REPO/results/greedy-dense-spec-B2.json" >>"$LOG" 2>&1
    echo "greedy diff exit $?" >>"$LOG" ;;
  abab-spec)
    WARM_PROMPTS=8 bash "$C" ab dense-spec3 1 "$SPEC" "$SPEC $KNOBS" ABAB ;;
  abab-single)    # on top of the fast-cache agent's c1 preset (reactive tier, 15.2 GiB experts, DFlash2 k=3, CPU experts);
                  # B also gives the ~0.9 GiB the Q8 head + QSA free to hot experts (16.0 GiB, plans/plan-s1-16.0.json)
    WARM_PROMPTS=8 bash "$C" ab dense-single 1 "PRESET=single EDGE0_DEV=1" "PRESET=single EDGE0_DEV=1 $KNOBS $B16" ABAB ;;
  gate-single)
    bash "$C" gate dense-single-B "PRESET=single EDGE0_DEV=1 $KNOBS $B16" 0 ;;
  abab-tier)      # same on the tuned preset + EDGE0_TIER=1 (if PRESET=single is not usable)
    WARM_PROMPTS=8 bash "$C" ab dense-tier 1 "$SPEC EDGE0_TIER=1" "$SPEC EDGE0_TIER=1 $KNOBS" ABAB ;;
  ab-c8)          # regression check of the knobs at c8 without speculation (decode calls of 8 tokens use them)
    bash "$C" ab dense-c8 8 "$NOSPEC" "$NOSPEC $KNOBS" ABBA ;;
  all)
    "$0" headbench; "$0" gates; "$0" gate-single; "$0" abab-single ;;
esac
echo "# $(date +%H:%M:%S) run2 ${1:-all} DONE" >>"$LOG"
