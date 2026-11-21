#!/usr/bin/env bash
# dense-c1 run 1 (TUNING-LOG "Single-stream" D1): gates and ABBAs of the dense-path knobs. ON the 5090, in tmux, GPU lock held.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
C=$REPO/bench/chain-dense.sh
SPEC="EDGE0_DEV=1 SPEC_METHOD=dflash2 SPEC_K=3 EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55"
NOSPEC="EDGE0_DEV=1 EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55"
KNOBS="EDGE0_DU_MASK=1 EDGE0_SHARED_FUSE=1 EDGE0_LMHEAD=q8"
case "${1:-all}" in
  gates)
    bash "$C" gate dense-nospec-B "$NOSPEC $KNOBS" 1
    bash "$C" gate dense-spec-B "$SPEC $KNOBS" 0
    python3 -I "$REPO/correctness/greedy_same.py" diff "$REPO/results/greedy-dense-nospec-B.json" \
      "$REPO/results/greedy-dense-spec-B.json" >>"$REPO/logs/chain-dense.log" 2>&1
    echo "greedy diff exit $?" >>"$REPO/logs/chain-dense.log" ;;
  abspec)
    WARM_PROMPTS=4 bash "$C" ab dense-spec 1,8 "$SPEC" "$SPEC $KNOBS" ;;
  abnospec)
    WARM_PROMPTS=4 bash "$C" ab dense-nospec 1 "$NOSPEC" "$NOSPEC $KNOBS" ;;
  abab-nospec)    # every pass on a fresh server after an 8-prompt warm pass (no second-pass warmth bias)
    WARM_PROMPTS=8 bash "$C" ab dense-nospec2 1 "$NOSPEC" "$NOSPEC $KNOBS" ABAB ;;
  abab-spec)
    WARM_PROMPTS=8 bash "$C" ab dense-spec2 1 "$SPEC" "$SPEC $KNOBS" ABAB ;;
  nsys)            # in-situ kernel attribution of B (warm capture), to compare with profiles/gapdf3w (A)
    bash "$REPO/bench/gap-nsys.sh" denseB "$SPEC $KNOBS EDGE0_NVTX=1"
    for _ in $(seq 1 120); do pgrep -f "^/usr/local/bin/nsys export" >/dev/null || break; sleep 5; done
    python3 "$REPO/tools/nsys_kernels.py" "$REPO/profiles/denseBw/c1-k1.sqlite" --steps 30 \
      >"$REPO/results/nsys-kernels-denseBw.txt" 2>&1
    python3 "$REPO/tools/nsys_kernels.py" "$REPO/profiles/gapdf3w/c1-k1.sqlite" --steps 30 \
      >"$REPO/results/nsys-kernels-gapdf3w.txt" 2>&1 ;;
  all)
    "$0" gates; "$0" abspec; "$0" abnospec ;;
esac
echo "# $(date +%H:%M:%S) run1 ${1:-all} DONE" >>"$REPO/logs/chain-dense.log"
