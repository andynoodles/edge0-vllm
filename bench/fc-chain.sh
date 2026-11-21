#!/usr/bin/env bash
# Job chains of agent fast-cache (TUNING-LOG "Single-stream" G2). Runs ON the 5090 host in tmux edge0-fast-cache-*,
# holding the GPU lock. Every server: EDGE0_DEV=1 (plugin source from this tree).
#   bench/fc-chain.sh gate      GPU unit tests, probe of the tier (correctness + greedy), probe of the baseline (trace)
#   bench/fc-chain.sh ab1       ABBA: best c1 config (DFlash2 k=3 + CPU experts, tuned preset) vs + EDGE0_TIER=1
#   bench/fc-chain.sh ab2       ABBA: + EDGE0_TIER=1 vs PRESET=single
#   bench/fc-chain.sh ab3       ABBA without speculation: tuned + CPU experts vs + EDGE0_TIER=1; greedy of the tier
#                               without drafter (spec gate)
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE="EDGE0_DEV=1 SPEC_METHOD=dflash2 SPEC_K=3 EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55"
NOSPEC="EDGE0_DEV=1 EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55"
case "${1:-}" in
  gate)
    bash "$REPO/bench/fc-gate.sh"
    bash "$REPO/bench/fc-probe.sh" fctier "$BASE EDGE0_TIER=1"
    CORRECTNESS=0 GREEDY=0 bash "$REPO/bench/fc-probe.sh" fcbase "$BASE EDGE0_TRACE=bw" ;;
  ab1) bash "$REPO/bench/fc-ab.sh" fcab1 "$BASE" "$BASE EDGE0_TIER=1" ;;
  ab2) bash "$REPO/bench/fc-ab.sh" fcab2 "$BASE EDGE0_TIER=1" "EDGE0_DEV=1 PRESET=single" ;;
  ab3)
    bash "$REPO/bench/fc-ab.sh" fcab3 "$NOSPEC" "$NOSPEC EDGE0_TIER=1"
    CORRECTNESS=0 bash "$REPO/bench/fc-probe.sh" fctier-nospec "$NOSPEC EDGE0_TIER=1"
    ( cd "$REPO/correctness" && python3 -I greedy_same.py diff "$REPO/results/greedy-fctier-nospec.json" \
        "$REPO/results/greedy-fctier.json" ) >"$REPO/results/greedy-fctier-diff.txt" 2>&1
    echo "GREEDY_DIFF exit $? $(tail -1 "$REPO/results/greedy-fctier-diff.txt")" ;;
  ab4)       # PRESET=single vs + dense-c1's kept knobs (D1) with the 0.6 GiB the Q8 LM head frees given to experts
    S2="EDGE0_DEV=1 PRESET=single EDGE0_DU_MASK=1 EDGE0_SHARED_FUSE=1 EDGE0_LMHEAD=q8 EDGE0_EXPERT_VRAM_GIB=15.7"
    S2="$S2 EDGE0_PLACEMENT=plan:/workspace/edge0/plans/plan-s1-15.7.json"
    bash "$REPO/bench/fc-ab.sh" fcab4 "EDGE0_DEV=1 PRESET=single" "$S2"
    LONG=6000 bash "$REPO/bench/fc-probe.sh" fcsingle2 "$S2" ;;
  *) echo "usage: $0 gate|ab1|ab2|ab3|ab4" >&2; exit 1 ;;
esac
echo "FC_CHAIN_DONE ${1:-}"
