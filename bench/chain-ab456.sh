#!/usr/bin/env bash
# One-off chain (tune-experts agent, tmux edge0-chain2): after chain-ab23, against one common baseline BEST
# (q8 dense, 32 sequences, bf16 GDN state, no prefix caching, seed-1 placement plan at 12.8 GiB):
#   4. cold-call M block 64 for calls above 64 tokens (Marlin re-read fix)
#   5. sticky next-layer prefetch, 8 staging slots/layer (plan shrunk to 12.1 GiB: same total VRAM)
#   6. adaptive hot/cold swaps
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG=$REPO/logs/chain-ab456.log
say() { echo "# $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
until grep -q CHAIN_DONE "$REPO/logs/chain-ab23.log" 2>/dev/null; do sleep 20; done
P=/workspace/edge0/plans
BASE="KV_BYTES= EDGE0_DENSE=q8 SSM_DTYPE=bfloat16 PREFIX_CACHING=0 MAX_SEQS=32 EDGE0_EXPERT_VRAM_GIB=12.8"
BEST="$BASE EDGE0_PLACEMENT=plan:$P/plan-s1-12.8.json"
say "A/B cold bsm"
bash "$REPO/bench/ab.sh" coldbsm 1,8,32 "$BEST" "$BEST EDGE0_COLD_BSM=64 EDGE0_BSM_MIN_TOKENS=64" ABA
say "A/B prefetch"
bash "$REPO/bench/ab.sh" prefetch 1,8,32 "$BEST" \
  "$BASE EDGE0_PLACEMENT=plan:$P/plan-s1-12.1.json EDGE0_PREFETCH=8" ABA
say "A/B adapt"
bash "$REPO/bench/ab.sh" adapt 1,8,32 "$BEST" "$BEST EDGE0_ADAPT=1" ABA
say "CHAIN2_DONE"
