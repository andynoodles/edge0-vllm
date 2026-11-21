#!/usr/bin/env bash
# One-off chain (agent cpu-experts, tmux edge0-cpu-experts-*): EDGE0_CPU_EXPERTS with speculative decoding and the
# in-server split / thread tuning (TUNING-LOG "CPU cold experts (user request)"). Runs ON the 5090 host; waits for the
# GPU lock itself, releases it at the end.
#
#   bench/chain-cpuexp.sh spec      greedy gate (CPU experts with / without DFlash2 k=3) + ABBA DFlash2 k=3 vs + CPU
#   bench/chain-cpuexp.sh tune      single passes at c8/c32: frac 0.5 / 0.6 / auto, 6 / 8 threads, min_tokens 1
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOCK=/home/andychen/work/edge0-vllm-GPU_LOCK
LOG=$REPO/logs/chain-cpuexp.log
say() { echo "# $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
until mkdir "$LOCK" 2>/dev/null; do sleep 60; done
echo "cpu-experts $(date -Is)" > "$LOCK/owner"
say "lock acquired ($1)"
D="EDGE0_DEV=1"
CPU="EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55"
DF="SPEC_METHOD=dflash2 SPEC_K=3"
case "${1:-}" in
  spec)
    GREEDY=1 bash "$REPO/bench/run-configs.sh" cpuexp-greedy 1 "$D $CPU" "$D $DF $CPU"
    ( cd "$REPO/correctness" && python3 -I greedy_same.py diff "$REPO/results/greedy-cpuexp-greedy-1.json" \
        "$REPO/results/greedy-cpuexp-greedy-2.json" ) > "$REPO/results/greedy-cpuexp-diff.txt" 2>&1
    say "greedy diff exit $? (results/greedy-cpuexp-diff.txt)"
    bash "$REPO/bench/ab.sh" cpuexp-df3 1,8,32 "$D $DF" "$D $DF $CPU" ABBA ;;
  tune)
    bash "$REPO/bench/run-configs.sh" cpuexp-tune 8,32 \
      "$D EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.6" \
      "$D EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=auto" \
      "$D EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=6 EDGE0_CPU_FRAC=0.55"
    bash "$REPO/bench/run-configs.sh" cpuexp-tune1 1 \
      "$D EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55 EDGE0_CPU_MIN_TOKENS=1" ;;
  *) echo "usage: $0 spec | tune" >&2 ;;
esac
docker stop edge0-vllm >/dev/null 2>&1; tmux kill-session -t edge0-vllm 2>/dev/null
for _ in $(seq 1 60); do docker ps -a --format '{{.Names}}' | grep -q '^edge0-vllm$' || break; sleep 2; done
rm -rf "$LOCK"
say "CPUEXP_${1}_DONE, lock released"
