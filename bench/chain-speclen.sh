#!/usr/bin/env bash
# One-off chain (agent spec-len, tmux edge0-spec-len-*): the draft-length sweep of TUNING-LOG "Draft length sweep".
# Runs ON the 5090 host while holding the GPU lock. Every config is the tuned preset (PRESET=tuned, 32 sequences,
# 12.8 GiB plan) plus serve.sh's speculative defaults (MAX_BATCHED 2048, GPU_UTIL 0.94). Long drafts do not fit the
# tuned VRAM split (one 32k request needs 1.66 GiB of KV at k=7, 2.45 GiB at k=15), so they run with
# MAX_MODEL_LEN=8192 and k=15 with a smaller hot-expert plan; they get c1/c8 only (1-2 requests fit, c32 = c8).
#
#   bench/chain-speclen.sh part2      DFlash2 k=7, k=15 (after part 1 = run-configs lendf: none, k=3, k=5)
#   bench/chain-speclen.sh mtp        MTP k=3, 5, 7, 15, with and without draft_confidence_threshold 0.5
#   bench/chain-speclen.sh sched      DFlash2 k=3/5/7 with CPU cold experts at c1; DFlash2 k<=3 with a batch-size
#                                     schedule (with / without CPU experts, 12.8 / 11.0 GiB); MTP k=15 + confidence
#   bench/chain-speclen.sh ab         ABBA: tuned + speclen-A.env vs tuned + speclen-A.env + speclen-B.env
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RC="$REPO/bench/run-configs.sh"
P=/workspace/edge0/plans
D="EDGE0_DEV=1"
L8="MAX_MODEL_LEN=8192"
case "${1:-}" in
  part2)
    GREEDY=1 bash "$RC" lendf2 1,8 \
      "$D SPEC_METHOD=dflash2 SPEC_K=7 $L8" \
      "$D SPEC_METHOD=dflash2 SPEC_K=15 $L8 EDGE0_EXPERT_VRAM_GIB=12.1 EDGE0_PLACEMENT=plan:$P/plan-s1-12.1.json" ;;
  mtp)
    FACTOR_C32=1 GREEDY=1 bash "$RC" lenmtp 1,8,32 \
      "$D SPEC_METHOD=mtp SPEC_K=3" \
      "$D SPEC_METHOD=mtp SPEC_K=5"
    GREEDY=1 bash "$RC" lenmtp2 1,8 \
      "$D SPEC_METHOD=mtp SPEC_K=7 $L8" \
      "$D SPEC_METHOD=mtp SPEC_K=7 SPEC_CONF=0.5 $L8" \
      "$D SPEC_METHOD=mtp SPEC_K=15 $L8 EDGE0_EXPERT_VRAM_GIB=11.0 EDGE0_PLACEMENT=plan:$P/plan-s1-11.0.json" \
      "$D SPEC_METHOD=mtp SPEC_K=15 SPEC_CONF=0.5 $L8 EDGE0_EXPERT_VRAM_GIB=11.0 EDGE0_PLACEMENT=plan:$P/plan-s1-11.0.json" ;;
  sched)
    # does a cheaper verify (CPU cold experts, TUNING-LOG "CPU cold experts", kept) move the c1 draft-length optimum?
    CPU="EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55"
    GREEDY=1 bash "$RC" lenschcpu 1 \
      "$D SPEC_METHOD=dflash2 SPEC_K=3 $CPU" \
      "$D SPEC_METHOD=dflash2 SPEC_K=5 $CPU" \
      "$D SPEC_METHOD=dflash2 SPEC_K=7 $L8 $CPU"
    # batch-size-dependent draft length (num_speculative_tokens_per_batch_size); the GDN state is sized by the
    # maximum k (3), so the hot-expert budget decides how many requests fit: 12.8 (tuned, 6 fit) / 11.0 GiB (~11)
    S1='[[1,2,3],[3,32,2]]'
    S2='[[1,2,3],[3,8,2],[9,32,0]]'
    FACTOR_C32=1 GREEDY=1 CORRECTNESS=1 bash "$RC" lensch 1,8,32 \
      "$D SPEC_METHOD=dflash2 SPEC_K=3 SPEC_SCHED=$S1" \
      "$D SPEC_METHOD=dflash2 SPEC_K=3 SPEC_SCHED=$S1 $CPU" \
      "$D SPEC_METHOD=dflash2 SPEC_K=3 SPEC_SCHED=$S2 $CPU EDGE0_EXPERT_VRAM_GIB=11.0 EDGE0_PLACEMENT=plan:$P/plan-s1-11.0.json"
    # MTP k=15 + confidence 0.5 OOMed in CUDA-graph capture at 11 GiB / 0.94 (394 FULL graphs): retry with headroom
    GREEDY=1 bash "$RC" lenmtp3 1 \
      "$D SPEC_METHOD=mtp SPEC_K=15 SPEC_CONF=0.5 $L8 GPU_UTIL=0.90 EDGE0_EXPERT_VRAM_GIB=9.0 EDGE0_PLACEMENT=plan:$P/plan-s1-9.0.json" ;;
  ab)
    # interleaved A B B A: A = the best setting without a drafter (bench/speclen-A.env, appended to the tuned
    # preset), B = A + the best speculative setting (bench/speclen-B.env, from the sweep and the schedule test).
    # ABBA ends on A: a batched-decode greedy control on that server (8 prompts in flight, no drafter) shows how
    # often bf16 near-ties flip with batch shape alone.
    A=$(cat "$REPO/bench/speclen-A.env" 2>/dev/null)
    B=$(cat "$REPO/bench/speclen-B.env")
    # first a probe of vLLM's adaptive verification (the TidalMoE AV port: per-request draft admission from drafter
    # confidence and a startup-profiled cost curve), DFlash2 k=3 and k=5 on top of A
    AV='SPEC_EXTRA="enable_adaptive_verification":true'
    GREEDY=1 bash "$RC" lenav 1,8 \
      "$D $A SPEC_METHOD=dflash2 SPEC_K=3 $AV" \
      "$D $A SPEC_METHOD=dflash2 SPEC_K=5 $AV"
    bash "$REPO/bench/ab.sh" speclen 1,8,32 "$D $A" "$D $A $B" ABBA
    ( cd "$REPO/correctness" && python3 -I greedy_same.py gen --out "$REPO/results/greedy-speclen-A.json" \
        && python3 -I greedy_same.py gen --conc 8 --out "$REPO/results/greedy-speclen-A-c8.json" \
        && python3 -I greedy_same.py diff "$REPO/results/greedy-speclen-A.json" "$REPO/results/greedy-speclen-A-c8.json" ) \
      > "$REPO/results/greedy-speclen-control.txt" 2>&1
    tail -1 "$REPO/results/greedy-speclen-control.txt" | tee -a "$REPO/logs/chain-speclen.log" ;;
  *) echo "usage: $0 part2 | mtp | sched | ab" >&2; exit 1 ;;
esac
docker stop edge0-vllm >/dev/null 2>&1; tmux kill-session -t edge0-vllm 2>/dev/null
for _ in $(seq 1 60); do docker ps -a --format '{{.Names}}' | grep -q '^edge0-vllm$' || break; sleep 2; done
echo "# $(date +%H:%M:%S) SPECLEN_${1}_DONE" | tee -a "$REPO/logs/chain-speclen.log"
