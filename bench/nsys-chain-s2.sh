#!/bin/bash
# GPU block 2 of docs/PROFILE.md: DFlash2 k=3 (the dfcal-3 calibration config + NVTX), the same config without
# speculation, then the DRAM contention microbenchmark (server stopped).
cd ~/work/edge0-vllm
SPEC11="EDGE0_DEV=1 EDGE0_NVTX=1 MAX_SEQS=16 EDGE0_EXPERT_VRAM_GIB=11.0 EDGE0_PLACEMENT=plan:/workspace/edge0/plans/plan-s1-11.0.json"
bash bench/nsys-capture.sh dflash3 "$SPEC11 SPEC_METHOD=dflash2 SPEC_K=3" 1:100:8 8:40:12
GPUM=0 bash bench/nsys-capture.sh nospec11 "$SPEC11" 1:150:6 8:60:10
bash bench/dram-contention.sh
echo CHAIN_DONE
