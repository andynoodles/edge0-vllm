#!/bin/bash
cd ~/work/edge0-vllm
until grep -q NSYS_DONE logs/nsys-nospec.log; do sleep 5; done
GPUM=0 NSYS_OPTS="--trace=cuda,nvtx,osrt --cuda-graph-trace=graph --capture-range=cudaProfilerApi --capture-range-end=repeat --trace-fork-before-exec=true --sample=none --cpuctxsw=none --force-overwrite=true -o /workspace/edge0/profiles/graph/rep" \
  bash bench/nsys-capture.sh graph "EDGE0_DEV=1 EDGE0_NVTX=1" 1:150:6 8:60:10 32:30:15
GPUM=0 SETTLE_S=8 bash bench/nsys-capture.sh eager "EDGE0_DEV=1 EDGE0_NVTX=2 EAGER=1" 1:20:15 8:12:25 32:8:40
echo CHAIN_DONE
