| label | config | KV GiB | conc | out tok/s | TPOT ms | acc len | step ms | per-position acceptance | preempt | peak MiB | max running |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| memfix-1 | `EDGE0_DEV=1` | 4.61 | 32 | 140.7 | 208.9 |  | 208.9 |  | 0 | 31997 | 32 |
| memfix-2 | `MAX_SEQS=32 EDGE0_EXPERT_VRAM_GIB=8.0 EDGE0_PLACEMENT=plan:/workspace/edge0/plans/plan-s1-8.0.json SPEC_METHOD=mtp SPEC_K=1 GPU_UTIL=0.92` | 5.83 | 32 | 117.7 | 176.1 | 1.77 | 311.7 | 0.77 | 0 | 31389 | 27 |
| memfix-3 | `MAX_SEQS=32 EDGE0_EXPERT_VRAM_GIB=8.0 EDGE0_PLACEMENT=plan:/workspace/edge0/plans/plan-s1-8.0.json SPEC_METHOD=mtp SPEC_K=1 GPU_UTIL=0.92 MAX_BATCHED=2048` | 6.75 | 32 | 125.4 | 208.4 | 1.764 | 367.7 | 0.76 | 0 | 30789 | 31 |
| memfix-4 | `MAX_SEQS=32 EDGE0_EXPERT_VRAM_GIB=8.0 EDGE0_PLACEMENT=plan:/workspace/edge0/plans/plan-s1-8.0.json SPEC_METHOD=dflash2 SPEC_K=3 GPU_UTIL=0.92` | 5.78 | 32 | 104.0 | 126.3 | 2.603 | 328.8 | 0.74/0.52/0.35 | 0 | 30495 | 15 |
| lendf-1 | `EDGE0_DEV=1` | 4.61 | 1 | 50.7 | 19.3 |  | 19.3 |  | 0 | 31263 | 32 |
| lendf-1 | `EDGE0_DEV=1` | 4.61 | 8 | 105.9 | 72.2 |  | 72.2 |  | 0 | 31609 | 32 |
| lendf-1 | `EDGE0_DEV=1` | 4.61 | 32 | 143.9 | 215.5 |  | 215.5 |  | 0 | 31943 | 32 |
| lendf-2 | `SPEC_METHOD=dflash2 SPEC_K=3` | 2.47 | 1 | 70.3 | 13.8 | 2.693 | 37.1 | 0.77/0.55/0.38 | 0 | 30813 | 6 |
| lendf-2 | `SPEC_METHOD=dflash2 SPEC_K=3` | 2.47 | 8 | 109.6 | 48.7 | 2.646 | 128.9 | 0.74/0.53/0.37 | 0 | 30945 | 6 |
| lendf-2 | `SPEC_METHOD=dflash2 SPEC_K=3` | 2.47 | 32 | 116.0 | 48.6 | 2.697 | 131.2 | 0.76/0.55/0.39 | 0 | 30945 | 6 |
| lendf-3 | `SPEC_METHOD=dflash2 SPEC_K=5` | 1.88 | 1 | 71.6 | 13.5 | 3.242 | 43.8 | 0.78/0.58/0.41/0.28/0.19 | 0 | 30863 | 3 |
| lendf-3 | `SPEC_METHOD=dflash2 SPEC_K=5` | 1.88 | 8 | 79.3 | 35.8 | 3.124 | 111.9 | 0.74/0.54/0.38/0.27/0.20 | 0 | 31169 | 3 |
| lendf-3 | `SPEC_METHOD=dflash2 SPEC_K=5` | 1.88 | 32 | 81.0 | 35.7 | 3.091 | 110.4 | 0.74/0.53/0.37/0.26/0.19 | 0 | 31169 | 3 |
| lendf2-1 | `SPEC_METHOD=dflash2 SPEC_K=7 MAX_MODEL_LEN=8192` | 1.57 | 1 | 63.9 | 15.2 | 3.316 | 50.3 | 0.77/0.53/0.38/0.26/0.18/0.12/0.07 | 0 | 30817 | 2 |
| lendf2-1 | `SPEC_METHOD=dflash2 SPEC_K=7 MAX_MODEL_LEN=8192` | 1.57 | 8 | 73.6 | 24.6 | 3.302 | 81.1 | 0.73/0.52/0.36/0.26/0.19/0.14/0.10 | 10 | 31185 | 2 |
| lendf2-2 | `SPEC_METHOD=dflash2 SPEC_K=15 MAX_MODEL_LEN=8192 EDGE0_EXPERT_VRAM_GIB=12.1 EDGE0_PLACEMENT=plan:/workspace/edge0/plans/plan-s1-12.1.json` | 1.92 | 1 | 42.4 | 23.1 | 3.273 | 75.6 | 0.76/0.52/0.33/0.24/0.16/0.11/0.07/0.04/0.03/0.01/0.01/0.00/0.00/0.00/0.00 | 0 | 30689 | 1 |
| lendf2-2 | `SPEC_METHOD=dflash2 SPEC_K=15 MAX_MODEL_LEN=8192 EDGE0_EXPERT_VRAM_GIB=12.1 EDGE0_PLACEMENT=plan:/workspace/edge0/plans/plan-s1-12.1.json` | 1.92 | 8 | 48.0 | 20.1 | 3.423 | 68.6 | 0.73/0.52/0.36/0.25/0.17/0.12/0.09/0.07/0.04/0.03/0.02/0.01/0.01/0.00/0.00 | 0 | 30929 | 1 |
| lenmtp-1 | `SPEC_METHOD=mtp SPEC_K=3` | 2.15 | 1 | 71.1 | 13.6 | 2.635 | 35.7 | 0.74/0.53/0.36 | 0 | 31113 | 5 |
| lenmtp-1 | `SPEC_METHOD=mtp SPEC_K=3` | 2.15 | 8 | 104.1 | 43.3 | 2.713 | 117.4 | 0.76/0.56/0.40 | 0 | 31243 | 5 |
| lenmtp-1 | `SPEC_METHOD=mtp SPEC_K=3` | 2.15 | 32 | 111.1 | 42.2 | 2.725 | 115.0 | 0.77/0.56/0.40 | 0 | 31243 | 5 |
| lenmtp-2 | `SPEC_METHOD=mtp SPEC_K=5` | 1.56 | 1 | 63.6 | 15.2 | 3.042 | 46.4 | 0.72/0.52/0.36/0.26/0.19 | 0 | 30965 | 2 |
| lenmtp-2 | `SPEC_METHOD=mtp SPEC_K=5` | 1.56 | 8 | 81.3 | 22.8 | 3.219 | 73.3 | 0.76/0.55/0.40/0.29/0.22 | 0 | 31269 | 2 |
| lenmtp-2 | `SPEC_METHOD=mtp SPEC_K=5` | 1.56 | 32 | 83.5 | 22.9 | 3.135 | 71.6 | 0.75/0.53/0.38/0.27/0.20 | 0 | 31269 | 2 |
| lenmtp2-1 | `SPEC_METHOD=mtp SPEC_K=7 MAX_MODEL_LEN=8192` | 1.37 | 1 | 54.5 | 17.9 | 3.29 | 58.8 | 0.73/0.53/0.38/0.26/0.19/0.12/0.08 | 0 | 30917 | 1 |
| lenmtp2-1 | `SPEC_METHOD=mtp SPEC_K=7 MAX_MODEL_LEN=8192` | 1.37 | 8 | 60.7 | 15.7 | 3.382 | 53.2 | 0.74/0.53/0.38/0.28/0.20/0.14/0.11 | 0 | 31155 | 1 |
| lenmtp2-2 | `SPEC_METHOD=mtp SPEC_K=7 SPEC_CONF=0.5 MAX_MODEL_LEN=8192` | 1.22 | 1 | 63.0 | 15.3 | 3.006 | 46.1 | 0.74/0.45/0.29/0.21/0.14/0.10/0.07 | 0 | 31735 | 1 |
| lenmtp2-2 | `SPEC_METHOD=mtp SPEC_K=7 SPEC_CONF=0.5 MAX_MODEL_LEN=8192` | 1.22 | 8 | 64.0 | 14.8 | 3.05 | 45.2 | 0.72/0.45/0.30/0.21/0.16/0.12/0.09 | 0 | 32037 | 1 |
| lenmtp2-3 | `SPEC_METHOD=mtp SPEC_K=15 MAX_MODEL_LEN=8192 EDGE0_EXPERT_VRAM_GIB=11.0 EDGE0_PLACEMENT=plan:/workspace/edge0/plans/plan-s1-11.0.json` | 3.07 | 1 | 34.2 | 28.6 | 3.247 | 93.0 | 0.72/0.50/0.34/0.24/0.16/0.11/0.06/0.04/0.03/0.02/0.01/0.01/0.00/0.00/0.00 | 0 | 30875 | 1 |
| lenmtp2-3 | `SPEC_METHOD=mtp SPEC_K=15 MAX_MODEL_LEN=8192 EDGE0_EXPERT_VRAM_GIB=11.0 EDGE0_PLACEMENT=plan:/workspace/edge0/plans/plan-s1-11.0.json` | 3.07 | 8 | 36.4 | 26.7 | 3.477 | 92.7 | 0.72/0.51/0.35/0.24/0.18/0.13/0.10/0.07/0.06/0.03/0.03/0.02/0.01/0.01/0.01 | 0 | 31177 | 1 |
| lenschcpu-1 | `SPEC_METHOD=dflash2 SPEC_K=3 EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55` | 2.46 | 1 | 81.0 | 11.9 | 2.681 | 31.9 | 0.77/0.55/0.36 | 0 | 30817 | 1 |
| lenschcpu-2 | `SPEC_METHOD=dflash2 SPEC_K=5 EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55` | 1.87 | 1 | 75.6 | 12.8 | 3.019 | 38.6 | 0.74/0.54/0.35/0.24/0.15 | 0 | 30855 | 1 |
| lenschcpu-3 | `SPEC_METHOD=dflash2 SPEC_K=7 MAX_MODEL_LEN=8192 EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55` | 1.56 | 1 | 74.7 | 13.0 | 3.312 | 43.0 | 0.76/0.54/0.38/0.26/0.18/0.12/0.07 | 0 | 30829 | 1 |
| lensch-1 | `SPEC_METHOD=dflash2 SPEC_K=3 SPEC_SCHED=[[1,2,3],[3,32,2]]` | 2.47 | 1 | 70.6 | 13.7 | 2.67 | 36.6 | 0.76/0.55/0.36 | 0 | 31053 | 6 |
| lensch-1 | `SPEC_METHOD=dflash2 SPEC_K=3 SPEC_SCHED=[[1,2,3],[3,32,2]]` | 2.47 | 8 | 103.6 | 51.1 | 2.39 | 122.2 | 0.77/0.57/0.05 | 0 | 31183 | 6 |
| lensch-1 | `SPEC_METHOD=dflash2 SPEC_K=3 SPEC_SCHED=[[1,2,3],[3,32,2]]` | 2.47 | 32 | 104.2 | 54.4 | 2.357 | 128.1 | 0.77/0.56/0.03 | 0 | 31311 | 6 |
| lensch-2 | `SPEC_METHOD=dflash2 SPEC_K=3 SPEC_SCHED=[[1,2,3],[3,32,2]] EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55` | 2.46 | 1 | 79.5 | 12.1 | 2.659 | 32.3 | 0.75/0.54/0.37 | 0 | 31067 | 6 |
| lensch-2 | `SPEC_METHOD=dflash2 SPEC_K=3 SPEC_SCHED=[[1,2,3],[3,32,2]] EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55` | 2.46 | 8 | 116.5 | 45.2 | 2.393 | 108.2 | 0.77/0.57/0.05 | 0 | 31197 | 6 |
| lensch-2 | `SPEC_METHOD=dflash2 SPEC_K=3 SPEC_SCHED=[[1,2,3],[3,32,2]] EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55` | 2.46 | 32 | 121.5 | 46.4 | 2.344 | 108.7 | 0.76/0.55/0.03 | 0 | 31197 | 6 |
| lensch-3 | `SPEC_METHOD=dflash2 SPEC_K=3 SPEC_SCHED=[[1,2,3],[3,8,2],[9,32,0]] EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55 EDGE0_EXPERT_VRAM_GIB=11.0 EDGE0_PLACEMENT=plan:/workspace/edge0/plans/plan-s1-11.0.json` | 4.31 | 1 | 75.3 | 12.8 | 2.716 | 34.8 | 0.77/0.56/0.39 | 0 | 31147 | 11 |
| lensch-3 | `SPEC_METHOD=dflash2 SPEC_K=3 SPEC_SCHED=[[1,2,3],[3,8,2],[9,32,0]] EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55 EDGE0_EXPERT_VRAM_GIB=11.0 EDGE0_PLACEMENT=plan:/workspace/edge0/plans/plan-s1-11.0.json` | 4.31 | 8 | 132.8 | 54.5 | 2.367 | 129.0 | 0.76/0.56/0.04 | 0 | 31369 | 11 |
| lensch-3 | `SPEC_METHOD=dflash2 SPEC_K=3 SPEC_SCHED=[[1,2,3],[3,8,2],[9,32,0]] EDGE0_CPU_EXPERTS=1 EDGE0_CPU_THREADS=8 EDGE0_CPU_FRAC=0.55 EDGE0_EXPERT_VRAM_GIB=11.0 EDGE0_PLACEMENT=plan:/workspace/edge0/plans/plan-s1-11.0.json` | 4.31 | 32 | 119.3 | 87.2 | 2.688 | 234.4 | 0.76/0.57/0.36 | 1 | 31499 | 11 |
