# Edge0-145B-NVFP4: model spec as Strata computes it

Status: written 2026-11-19 by the `strata-spec` agent. Sources were read from the 5090 box:

- `S:` = `/home/andychen/work/edge0/edge0-strata/` (the Strata engine, MIT)
- `L:` = `S:third_party/llama.cpp/` (pinned llama.cpp with the `qwen4exp` graph and converter)
- `V:` = `/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/` inside `nvcr.io/nvidia/vllm:26.11-py3` (vLLM `0.33.0+121396e7.dev`)
- `M:` = `/home/andychen/models/Edge0-145B-NVFP4/`

Every format below was checked against the real files with `edge0/convert/strata_formats.py check --moe`, reading small slices only (max RSS 614 MB, about 1 s). It printed `ALL CHECKS PASSED`; the results are in section 10. The full skeleton tensor dump is in `edge0/docs/skeleton-tensors.txt`.

Confidence tags: **[src]** read in the Strata or llama.cpp source; **[file]** measured on the model files; **[vllm]** read in the vLLM image; **[inferred]** derived from the above but not tested end to end; **[unconfirmed]** open question.

> **Most important finding for the port.** vLLM 0.33 in the NGC image already ships the base architecture as `Qwen4ExpForCausalLM` / `Qwen4ExpMTP` (`vllm.models.qwen4_exp`; registry entries `Qwen4ExpForCausalLM`, `Qwen4ExpForConditionalGeneration` and `Qwen4ExpMTP`). It covers the hyper-connections, GDN, QSA sparse attention with the indexer, the PLE n-gram embedding with FP8/NVFP4 tables and its conv state, and the MTP. transformers 5.20.2 in the image has `Qwen4ExpConfig`. Edge0 differs from Qwen3.8-Flash-Next only in the routed-expert block, which is a latent MoE (section 2.6). So the port is: reuse `qwen4_exp`, replace `Qwen4ExpSparseMoeBlock`'s routed experts with a latent-MoE block, and convert the weights from the GGUF/Strata files into the names and conventions vLLM expects (section 7).

---

## 1. Geometry at a glance

| Item | Value | Source |
|---|---|---|
| Architecture | `qwen4exp` (Qwen3.8-Flash-Next), latent MoE | GGUF `general.architecture` **[file]** |
| Layers | 48. Layer `l` is QSA (full attention with a sparse indexer) when `l % 4 == 3` (layers 3, 7, …, 47, 12 layers); otherwise GDN (gated delta-net linear attention, 36 layers) | `S:include/strata/core/geometry.hpp` `is_qsa_layer`; GGUF `full_attention_interval=4`, `recurrent_layers`, `compress_ratios` **[src][file]** |
| Hidden `H` | 2560 | |
| Residual | 4 hyper-connection streams (`hc=4`, `hc_dim=10240`), low rank `hc_lr=320`. There are no plain layer norms and no `output_norm`. | |
| Vocabulary | 248,320 (Q8_0 `token_embd`, separate Q8_0 `output`, not tied) | |
| RMSNorm eps | 1e-6 everywhere (`attention.layer_norm_rms_epsilon`) | |
| GDN | `S=128` state/head width, 16 q/k heads, 48 v heads, conv kernel 4, 10240 conv channels (`q 2048 | k 2048 | v 6144`), value dim 6144 | |
| QSA attention | 24 q heads, 2 kv heads, head_dim 256, output gate, partial NEOX RoPE on 64 of 256 dims, theta 1e7 | |
| QSA indexer | 4 query heads × 128, 1 key head × 128, block (compress ratio) 4, budget 2048 cells (`top_k`), selection width `min(n_kv, 2051)` | |
| MoE | 512 routed experts, top-10, softmax → top-k → renormalize, scale 1, no bias; 1 shared expert (2560→640→2560) with a sigmoid scalar gate | |
| Latent MoE | 8 routing groups; `D_g`: 2560→1792, `U_g`: 1792→2560 per group per layer; expert `e` uses group `labels[l][e]`; expert = SwiGLU 1792→640→1792, NVFP4 | |
| PLE | layer 1 only; 16 hashed n-gram rows × 160 = 2560; FP8 table of 320,001,536 rows (51.2 GB) | |
| MTP | 1 draft layer: full-width Q2_0 experts (512 × 2560→640), QSA-shaped attention (Strata runs it dense), hyper-connections | |
| Context | 262,144 trained (rope.scaling none) | |
| Tokenizer | GPT-2 byte BPE, pre-tokenizer `qwen35`, `add_bos=false`; EOS 248046 `<|im_end|>`, also stops on 248044 `<|endoftext|>` | |

---

## 2. Forward pass (one token, decode)

The reference is Strata's `block_layer_pre` / `block_layer_post` (`S:src/core/layer.cpp`), with the semantics in `S:include/strata/core/layer.hpp`, transcribed from `L:src/models/qwen4exp.cpp`. Every matrix below uses `(out, in)` layout, so `y = W @ x`. A GGUF tensor `[ne0, ne1]` stores `ne1` rows of `ne0` contiguous inputs, so `W` has shape `(ne1, ne0)`.

### 2.0 Pseudo-code

```
R = repeat(embed(tok), 4)                       # (4, 2560): every stream = the token's embedding row; no scale
for l in 0..47:
    if l == 1: R = PLE(R, tok, prev2, hist)      # residual update, BEFORE layer 1's first hc read (2.7)
    x, inj = hc_read(R, hc_attn_*[l])            # (2560,), (4,)
    y = QSA(l, x) if l % 4 == 3 else GDN(l, x)   # (2560,)
    R = hc_write(R, y, inj)
    x, inj = hc_read(R, hc_ffn_*[l])
    y = LatentMoE(l, x) + SharedExpert(l, x)
    R = hc_write(R, y, inj)
h = hc_read(R, output_hc_*, inject=None)         # final mixer = the output norm
logits = output.weight @ h                       # (248320,); no softcap, no scale
```

### 2.1 Hyper-connection read and write (gated residual) **[src][vllm]**

`S:src/kernels/cuda/gr.cu` (`gr_read`, `gr_write`), `L:qwen4exp.cpp` `build_hc_mix` / `build_hc_combine`, and `V:common/hyperconnection.py` `GatedResidual` all agree.

```
hc_read(R, norm, down, up, inject):             # R (4, 2560) flattened stream-major: R_flat[c*2560 + d]
    xn[c,:] = R[c,:] * rsqrt(mean(R[c,:]^2) + 1e-6) * norm[c*2560:(c+1)*2560]   # per-stream RMS, (10240,) gamma
    lo      = silu( (down @ xn_flat) / 4 )                     # down: (320, 10240); the /hc is INSIDE the silu
    gate    = sigmoid( up @ lo )                               # up: (10240, 320)
    mixed[d]= mean_c( xn[c,d] * gate[c*2560 + d] )             # (2560,) -> the mixer/MoE input
    inj[c]  = inject[c,:] @ xn_flat                            # inject: (4, 10240); absent for the final mixer
hc_write(R, y, inj):
    R[c,:] += y * 2*sigmoid(inj[c] / 4)                        # the same y goes into every stream
```

- The GGUF `hc_*_norm` and `output_hc_norm` are **already `1 + w`** (the llama.cpp converter folds the GemmaRMSNorm offset). Use them as plain gammas. In `[file]` they range from −4.9 to 7.6 with mean 0.94 (`blk.0.hc_attn_norm`). vLLM's `GroupedGemmaRMSNorm` multiplies by `1 + weight`, so the converter must subtract 1 (section 7).
- `hc_*_down` is GGUF `[10240, 320]`, so `W` is `(320, 10240)`. `hc_*_up` is `[320, 10240]`, so `W` is `(10240, 320)`. `hc_*_inject` is `[10240, 4]`, so `W` is `(4, 10240)`. All are BF16.
- Strata's default activation precision rounds `xn` and `lo` to BF16 before the two projections. Its native path (`gr_set_native_mmvf`) uses FP32 activations. Reductions are FP32.

### 2.2 Embedding and output

- Embedding: row `tok` of `token_embd.weight` (Q8_0 `[2560, 248320]`). There is no multiplier. The residual starts as 4 identical copies (`S:src/core/session.cpp` `session_zero`; `V:nvidia/model.py` `hidden_states.repeat(1, hc_count)`). **[src][vllm]**
- Output: `hc_read` with `output_hc_{norm,down,up}` and no inject, then `logits = output.weight @ mixed`. Strata quantizes `mixed` to Q8_K for that GEMV. **[src]**

### 2.3 GDN layers (36 layers, `l % 4 != 3`) **[src]**

`S:src/core/layer.cpp` `gdn_layer`, `S:include/strata/kernels/gdn.hpp`, `L:qwen4exp.cpp` `build_layer_attn_linear`. The input is `x` = `mixed` from `hc_read(attn)`.

```
qkv  = attn_qkv @ x                        # (10240,) = q[0:2048] | k[2048:4096] | v[4096:10240], 128-wide heads
z    = attn_gate @ x                       # (6144,)  = 48 heads x 128
conv = causal depthwise conv1d over time, kernel 4, on all 10240 channels, carried state of the last 3 inputs:
       conv[c] = sum_{i<4} w[c][i] * inp[t-3+i][c]        # w = ssm_conv1d [4,10240] -> W[c][i] (tap 0 = oldest)
h    = silu(conv)                          # SiLU on the WHOLE 10240 before the split
q,k  = h[0:2048], h[2048:4096] viewed (16,128); q,k = l2norm(q), l2norm(k)  # x / sqrt(sum x^2 + 1e-6), per head
q   *= 1/sqrt(128)
v    = h[4096:] viewed (48,128)            # v is NOT normalized
beta = sigmoid(ssm_beta @ x)               # (48,)  ssm_beta BF16 [2560,48]
g    = softplus(ssm_alpha @ x + ssm_dt.bias) * ssm_a      # (48,); ssm_a = -exp(A_log) < 0 (file: -158 .. -0.028)
for v-head h (k/q head kh = h % 16  <-- TILED pairing in GGUF order, see section 7):
    S_h  = S_h * exp(g[h])                                 # S_h (128 x 128), decay before the update
    d    = (v[h] - S_h^T k[kh]) * beta[h]
    S_h += k[kh] outer d
    o[h] = S_h^T q[kh]
y[h] = rmsnorm(o[h]) (per head over 128, eps 1e-6) * ssm_norm * sigmoid(z[h])   # SIGMOID, not SiLU (Qwen3.5 used SiLU)
out  = ssm_out @ y_flat                    # (2560,)
```

- `ssm_norm` is **not** `+1` folded. It is the plain gamma (mean 0.967 in `[file]`).
- Softplus is `log1p(exp(x))`, with the branch `x > 20 → x`.
- Recurrent state per sequence and layer: `48 × 128 × 128` fp32 (3 MiB). Conv state: `3 × 10240`. Both start at zero.

### 2.4 QSA layers (12 layers: 3, 7, …, 47) **[src]**

`S:src/core/layer.cpp` `qsa_layer`, `S:include/strata/kernels/qsa.hpp`, `qsa_select.hpp`, `L:qwen4exp.cpp` `build_qsa_top_k`, `build_attn_qsa`, `build_layer_attn`.

```
qg    = attn_q @ x                 # (12288,) = 24 heads x [q(256) | gate(256)]  -- halves per head, not interleaved
q     = qg[:, :256] ; gate = qg[:, 256:]
k     = attn_k @ x (2,256) ; v = attn_v @ x (2,256)
q     = rope(rmsnorm_head(q) * attn_q_norm, pos)     # per head over 256; attn_q_norm/k_norm are 1+w folded (mean ~1.28)
k     = rope(rmsnorm_head(k) * attn_k_norm, pos)
KV cache append (k after norm+rope, v raw)
# ---- indexer
kraw  = indexer.k_proj @ x         # (128,) BF16; cached RAW (never normed, never roped)
qi    = rope(rmsnorm_head(indexer.q_proj @ x) * indexer.q_norm, pos)        # (4,128)
for each COMPLETE block b of 4 cells (cells 4b..4b+3):
    kb = rope( rmsnorm(mean(kraw[4b..4b+3])) * indexer.k_norm , pos = 4b )   # rotated at the block's FIRST cell
    score_b = sum_{h<4} relu( qi[h] . kb )        # relu per head, then sum; no softmax, no 1/sqrt(d), no weights
cell score = score of its block; cells of the incomplete tail block (j >= 4*floor(n_kv/4)) get +1e9 (always selected)
selection = top W cells, W = min(n_kv, 2048 + 4 - 1 = 2051), ties -> lower index (so W >= n_kv means dense causal)
# ---- attention over the selected cells only
attn[h] = softmax_j( q[h].k[j, h // 12] / sqrt(256) ) . v[j, h // 12]   # GQA: q head h -> kv head h // 12
o     = attn * sigmoid(gate)       # per element, per head
out   = attn_output @ o_flat       # (6144 -> 2560)
```

- **RoPE**: NEOX pairing `(i, i+32)` on the first 64 dims of each 256-dim q/k head and of each 128-dim indexer head. `angle_i = pos · 1e7^(−2i/64)` for `i < 32`. The GGUF declares interleaved M-RoPE (`rope.dimension_sections [11,11,10,0]`, llama.cpp IMROPE). For text, t = h = w = pos, so it reduces to plain NEOX RoPE (`S:include/strata/kernels/mrope.hpp`). Positions are cell indices from 0.
- **Selection details** (`qsa_select.hpp`): the selection is over cells, and a block's 4 cells share one score. When more than 512 complete blocks exist and the tail holds `t < 3` cells, the 2051-cell budget takes 512 whole blocks plus `3 − t` cells of the next-best block (its lowest-index cells). **[src]** vLLM's `V:nvidia/qsa.py` / `ops/qsa_indexer.py` may select whole blocks plus the tail instead. That only matters past 2051 cells. **[unconfirmed]**
- Prefill uses the same rule per query position. Query `i` scores the complete blocks whose last cell is `<= i`, and its own incomplete block is forced in (`qsa.hpp` notes; llama.cpp `set_input_qsa` per-block bias).
- Strata stores KV as INT8 with one fp16 scale per 64 values (`--kv int8`) and keeps 32,768 cells resident (`--kv-resident 32768`), streaming the rest from pinned host RAM. Pooled indexer keys are FP32.
- Per-token KV in BF16 is `12 layers × 2 kv-heads × 256 × 2 (K,V) × 2 B = 24 KiB`.

### 2.5 Shared expert (every layer) **[src]**

`S:include/strata/kernels/shared_expert.hpp`:

```
sh = ffn_down_shexp @ ( silu(ffn_gate_shexp @ x) * (ffn_up_shexp @ x) )    # 2560 -> 640 -> 2560, Q8_0
sh *= sigmoid( ffn_gate_inp_shexp . x )                                      # BF16 [2560,1] -> one scalar per token
```

The result is **added** to the routed output, with no weighting against it.

### 2.6 Latent MoE (every layer); the Edge0-specific part **[src][file]**

`S:src/latent/latent_moe.cpp`, `S:src/kernels/latent_cpu_verify.cpp` (the clearest statement), `S:src/core/verify.cpp` (decode), `S:src/prefill/prefill.cpp` (prompt).

```
logits = ffn_gate_inp @ x                    # BF16 (512, 2560); router input is x itself (NOT D x)
p      = softmax(logits)                     # over all 512
ids    = top-10 by p (stable, ties -> lower index); w = p[ids] / max(sum p[ids], 2^-14)   # scale 1, no bias
for g in 0..7: xl[g] = D[l][g] @ x           # (1792,), D[l][g]: (1792, 2560). Strata decode computes ALL 8 groups.
for j in 0..9:
    e  = ids[j]; g = labels[l][e]            # labels.bin int32 [48][512], values 0..7
    hj = silu(Wg[e] @ xl[g]) * (Wu[e] @ xl[g])       # Wg, Wu: (640, 1792) NVFP4 * s_gate / s_up
    yj = Wd[e] @ hj                                    # Wd: (1792, 640) NVFP4 * s_down
    out += w[j] * (U[l][g] @ yj)                       # U[l][g]: (2560, 1792)
out += shared expert (2.5)
```

- D and U are per layer and per group, shared by every expert of that group. They are not per expert. Groups are very uneven: layer 0 has `[25, 165, 33, 53, 25, 46, 67, 98]` experts per group, and over all layers the per-group count ranges from 8 to 200. **[file]**
- Because U is linear, `out_routed = U_cat @ concat_g(Σ_{j: g_j = g} w_j y_j)` with `U_cat = [U_0 | … | U_7]` of shape (2560, 14336). Likewise `xl_all = D_cat @ x` with `D_cat` of shape (14336, 2560). So D and U are two dense GEMMs per token (14336×2560 each). The experts are a grouped GEMM whose input row for (token t, expert e) is `xl_all[t, labels[e]]`, which is Strata's `tok = t*G + label[e]` plan (`build_plan`). **[inferred, exact algebra]** Strata applies `w_j` after U and sums the parts. The order differs only in rounding.
- There are no Q3_K column scales (`scales/l*.f32`) and no Hadamard folding (`hadamard.txt`) in this folder, so neither applies to the NVFP4 pack. **[file]**
- Strata's activation precision: in decode, x and xl are quantized to q8_1 for the D GEMV and the expert gate/up, h to q8_1 for down, and yl to q8_1 for U (W4A8 / W8A8 integer dot). The CPU-computed misses use the AVX-512 NVFP4 kernels with Q8 activations.

### 2.7 PLE (layer 1 only) **[src][vllm]**

`S:include/strata/kernels/ple.hpp`, `S:src/kernels/cuda/ple.cu` `ple_block`, `S:src/core/layer.cpp` `block_layer_pre`, `L:qwen4exp.cpp` `build_ple`, `V:nvidia/ple_layer.py`. GGUF `qwen4exp.ple.layers = [1]` is 0-based. The HF config's `ple_layer_ids` is 1-based (`[2]`), and vLLM attaches the PLE to `layer_idx` with `layer_idx + 1 in ple_layer_ids`. Both mean "before layer index 1".

```
emb   = concat_{h<16} PLE_TABLE[row_h]            # (2560,), head-slowest: row h occupies [160h, 160h+160)
key   = gnorm(ple_key @ emb, ple_norm_key)        # ple_key BF16 [2560,10240] -> (10240, 2560); gnorm = per-stream RMS over 2560
query = gnorm(R_flat, ple_norm_query)             # R = the residual stack after layer 0
value = ple_value @ emb                           # BF16 (2560, 2560)
s[c]  = dot(key[c], query[c]) / sqrt(2560)        # per stream c < 4
gate[c] = sigmoid( sign(s) * sqrt(max(|s|, 1e-6)) )
gated[c,:] = value * gate[c]                      # (4, 2560)
normed = gnorm(gated, ple_norm_conv)              # THIS is the conv input and the history row
conv[c] = sum_{k<4} W[c][k] * normed_{t - 3*(3-k)}[c]   # dilation 3: taps at t-9, t-6, t-3, t; W = ple_conv1d F16 [4,10240]
R_flat = R_flat + gated_flat + silu(conv)
history <- append(normed)   (keeps the last 9 rows; zeros at sequence start)
```

- The `ple_norm_*` gammas are `1 + w` folded (mean ~0.89). **[file][src]**
- The hash, the table layout and the FP8 scaling are in section 3.

### 2.8 Constants

| Constant | Value |
|---|---|
| rms eps | 1e-6 (all RMSNorms, the l2norm floor, and PLE gnorm) |
| hc divisor | `/4` inside the down-silu, and `inject/4` inside the write gate `2·sigmoid` |
| router | softmax over 512, top-10, renormalize with sum clamped ≥ 2^-14, routed scale 1.0 |
| QSA | scale 1/16, kv map `h // 12`, selection width 2051, block 4, tail bias +1e9, rope theta 1e7, 64 rotated dims |
| GDN | q scale 1/sqrt(128); state decay `exp(softplus(α+dt)·ssm_a)` |
| PLE | gate `sigmoid(sign(s)·sqrt(max(|s|,1e-6)))`, `s` scaled by 1/sqrt(2560); conv kernel 4, dilation 3 (= ngram_size), history 9 |
| sampling defaults (GGUF) | top_k 20, top_p 0.95, temp 1.0 |

### 2.9 Prefill vs decode in Strata **[src]**

The math is the same per position. What differs is batching and numerics:

- Prompts run in chunks of up to 8192 tokens (`--prefill auto` picks the largest chunk the expert-cache loan allows). GDN uses a chunked delta rule (`S:src/prefill/gdn_chunk.cu`). The PLE conv history and the QSA indexer state carry across chunks.
- The prefill MoE runs the routed experts on the GPU. Records that are not VRAM-resident stream over PCIe into a byte-sized ring **while the previous layer's attention runs** (`S:docs/HOW_IT_WORKS.md` "Reading long texts", `S:docs/DETAILS.md` "How it works"), which is the overlap TidalMoE also uses. The latent path is fused: D, then grouped NVFP4 A (`STRATA_LATENT_FUSED=1`), then U (`STRATA_LATENT_FUSED_DU=1`).
- Decode runs in speculative verify windows of `T ≤ 4` rows (`--spec 4`). VRAM-resident experts run on the GPU. Misses split between in-place CPU compute (AVX-512 over pinned RAM, 24 pool workers) and a DMA share computed on the GPU (`--pcie-frac 0.1 --pcie-mode dma`). With `STRATA_LATENT_U_GPU=1`, the CPU's latent rows are U-projected on the GPU.
- Because of these Q8 activation paths, a BF16 vLLM implementation will not match Strata token for token over long greedy runs. Judge correctness by teacher-forced top-1 agreement and logit KL against `strata-ref.json`, not by exact strings.

---

## 3. PLE in detail

### 3.1 Row ids from token ids (`S:src/kernels/ngram.cpp` `ngram_rows`) **[src]**

```
consts (GGUF qwen4exp.ple.*):
  mult  = [23703573157769, 20109073645365, 8052911324071]          # uint64
  vocab = [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
           20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]
  off   = running sum of vocab: [0, 20000003, 40000026, ..., 300001275]   (last head ends at 320,001,446)
  EOS   = 248044 (<|endoftext|>)
ctx[0] = token t_i
for s in 1, 2:                                   # walk back through the two predecessors
    p   = cut ? -1 : t_{i-s}                     # -1 when there is no predecessor (sequence start)
    cut = cut or p < 0 or p == EOS               # the cut is checked BEFORE storing
    ctx[s] = EOS if cut else p
for n in 2, 3:                                    # bigram heads 0..7, trigram heads 8..15
    mixed = XOR_{j<n} (ctx[j] * mult[j] mod 2^64)    # XOR, not sum; uint64 wraparound
    row[(n-2)*8 + g] = mixed % vocab[h] + off[h]     # h = (n-2)*8 + g, g < 8;  % not a mask
```

The token's own EOS does not cut its context; only predecessors are checked. Token id 0 is a real token. For images, the GGUF carries `ple.image_token_id 248056`. Strata's image path is not covered here.

**vLLM derives the same constants from the config.** This was verified by running vLLM's code on its own. `V:nvidia/ngram_embedding.py` `Qwen4ExpNGramEmbedding._make_layer_multipliers` (splitmix64 with seed 1234, vocab 248320, PLE layer 0) gives exactly `[23703573157769, 20109073645365, 8052911324071]`. `_make_vocab_layout` (the first 16 primes above `ngram_vocab_size_base=20,000,000`) gives exactly the 16 vocab sizes above. Its `compute_ngram_ids` applies the same EOS-cut rule as Strata, and products stay below 2^63 because `mult < (2^63−1)/vocab`, so int64 and uint64 agree. vLLM reads the reset id as `int(config.eos_token_id)`, **so the HF text config must carry `eos_token_id = 248044`**, not `<|im_end|>` and not a list. Put the stop ids in `generation_config.json` instead. **[vllm][file]**

### 3.2 Table layout and dequantization (`M:ple-fp8.gguf`) **[file][src]**

- The file is GGUF v3 with 5 KVs: `general.architecture='strata-ple'`, `strata.ple.format='f8_e4m3'`, `strata.ple.scale=0.00019931793212890625`, `strata.ple.source='nvfp4-src'`. It holds one tensor, `per_layer_token_embd.weight`, of GGUF type I8 (GGUF has no FP8 type), shape `[160, 320001536]`, at data offset 352. The tensor fills the rest of the file exactly (51,200,245,760 B).
- Row `r` is bytes `[352 + 160r, 352 + 160r + 160)`. `value = e4m3fn(byte) * scale`. e4m3fn: sign bit 7, 4 exponent bits with bias 7, 3 mantissa bits, subnormal `m/8 · 2^-6`, and 0x7F/0xFF (NaN) read as 0. The rows are padded to 320,001,536, but only rows below 320,001,446 are reachable.
- Measured rows have |v| < 0.05 and rms ~0.01. **[file]**
- vLLM's `Qwen4ExpPLEFp8EmbeddingMethod` (`V:common/ngram_embedding.py`) takes an FP8 table plus one global `weight_scale`. Strata's file is that table, byte for byte (`S:tools/ple_fp8_pack.py`). Hardlink or memory-map it; never copy it. vLLM also has a pinned-host lookup (`Qwen4ExpPLEPinnedHostEmbedding`) and a pageable path (`V:nvidia/ple_pageable.py`).
- Strata reads 16 rows per token (2,560 B) straight from the SSD with `--ple-io direct` (O_DIRECT 4 KiB reads, a 1M-row LRU, prefetch issued as soon as the token id is known). The table never enters RAM.

---

## 4. MTP draft layer and Strata speculation

### 4.1 Structure (`S:include/strata/core/mtp.hpp`, `S:src/core/mtp.cpp` `record_front`/`record_rest`; `V:nvidia/mtp.py`) **[src][vllm]**

```
inputs at cell i: R_i = main model's final multi-stream residual (4, 2560) BEFORE the final mixer, token t_{i+1}
e  = fc_embedding @ ( rmsnorm(embed(t_{i+1})) * pre_fc_norm_embedding )      # Q8_0 (2560,2560)
hn = rmsnorm over ALL 10240 of R_i * pre_fc_norm_hidden                       # one RMS over all 4 streams (default)
h[c] = fc_hidden @ hn[c]   for each stream c (shared weights)
R  = h + e (broadcast to every stream)
x, inj = hc_read(R, attn_hyper_connection.*); y = attention(x); R = hc_write(R, y, inj)
x, inj = hc_read(R, mlp_hyper_connection.*);  y = MoE(x) + shared(x); R = hc_write(R, y, inj)
draft  = argmax over draft_vocab subset of output.weight @ hc_read(R, hyper_connection_mixer.*)
next step feeds this R as R_{i+1} with the drafted token
```

- The attention has the main QSA shape (q 12288 with gate, 2 kv heads, q/k norms, NEOX rope at position i). **Strata runs it dense** over every cell it has seen and ignores `self_attn.indexer.*`. vLLM's `Qwen4ExpMTP` does use the QSA indexer.
- `--mtp-hnorm stream` switches to llama.cpp's per-stream norm. vLLM's `GemmaRMSNorm(hidden*hc)` is the global one, the same as Strata's default.
- The MTP MoE is **not latent**. It has 512 full-width experts (2560→640→2560), router `mlp.gate.weight` BF16 (512, 2560), and the same softmax/top-10/renorm routing. The shared expert is `mlp.shared_expert.{gate,up,down}_proj` (Q8_0) with gate `mlp.shared_expert_gate.weight` (BF16 1×2560).
- The head is the main `output.weight` restricted to the rows in `mtp/draft_vocab.bin` (106,299 ids, unsorted).

### 4.2 Speculation in Strata (`M:model.json`: `--spec 4 --spec-min-p 0.5 --mtp ./mtp`) **[src]**

- The verify window is T = 4: 1 committed token plus up to 3 MTP drafts. The draft chain stops early when a draft's probability falls below 0.5. One batched pass over all 48 layers verifies the window, and the accepted prefix is committed. Docs report 2.4 to 3.2 tokens per pass.
- A suffix/prompt-lookup drafter can add up to 5 tokens when the reply copies the context.
- Greedy output equals non-speculative greedy up to numerics (`S:docs/DETAILS.md` "Draft layer").

---

## 5. File formats

### 5.1 `experts.bin`: routed latent experts (44.30 GiB) **[file][src]**

- The file is `48 × 512` records of **1,935,376 B**. Record (l, e) is at `native_experts[l].offset + e × 1,935,376`, and `offset_l = l × 990,912,512`. There is no header, and the file size is exactly 47,563,800,576.
- The model card's "1,630,720 B per expert" is the **Q3_K** build's record (Q3_K gate/up plus IQ4_NL down). It is not this NVFP4 file.
- Record layout (`S:src/kernels/cuda/iq_kernels.cu` `native_expert_layout(40, 40, 1792, 640)`):

| Bytes | Content |
|---|---|
| `[0, 645,120)` | gate: 640 rows × 1792 inputs, NVFP4, 1008 B/row (28 blocks) |
| `[645,120, 1,290,240)` | up: same layout |
| `[1,290,240, 1,935,360)` | down: 1792 rows × 640 inputs, NVFP4, 360 B/row (10 blocks) |
| `[1,935,360, 1,935,376)` | tail: 4 × f32 `{s_gate, s_up, s_down, 0}` = ModelOpt `weight_scale_2` per projection |

- **NVFP4 block** (ggml `block_nvfp4`, type 40, 64 values in 36 B):
  - `d[4]` is one UE4M3 scale per 16 values, followed by `qs[32]`.
  - Sub-block `s` covers values 16s to 16s+15. Its codes are in `qs[8s .. 8s+7]`: byte j holds value j in the **low** nibble and value **j+8** in the **high** nibble.
  - `value = E2M1[code] × ue4m3(d[s]) × s_proj`, with `E2M1 = {0, .5, 1, 1.5, 2, 3, 4, 6}` and sign bit 3.
  - `ue4m3` uses exponent bits 6..3 (bias 7), mantissa bits 2..0, and subnormal `m · 2^-9`; 0x00/0x7F read as 0 and bit 7 is ignored.
  - ggml stores doubled E2M1 values and a halved scale (`ggml_ue4m3_to_fp32`), which is numerically the same (`L:ggml/src/ggml-quants.c` `dequantize_row_nvfp4`).
  - This nibble order differs from ModelOpt/vLLM, which pack value 2i low and 2i+1 high. Section 7 has the lossless repack.
- Checks on real data:
  - Every 16-value sub-block holds a code of magnitude 6, which confirms the scale grouping.
  - `s_gate == s_up` for all 512 experts of layers 0, 23 and 47.
  - `s_gate` ranges 5.6e-4 to 1.2e-2 and `s_down` 4.5e-5 to 5.0e-4.
  - Decoded weights: gate/up rms ~0.08 to 0.11 and down rms ~0.025 to 0.029, with no NaN.
- `native_experts.txt` starts with `# strata native experts v6: latent A-only (expert_embd 1792, expert_ff 640)`. Each line is `layer gu_type d_type offset blob_bytes`, and every layer is `40 40 <l×990912512> 1935376`.

### 5.2 `labels.bin` **[file][src]**

int32 `[48][512]` (98,304 B). It holds the D/U group (0..7) of each routed expert (`S:src/core/verify.cpp`, labels loaded beside the A pack).

### 5.3 `latent_dense_q8.bin` / `latent_dense_q4.bin`: D and U **[file][src]**

`S:src/core/latent_dense.cpp`. The env var `STRATA_LATENT_DENSE_FMT=8|4` selects the format, and the file size is cross-checked. There is no header. Layers are back to back:

```
layer l (fmt 8: 77,987,840 B; fmt 4: 41,287,680 B):
  D: 8*1792 = 14,336 rows x 2560 in   -> group g = rows [1792g, 1792g+1792)
  U: 8*2560 = 20,480 rows x 1792 in   -> group g = rows [2560g, 2560g+2560)
rows: fmt 8 = Q8_0 (34 B / 32 values: fp16 d, 32 x int8; v = d*q), D row 2720 B, U row 1904 B
      fmt 4 = Q4_0 (18 B / 32: fp16 d, 16 B; value j low nibble of qs[j], j+16 high nibble; v = (code-8)*d)
```

`model.json` uses Q8 (`latent_dense_q8.bin`, 3,743,416,320 B = 3.49 GiB). Q4 is 1.85 GiB. Measured: D rms ~0.011 (|v| < 0.1) and U rms ~0.0099 (|v| < 0.19).

### 5.4 `index.txt` + `dense.bin`: strata pack index v3 **[file][src]**

- The header is `# strata pack index v3 -- generated by tools/iq_pack.py (native experts) from model.gguf` followed by `# align 64 pool 1538625024 tensors 1367`.
- Each row has 19 space-separated fields: `name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits code_bias group_elems codebook has_offset codes_bytes scales_bytes offset_bytes scales_fp16 act_kind` (`S:src/core/weights.cpp` `WeightTable::load`).
- `file 0` is `dense.bin`. `kind`:
  - 0: quantized. The tensor is served natively from `model-skeleton.gguf` by `--native`, and the row carries only its shape.
  - 2: F32.
  - 4: BF16.
  - 5: F16.
- `dense.bin` (1.54 GB) is a 64-B-aligned copy of every float tensor of the skeleton (byte-identical; checked for `blk.0.ffn_gate_inp.weight`). It also carries 96 tensors the skeleton lacks: `blk.N.ffn_{gate,up,down}_exps.{scale,input_scale}`, 512 F32 each.
- **Those `.scale` / `.input_scale` values are NOT the experts' scales.** For example, layer 0 expert 0 has `.scale` = 6.6e-5 / 6.6e-5 / 6.0e-5 while its blob tail is 3.0e-3 / 3.0e-3 / 7.9e-5. No Strata source reads them; the NVFP4 kernels multiply by the blob tail. They are most likely left over from the base model's full-width experts. **Use the blob tails. There is no valid activation `input_scale` for the latent experts** (relevant to W4A4 kernels, section 7). **[file][src]**

### 5.5 `model-skeleton.gguf` (5.99 GB, 1079 tensors, GGUF v3, `general.file_type 7`) **[file]**

All non-expert weights of the main model are here. The full list is `edge0/docs/skeleton-tensors.txt`. Per-layer pattern:

| Tensors | Type | GGUF shape | Layers |
|---|---|---|---|
| `hc_{attn,ffn}_norm` | F32 | [10240] | all 48 |
| `hc_{attn,ffn}_down` | BF16 | [10240, 320] | all 48 |
| `hc_{attn,ffn}_up` | BF16 | [320, 10240] | all 48 |
| `hc_{attn,ffn}_inject` | BF16 | [10240, 4] | all 48 |
| `ffn_gate_inp` (router) | BF16 | [2560, 512] | all 48 |
| `ffn_gate_inp_shexp` | BF16 | [2560, 1] | all 48 |
| `ffn_{gate,up}_shexp` | Q8_0 | [2560, 640] | all 48 |
| `ffn_down_shexp` | Q8_0 | [640, 2560] | all 48 |
| `attn_qkv` | Q8_0 | [2560, 10240] | GDN (36) |
| `attn_gate` | Q8_0 | [2560, 6144] | GDN (36) |
| `ssm_out` | Q8_0 | [6144, 2560] | GDN (36) |
| `ssm_alpha`, `ssm_beta` | BF16 | [2560, 48] | GDN (36) |
| `ssm_conv1d` | F32 | [4, 10240] | GDN (36) |
| `ssm_a`, `ssm_dt.bias` | F32 | [48] | GDN (36) |
| `ssm_norm` | F32 | [128] | GDN (36) |
| `attn_q` | Q8_0 | [2560, 12288] | QSA (12) |
| `attn_k`, `attn_v` | Q8_0 | [2560, 512] | QSA (12) |
| `attn_output` | Q8_0 | [6144, 2560] | QSA (12) |
| `attn_{q,k}_norm` | F32 | [256] | QSA (12) |
| `indexer.q_proj` | BF16 | [2560, 512] | QSA (12) |
| `indexer.k_proj` | BF16 | [2560, 128] | QSA (12) |
| `indexer.{q,k}_norm` | F32 | [128] | QSA (12) |
| `ple_key` | BF16 | [2560, 10240] | layer 1 |
| `ple_value` | BF16 | [2560, 2560] | layer 1 |
| `ple_norm_{key,query,conv}` | F32 | [10240] | layer 1 |
| `ple_conv1d` | F16 | [4, 10240] | layer 1 |
| `token_embd`, `output` | Q8_0 | [2560, 248320] | global |
| `output_hc_norm` | F32 | [10240] | global |
| `output_hc_down`, `output_hc_up` | BF16 | [10240, 320], [320, 10240] | global |

- There are no `ffn_*_exps`, no D/U, no PLE table, no `output_norm` and no MTP.
- Bytes: Q8_0 4.14 GiB, BF16 1.42 GiB, F32 0.009 GiB, total 5.57 GiB. In BF16 the dense part would be about 9.2 GiB; in FP8 about 5.3 GiB.
- Metadata (all in the dump): `qwen4exp.*` geometry, the PLE hash constants, the tokenizer (`gpt2`/`qwen35`, 248,320 tokens, 247,587 merges, token types), the chat template, and sampling defaults.
- Measured fold status: `attn_{q,k}_norm` mean ~1.28, `indexer.q_norm` ~0.96, `hc_*_norm` ~0.94, `ple_norm_*` ~0.89 (all `1+w` folded); `ssm_norm` ~0.97 (plain); `ssm_a` −158 to −0.028 (stored as `−exp(A_log)`); `ssm_dt.bias` −8 to 2.5.

### 5.6 `mtp/` **[file][src]**

- `dense.txt`: one line per tensor, `name kind rows cols offset bytes`, with kind `q8_0|bf16|f32`, rows = outputs and cols = inputs; 29 tensors, 256-B aligned in `dense.bin` (116 MB).
  - Q8_0: `fc_embedding`, `fc_hidden`, `self_attn.{q,k,v,o}_proj`, `self_attn.indexer.index_qk_proj` (640 = 512 q + 128 k), `mlp.shared_expert.{gate,up,down}_proj`.
  - BF16: the hyper-connection matrices, `mlp.gate`, `mlp.shared_expert_gate`.
  - F32 norms **already `1+w`** (`S:tools/mtp_rt.py` adds 1 to the raw GemmaRMSNorm weight).
- `experts.bin`: 512 × 1,382,400 B of full-width Q2_0 experts (`S:tools/mtp_rt.py` `blob_of`). Each record is:
  - gate/up code planes interleaved by row (row 2r = gate r, 2r+1 = up r), 1280 rows × 40 blocks × 16 B;
  - down codes, 2560 rows × 10 blocks × 16 B;
  - gate/up fp16 scales, 1280 × 40 × 2 B;
  - down scales, 2560 × 10 × 2 B.
  - Q2_0 is 64 values per block, `value = (code − 1) × d`, with value 4i+k at bits 2k of byte i.
- `draft_vocab.bin`: int32 token ids (106,299), the rows of `output.weight` the draft head scores.

### 5.7 `profile.bin` (expert ranking for the VRAM cache) **[file][src]**

`S:tools/make_profile.py`, `S:src/core/expert_cache.cpp` `read_expert_profile`:

```
"STRP" | <5I: version=1, n_layers=48, n_expert=512, slots=24576, n_ranked=24576>
| n_ranked x <HH: layer, expert>  (best first)
| int32 table[48][512] = rank of (layer, expert)
```

The size is 196,632 B. The top pairs are (15,494), (15,209), (31,6), (15,6), (3,507), …. The first 4096 ranked pairs are spread unevenly over the layers, 40 to 158 pairs per layer. `--expert-cache auto` fills the free VRAM slots in this order, and the adaptive tier then swaps experts per conversation (`--adapt-swaps`).

### 5.8 Other files

- `model.json`: the Strata launch contract, giving the args and env above.
- `mmproj-BF16.gguf`: Qwen3-VL vision tower (not covered).
- `tokenizer/`: section 6.
- `LICENSE` / `NOTICE` / `release-scope.json`: Edge0 Community Model License 1.0 plus Qwen Community License. Serving it as a commercial model service needs a separate Edge0 license.

---

## 6. Tokenizer and chat template **[file][src]**

- `M:tokenizer/` is **not an HF tokenizer directory**:
  - `tokenizer.json` is a 541-byte Strata summary (`model gpt2`, `pre qwen35`, `vocab_size`, special ids, `add_bos_token false`, the pre-tokenizer regex from llama.cpp `LLAMA_VOCAB_PRE_TYPE_QWEN35`). It is not a `tokenizers` JSON.
  - `vocab.json` (248,320 entries) and `merges.txt` (247,587) are GPT-2 byte-level BPE.
  - `token_type.json`: 1 = normal, 3 = control, 4 = user-defined, 5 = unused (`[PADnnnnnn]`).
  - `chat_template.jinja`.
  - **For vLLM, build an HF tokenizer directory**: `Qwen2TokenizerFast` from vocab and merges, the added special tokens below, and a `tokenizer_config.json` carrying the chat template, `eos_token <|im_end|>`, `pad <|endoftext|>` and `add_bos_token false`. Then validate encode() against Strata/llama.cpp on a sample corpus. `vllm-bench --tokenizer` also needs it.
- Special tokens:

| Id | Token |
|---|---|
| 248044 | `<|endoftext|>` (BOS/PAD, PLE EOS) |
| 248045 | `<|im_start|>` |
| 248046 | `<|im_end|>` (EOS) |
| 248053–248057 | `<|vision_start|> <|vision_end|> <|vision_pad|> <|image_pad|> <|video_pad|>` |
| 248058 / 248059 | `<tool_call>` / `</tool_call>` |
| 248066 / 248067 | `<tool_response>` / `</tool_response>` |
| 248068 / 248069 | `<think>` / `</think>` |
| 248060–248065 | fim and repo tokens |
| 248070–248076 | audio and tts tokens |
| 248077–248319 | `[PAD…]` (unused) |

- Stop ids: Strata stops on `{248044, 248046}` (`S:serve/server.py` `EOS_IDS`).
- Template (`M:tokenizer/chat_template.jinja`; `S:serve/server.py` uses the model folder's template when present):
  - The format is ChatML. The generation prompt is `<|im_start|>assistant\n<think>\n`.
  - **Thinking off** (`chat_template_kwargs={"enable_thinking": false}`) appends `<think>\n\n</think>\n\n` and injects no system effort sentence.
  - **Thinking on** (default) uses `reasoning_effort ∈ {xhigh (default), medium, low}`. When it is xhigh or low, the template puts an effort sentence at the top of the system turn, creating a system turn if none exists. Strata's server also accepts `none`/`high` and maps them.
  - Past assistant turns keep `<think>…</think>` unless `preserve_thinking` is false.
  - Tools use the Qwen3-Coder XML call format (`<tool_call><function=…><parameter=…>`).

---

## 7. Mapping to vLLM `qwen4_exp`: what to convert **[vllm][src][inferred]**

The GGUF was produced by llama.cpp's converter (`L:conversion/qwen4exp.py`, base `L:conversion/qwen.py` `Qwen3NextModel` / `_LinearAttentionVReorderBase`). vLLM loads HF-convention names. Each step below inverts something the converter did, so the weights must be undone into HF form (or vLLM's loader must be told the GGUF convention).

| GGUF (Strata) | HF / vLLM name | Transform |
|---|---|---|
| `token_embd` | `model.embed_tokens.weight` | Q8_0 → bf16 (lossy vs the original, but that is what Strata runs) |
| `output` | `lm_head.weight` | same |
| `blk.l.hc_attn_{norm,down,up,inject}` | `model.layers.l.attn_hyper_connection.{hc_norm, input_mix_weight_down, input_mix_weight_up, block_inject_weight}` | **norm: subtract 1** (vLLM `GroupedGemmaRMSNorm` uses `1+w`); vLLM stacks down+inject into `input_mix_weight_down_block_inject` (its mapper handles that) |
| `blk.l.hc_ffn_*` | `model.layers.l.mlp_hyper_connection.*` | same |
| `output_hc_{norm,down,up}` | `model.hyper_connection_mixer.{hc_norm,input_mix_weight_down,input_mix_weight_up}` | norm −1 |
| `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_a`, `ssm_dt.bias`, `ssm_norm`, `ssm_out` | `linear_attn.{in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, conv1d, A_log, dt_bias, norm, out_proj}` | **V-head reorder (inverse):** the GGUF stores v-heads TILED (v-head j pairs with k-head `j % 16`); HF/vLLM use GROUPED (v-head i pairs with k-head `i // 3`). `HF[i = 3k + r] = GGUF[16r + k]` (k < 16, r < 3), applied to the v rows of `in_proj_qkv` (rows 4096+), the rows of `in_proj_z`, `in_proj_a`/`in_proj_b` (head_dim 1), `A_log`, `dt_bias`, the v channels of `conv1d` (channels 4096+) and the **columns** of `out_proj`. Also `A_log = log(-ssm_a)`, `conv1d` [4,10240] → (10240,1,4), and `ssm_norm` stays as is (not folded). |
| `attn_q`, `attn_k`, `attn_v`, `attn_output`, `attn_{q,k}_norm` | `self_attn.{q_proj, k_proj, v_proj, o_proj, q_norm, k_norm}` | q_proj keeps the per-head [q\|gate] halves; **norms −1** (`L:conversion/qwen.py` adds 1 to every `norm.weight` except `linear_attn.norm`) |
| `indexer.{q_proj,k_proj}` | `self_attn.indexer.index_qk_proj` (rows: 512 q, then 128 k) | concatenate |
| `indexer.{q,k}_norm` | `self_attn.indexer.{q_layernorm,k_layernorm}` | −1 |
| `ffn_gate_inp`, `ffn_gate_inp_shexp`, `ffn_{gate,up,down}_shexp` | `mlp.gate`, `mlp.shared_expert_gate`, `mlp.shared_expert.{gate,up,down}_proj` | dequantize Q8_0, or keep 8-bit |
| `ple_{key,value}` | `model.layers.1.ple.{key_proj,value_proj}` (vLLM stacks them as `ple.kv_proj`) | |
| `ple_norm_{key,query,conv}` | `ple.norm_{key,query,conv}` | −1 |
| `ple_conv1d` | `ple.conv1d.weight` (10240,1,4) | F16 |
| `ple-fp8.gguf` | `model.layers.1.ple.ple_embedding.ngram_embedding` (+ `weight_scale`, hash constants) | The HF checkpoint shards the table into `split_ngram_parts=512` parts (`.shard_k`). vLLM accepts row shards (`PLEVocabParallelEmbedding.weight_loader(checkpoint_start=…)`), so a custom loader can map the GGUF memory in place. |
| `experts.bin` + `labels.bin` + `latent_dense_*.bin` | **no vLLM equivalent**: a new latent MoE block | see below |

- **ModelOpt NVFP4 repack** for vLLM's FP4 MoE kernels (inverse of `L:` `_repack_nvfp4`). Per 16-value sub-block, reorder the nibbles from ggml order (byte j: value j low, value j+8 high) to ModelOpt order (byte i: value 2i low, value 2i+1 high). Move the 4 scale bytes of each 64-block into a `weight_scale` E4M3 matrix (rows × K/16); they are already E4M3 bit patterns with sign 0. Put the tails into `weight_scale_2` (gate and up share one value, which `w13` requires). This is lossless. **[inferred]** There is **no valid `input_scale`** (section 5.4), so a W4A4 path needs calibration or dynamic activation global scales. W4A16 (Marlin-style FP4 weight-only) or W4A8 works without one.
- **Latent block in vLLM**: `xl = D_cat @ x` (one 14336×2560 GEMM), a grouped NVFP4 MoE over 1792-wide inputs where expert e reads `xl[:, labels[l][e]]`, the per-route U projection or the `U_cat` identity (section 2.6), and the existing shared-expert path. D/U are needed by every token and every layer (3.49 GiB in Q8, or keep 8-bit / FP8), so they must stay VRAM-resident.
- **HF `config.json` for vLLM** (transformers `Qwen4ExpTextConfig`, `model_type "qwen4_exp_text"`; the multimodal wrapper is `"qwen4_exp"`). Several transformers defaults are wrong for this model (hidden 2048, 40 layers, 16 heads, 32 v-heads, moe 512), so write every key explicitly **[vllm][inferred]**:
  - Architecture: `architectures ["Qwen4ExpForCausalLM"]`, `vocab_size 248320`, `hidden_size 2560`, `num_hidden_layers 48`, `tie_word_embeddings false`.
  - Attention: `num_attention_heads 24`, `num_key_value_heads 2`, `head_dim 256`, `rms_norm_eps 1e-6`, `max_position_embeddings 262144`, `rope_parameters {rope_type default, rope_theta 1e7, partial_rotary_factor 0.25, mrope_section [11,11,10], mrope_interleaved true}`.
  - GDN: `linear_conv_kernel_dim 4`, `linear_key_head_dim 128`, `linear_value_head_dim 128`, `linear_num_key_heads 16`, `linear_num_value_heads 48`, `output_gate_type "sigmoid"`.
  - MoE: `moe_intermediate_size 640`, `shared_expert_intermediate_size 640`, `num_experts 512`, `num_experts_per_tok 10`, `norm_topk_prob true`.
  - Layer pattern: `layer_types` with `"linear_attention"` for `l%4!=3`, else `"indexed_attention"` (or `"full_attention"`, which is remapped).
  - Hyper-connections: `hc_count 4`, `hc_lowrank 320`.
  - PLE: `ple_layer_ids [2]` (1-based), `ple_embed_dim 2560`, `ple_conv_kernel_size 4`, `ngram_size 3`, `heads_per_ngram 8`, `ngram_vocab_size_base 20000000`, `make_ngram_vocab_size_divisible_by 128`, `seed 1234`, `split_ngram_parts 512`, `eos_token_id 248044`.
  - Indexer: `indexer_n_heads 4`, `indexer_kv_heads 1`, `indexer_head_dim 128`, `indexer_budget 2048`, `indexer_compress_ratio 4`.
  - Edge0's own keys (new, read by our latent block): `latent_dim 1792`, `latent_groups 8`, plus where the group labels live.
- vLLM qwen4_exp references: `V:nvidia/model.py` (decoder, `Qwen4ExpSparseMoeBlock` subclasses `Qwen3NextSparseMoeBlock`), `V:common/hyperconnection.py`, `V:nvidia/qsa.py` + `ops/qsa*.py` + `common/qsa_cache.py`, `V:nvidia/ple_layer.py` + `common/ngram_embedding.py` (PLE as a Mamba-style state with `conv_state_len = (4−1)·3 = 9`), and `V:nvidia/mtp.py`. GDN is vLLM's `QwenGatedDeltaNetAttention(gqa_interleaved_layout=False)`. It needs the sigmoid output gate; check that qwen4_exp configures it.

---

## 8. Expert-loading facts for TidalMoE-style tuning **[file][inferred]**

TidalMoE's goal is to spend less time loading experts: move fewer expert bytes per step, hit VRAM more often, and overlap transfers with compute.

- Unit of transfer: one record of **1,935,376 B** (1.85 MiB), contiguous, holding gate, up, down and the scale tail. There are 24,576 of them (44.30 GiB). A latent expert has 30% fewer parameters than a Qwen3.8-Flash-Next full-width expert, which is the "fewer bytes per expert" win Edge0 already bakes in.
- Per decode step, one sequence touches 48 × 10 = 480 records (929 MB if all missed). At about 50 GB/s of practical PCIe Gen5 x16 H2D, that is about 18.6 ms per token of pure transfer, so the VRAM hit rate dominates. With B concurrent sequences, the unique experts per layer grow sublinearly; measure this from routing traces with `tools/collect_routing.py`-style capture.
- Always-resident, per-token-needed data (not cacheable per expert):
  - D/U: Q8 3.49 GiB, or Q4 1.85 GiB, or roughly 3.5 GiB in FP8.
  - Dense skeleton: 5.57 GiB as Q8_0/BF16; about 9.2 GiB if fully BF16.
  - MTP: about 0.82 GB, plus a 106K-row head subset.
  - KV: 24 KiB/token in BF16.
  - GDN state: about 108 MiB (113 MB) per sequence in fp32 (36 layers × 3 MiB recurrent state, plus 4.4 MB of conv state). **At high concurrency this recurrent state becomes a major VRAM consumer** (64 seqs ≈ 6.8 GiB).
  - On a 32 GB 5090 this leaves roughly 12 to 17 GiB for expert records, about 6.6K to 9.4K of 24,576 (27 to 38%), depending on the dense precision and the batch.
- Strata's placement policy, the baseline to beat:
  - `profile.bin` ranks all pairs, and `--expert-cache auto` fills VRAM in that order.
  - An adaptive tier swaps experts per conversation.
  - Decode misses are split between CPU compute in pinned RAM and a 10% DMA share to the GPU.
  - Prefill streams every expert over PCIe into a ring, overlapped with the previous layer's attention.
  - `--kv-resident 32768` keeps only 32K KV cells in VRAM.
- Routing skew to exploit: the per-layer coverage of the profile's top-4096 pairs varies 4× (40 to 158 pairs per layer), and group sizes vary 8 to 200.

---

## 9. Unconfirmed items and risks

1. **QSA selection past 2051 cells**: Strata selects at the cell level (whole blocks plus a partial block when the tail has fewer than 3 cells). vLLM's QSA kernels may cut on whole blocks. Measure at contexts above 2K. **[unconfirmed]**
2. **HF conventions** (norm +1, v-head reorder, A_log) come from llama.cpp's converter source and from vLLM's module code. No HF checkpoint of Qwen3.8-Flash-Next is on disk to cross-check. Verify with a teacher-forced comparison against Strata's reference outputs. **[inferred]**
3. **dense.bin `.scale`/`.input_scale`** do not match the expert tails. They are believed stale and unused; no source reads them. There are no activation input scales for the latent experts.
4. **Strata MTP attention is dense**, while vLLM's MTP uses the QSA indexer. This affects only draft acceptance, not outputs.
5. **README's "skeleton projections ≈ 2.9 GiB VRAM"** does not match the 5.57 GiB skeleton. Strata may keep `token_embd`/`output` elsewhere or reuse `dense.bin` forms. Not traced. **[unconfirmed]**
6. **The model card's 1,630,720 B record** is the Q3_K build's. The NVFP4 record is 1,935,376 B.
7. **Exact numerical parity** with Strata is impossible in BF16. Strata quantizes activations to q8_1/Q8_K/BF16 per weight's "activation contract", stores KV in INT8, and splits misses between CPU and GPU, which "changes bits" per its docs.
8. **Vision** (`mmproj`, the image-token path in PLE and M-RoPE) is not covered.

---

## 10. `edge0/convert/strata_formats.py`: readers and check results

```
python3 edge0/convert/strata_formats.py check --model /home/andychen/models/Edge0-145B-NVFP4 [--moe] [--du-fmt 4|8]
python3 edge0/convert/strata_formats.py skeleton --model ... --out edge0/docs/skeleton-tensors.txt
```

The checks ran on 5090 (2026-11-19) with `~/work/edge0-vllm/.venv-tools` (numpy 2.5.5), under `nice -n 19 ionice -c3` with 2 threads. Max RSS was 614 MB (with the whole-layer D/U check) and the run took about 1 s. Result: **ALL CHECKS PASSED**.

- experts.bin layout matches for all 48 layers.
- Decoded experts (layers 0, 47) are finite, with gate rms 0.083 to 0.110, up rms 0.074 to 0.086 and down rms 0.025 to 0.029. Every NVFP4 sub-block holds a code of magnitude 6.
- labels are in 0..7.
- D/U rms is about 0.010 to 0.011, and `LatentDense.layer(5)` (D_cat, U_cat) matches the per-group reads.
- profile: 24,576 ranked pairs, and the table is consistent.
- Skeleton Q8_0, BF16 and F32 tensors are finite; the router bytes are identical to dense.bin.
- PLE: the head offsets are consistent, the hash ran on a 6-token sequence, and the gathered 2560-d embedding has rms 0.0079.
- MTP: dense tensors and Q2_0 expert 5 decode finite (rms ~0.019 to 0.020).
- One-token latent MoE reference on layer 0 with a random x:
  - experts `[27, 134, 163, 95, 189, 119, 103, 390, 155, 75]` in groups `[1, 6, 6, 1, 1, 1, 1, 3, 3, 1]`;
  - shared gate 0.53;
  - routed output rms 0.032 and MoE output rms 0.042, both finite.

API: `EdgeModel(dir).expert(l, e)` → `{gate (640,1792), up, down (1792,640), scales}`; `.latent_du(l, g)` → `D (1792,2560), U (2560,1792)`; `.latent().layer(l)` → `D_cat (14336,2560), U_cat (2560,14336)`; `.labels()`; `.ple().rows(ids)`; `ngram_rows(...)` / `ngram_rows_for_sequence(...)`; `.skeleton().tensor(name, row0, nrows)`; `.pack().tensor(name)`; `read_profile`, `read_mtp_index`, `read_mtp_tensor`, `read_mtp_expert`, `read_draft_vocab`; `latent_moe_reference(m, layer, x)` (a float32 oracle for one token's MoE block, useful as a unit test for a vLLM latent-MoE kernel); `route(logits)`.
