# Checkpoint conversion log

The checkpoint converter (`edge0/convert/`, see its README for the commands) writes this file. It implements DESIGN.md
section 1. Every number below comes from the checkpoints' own `edge0_convert.json`.

## 2026-11-19: first conversion (converter agent)

Environment: the 5090 box, run in `nvcr.io/nvidia/vllm:26.11-py3` (numpy 2.1.0, transformers 5.20.2,
tokenizers 0.23.3, safetensors 0.8.0, torch 2.15.0a0), CPU only, after `GPU_FREE` existed. The source was
`/home/andychen/models/Edge0-145B-NVFP4` (read only, never modified).

### Outputs

| Checkpoint | Shards | Tensors | Tensor bytes | Wall time | Checks |
|---|---|---|---|---|---|
| `/home/andychen/models/Edge0-145B-vllm-L4` (4 layers, no MTP) | 5 | 136 | 7,308,469,892 | 26 s (its `edge0_convert.json` says 0.5 s: the recorded run was a resume that only rewrote the small files) | all PASS (18 s) |
| `/home/andychen/models/Edge0-145B-vllm` (48 layers + MTP) | 50 | 7,673 | 60,394,784,260 | 522 s, `--jobs 6` | all PASS (53 s) |

- Full checkpoint files:
  - shard 1 (dense) is 11,299,543,764 B; it took 425 s because it ran next to six expert writers on a saturated NVMe;
  - each expert shard holds 990,910,464 B of tensor data; one shard takes 3.5 s alone and up to 73 s when 6 run at
    once;
  - shard 50 (MTP) is 1,532,440,400 B and took 97 s (512 Q2_0 → NVFP4 re-quantizations).
- The checkpoint takes 57 GB of new disk; `ple-fp8.gguf` is a hardlink (inode 11014573, now 3 links: source, L4,
  full). The disk had 199 GB free afterwards.
- The L4 checkpoint is 7.31 GB, not the ~4.5 GB estimated in DESIGN 1.8: the BF16 `embed_tokens` and `lm_head` alone
  are 2.54 GB.

### Assertion counts (all 48 × 512 routed experts)

| Item | Count |
|---|---|
| E4M3 block scales with bit 7 set | 0 |
| block scales equal to 0x7F | 0 |
| experts with `s_gate != s_up` | 0 |
| experts whose tail `[3] != 0` | 0 |
| block scales below 2^-6 (w13 and w2, every layer) | 0.0% |

So the repack is purely lossless, and no scale is in the range that `nvfp4_marlin_process_scales` zeroes before the
per-layer factor is applied.

### MTP expert re-quantization (Q2_0 → NVFP4, DESIGN 1.5)

Relative Frobenius error, `||deq - ref|| / ||ref||`, over the 512 experts:

| Projection | Mean | Max | Worst expert |
|---|---|---|---|
| gate | 2.64% | 2.84% | 63 |
| up | 2.64% | 2.85% | 167 |
| down | 2.65% | 2.86% | 455 |

The E2M1 codes are exact for Q2 grids. All of the error comes from rounding each 16-value block scale to E4M3. It
affects only draft acceptance.

### Acceptance checks (DESIGN 1.7), full checkpoint

| # | Result |
|---|---|
| 1 contract | 7,673 tensors match name, dtype and shape. The shard rule, index, `total_size`, exact `config.json` and `generation_config.json`, hardlink, `profile.bin` and `draft_vocab.json` (106,299 ids) all check out. |
| 2 lossless | 5,997 sampled raw rows are byte-equal and 303 Q8→BF16 rows equal `ml_dtypes` bf16. All 351 small F32/F16/I32 tensors are exact. 1,600 experts (all of layers 0, 23 and 47, plus 64 random) are bit-identical to `EdgeModel.expert`. |
| 3 GDN reorder | On layer 0, 3 tokens: relative error 4.6e-7 between GGUF-tiled and HF-grouped. The control run with the wrong pairing gives 1.51, so the check can detect a bad reorder. |
| 4 latent MoE | Layers 0, 24 and 47 × 4 random x: D5 on the converted tensors vs `latent_moe_reference`. Max relative error 3.9e-7, mean 3.2e-7. |
| 5 tokenizer | id→token is identical for all 248,320 ids, and the merges are identical. `encode()` equals Qwen3.8 on 2,010 texts (edge cases plus ShareGPT turns), and every text round-trips through encode/decode. The thinking-off prompt ends with `<think>\n\n</think>\n\n`, and eos/pad are 248046/248044. **The prompt token counts equal Strata's `usage.prompt_tokens` for all 8 `strata-ref.json` prompts** (24, 28, 26, 26, 20, 28, 23, 29). All 261 tokens Strata generated are single vocabulary entries. |
| 6 safe_open | All 7,673 tensors open with `safe_open(framework="pt")` in the image, with the right torch dtypes. |

### Decisions taken inside the contract's latitude, and observations for the coordinator

1. **Tokenizer branch.** The Qwen3.8 HF tokenizer is identical in ids and merges, but it lacks the 243 `[PAD…]`
   fillers, so `tokenizer.json` is built from Edge0's own `vocab.json` and `merges.txt` with the `qwen35` regex, as
   DESIGN 1.3 requires:
   - The BPE vocab is dense (0..248319) and includes the specials. With an id gap, `tokenizers` silently re-numbers
     the added tokens.
   - The NFC normalizer and the ByteLevel stages are copied from Qwen3.8.
   - The added-token flags come from Qwen3.8. The six FIM/repo tokens are non-special there, although Edge0's
     `token_type` says "control". This only affects `skip_special_tokens` decoding.
2. **transformers warning.** transformers 5.20 logs "incorrect regex pattern … `fix_mistral_regex`" because
   `config.json` has no `transformers_version` key. It is harmless (encode parity is proven above). If the log noise
   matters, the coordinator can add `"transformers_version": "5.20.2"` to the contract.
3. **Alignment order.** Inside each shard, tensors whose size is not a multiple of 8 are stored last, which keeps
   every other tensor 8-byte aligned (safetensors allows no padding). The tensor set, shapes and dtypes are unchanged.
4. **L4 config** keeps `mtp_num_hidden_layers: 1`, following "same contract". It has no `mtp.*` tensors or quantized
   keys.
