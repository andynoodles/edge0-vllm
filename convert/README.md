# edge0/convert: Strata model folder to vLLM checkpoint

This directory converts `/home/andychen/models/Edge0-145B-NVFP4` (Strata's format) into the HF/safetensors checkpoint
that `edge0/docs/DESIGN.md` section 1 specifies. It also validates the result against that contract.

Status (2026-11-19): both checkpoints are built and pass every acceptance check (DESIGN 1.7, items 1 to 6).

| Checkpoint | Contents | Wall time |
|---|---|---|
| `/home/andychen/models/Edge0-145B-vllm-L4` | mini, 4 layers, no MTP, 7.31 GB | 26 s |
| `/home/andychen/models/Edge0-145B-vllm` | full, 48 layers + MTP, 60.39 GB of tensors (57 GB on disk; the PLE table is a hardlink) | 522 s |

Results and timings are in `edge0/docs/CONVERT.md`, and each checkpoint carries its own record in `edge0_convert.json`.

## Files

| File | What it does |
|---|---|
| `ckpt_contract.py` | The contract as code: every tensor name, dtype and shape, the shard plan and shard rule, and the exact `config.json` and `generation_config.json`. The converter and the checker both use it. Needs only the standard library, so the plugin can import it. `python3 ckpt_contract.py` prints the sizes. |
| `convert_checkpoint.py` | The streaming converter. Uses numpy, plus `strata_formats.py` for parsing. |
| `check_ckpt.py` | The validator, levels `contract`, `deep` and `full` (below). The model agent reuses it. |
| `hf_tokenizer.py` | Builds and checks the HF tokenizer files (DESIGN 1.3). |
| `st_io.py` | Low-memory IO helpers: a streaming safetensors writer (`ShardWriter`), a header and slice reader, an mmap source reader that drops pages after use, a throttle, and bf16/E4M3/E2M1 codecs. |
| `strata_formats.py` | numpy readers for the Strata files (written by the spec agent). It is the oracle for the checks. |
| `test_convert.py` | Unit tests for the repack, the codecs and the writer. When `ml_dtypes`, `torch` and `safetensors` are installed it also cross-checks against them. |
| `run_convert.sh` | Wrapper that runs `convert`, `check` or `test` inside `nvcr.io/nvidia/vllm:26.11-py3`. It runs CPU-only, with no network, as your own uid, under `nice -n 19 ionice -c3`, with docker `--cpus/--memory` caps. It can also run detached in tmux. |

## Commands (on the 5090; the code is at `~/work/edge0-vllm/convert` after the rsync)

Inside the container, `/models` is `/home/andychen/models`. `run_convert.sh` mounts the Qwen3.8 tokenizer at
`/tokref` and `~/work/edge0-bench` at `/bench`. It also adds `--src /models/Edge0-145B-NVFP4 --hf-tokenizer /tokref`
to `convert` automatically.

```bash
cd ~/work/edge0-vllm/convert

# unit tests (1.4 s)
./run_convert.sh test

# L4 mini checkpoint (DESIGN 1.8): 26 s plus 18 s of checks
CPUS=4 MEM=8g TMUX_SESSION=edge0-convert ./run_convert.sh convert \
  --out /models/Edge0-145B-vllm-L4 --num-layers 4 --no-mtp --checks full \
  --corpus /bench/datasets/ShareGPT_V3_unfiltered_cleaned_split.json --strata-ref /bench/strata-ref.json

# full checkpoint (DESIGN 1.1): 522 s on the NVMe (IO-bound), plus 53 s of checks
CPUS=8 MEM=16g TMUX_SESSION=edge0-convert ./run_convert.sh convert \
  --out /models/Edge0-145B-vllm --jobs 6 --checks full \
  --corpus /bench/datasets/ShareGPT_V3_unfiltered_cleaned_split.json --strata-ref /bench/strata-ref.json

# follow a detached run (the log ends with EXIT=<code>)
tail -f "$(ls -t ~/work/edge0-vllm/logs/edge0-convert-*.log | head -1)"
```

**Resume.** Run the same command again. A shard whose final file has exactly the expected header and size is
skipped, and leftover `*.partial` files are deleted and rewritten. The small files (config, tokenizer, hardlink,
profile, draft vocab, index) and the checks run every time. `--force` rewrites every shard.

**Low-impact mode** (for use while something else is benchmarking on the box): 2 CPUs, a 3 GB cgroup that also caps
this job's page cache, and an IO cap:

```bash
CPUS=2 MEM=3g ./run_convert.sh convert --out /models/X --num-layers 4 --no-mtp --throttle-mbps 150
```

### Validation only

```bash
# contract: names, dtypes, shapes, shard rule, index, config.json, generation_config.json, tokenizer files,
# PLE hardlink, profile.bin, draft_vocab.json. numpy only, so the tools venv works: 0.2 s, 49 MB RSS
~/work/edge0-vllm/.venv-tools/bin/python check_ckpt.py --ckpt /home/andychen/models/Edge0-145B-vllm

# deep: adds lossless and numeric checks against the Strata folder (1.7 items 2-4).
# Full model: 31 s, 0.59 GB max RSS (2 BLAS threads). L4: 13 s, 0.59 GB.
~/work/edge0-vllm/.venv-tools/bin/python check_ckpt.py --ckpt /home/andychen/models/Edge0-145B-vllm \
  --src /home/andychen/models/Edge0-145B-NVFP4 --level deep

# full: adds the tokenizer check (item 5) and safe_open(framework="pt") of every tensor (item 6). Needs the image.
./run_convert.sh check --ckpt /models/Edge0-145B-vllm --src /models/Edge0-145B-NVFP4 --level full \
  --corpus /bench/datasets/ShareGPT_V3_unfiltered_cleaned_split.json --strata-ref /bench/strata-ref.json --write-results
```

`--write-results` stores the result in `<ckpt>/edge0_convert.json["checks"]`. `--json FILE` also writes it to a
file. The exit status is 0 only when every check passes.

## What the checks prove

| # | Check | How |
|---|---|---|
| 1 | contract | Every one of the 7,673 tensors (136 for L4) has the contract's name, dtype and shape. Data offsets are contiguous, file sizes match, `weight_map` follows the shard rule, and `total_size` is right. `config.json` and `generation_config.json` must equal `ckpt_contract.build_config` exactly. `eos_token_id` must be the int 248044, and `quantization_config` must not contain a top-level `quant_algo`. `ple-fp8.gguf` must be the same inode as the source, `profile.bin` byte-equal, and `draft_vocab.json` equal to `mtp/draft_vocab.bin`, sorted. |
| 2 | lossless | Sampled rows of every raw tensor (Q8 blocks and BF16) must be byte-equal to the source. The row maps are re-derived from the contract text, including the GDN v-head reorder and the out_proj column-block permutation. Q8→BF16 rows must equal `ml_dtypes` bf16(deq_q8_0(src)). Every F32/F16/I32 tensor must equal its transformed source (norms −1, `log(-ssm_a)`, reorders, conv reshape). Experts: all 512 experts of layers 0, 23 and 47, plus 64 random pairs, are decoded from the converted ModelOpt tensors and must be bit-identical (float32 `array_equal`) to `EdgeModel.expert(l, e)`. MTP experts: the re-quantization error is reported. |
| 3 | GDN reorder | A numpy 3-token GDN recurrence from zero state, run on layer 0. The GGUF tensors with tiled pairing (`h % 16`) must match the converted tensors with grouped pairing (`h // 3`). A control run with the wrong pairing must disagree. |
| 4 | latent MoE | DESIGN D5 runs on the converted tensors: D_cat, route-as-token, per-group combine, one U_cat GEMM, then the shared expert. It is compared with `strata_formats.latent_moe_reference` on 3 layers × 4 random x. |
| 5 | tokenizer | id→token must equal Edge0's `vocab.json` for all 248,320 ids, and the merges must be equal. `encode()` must equal the Qwen3.8 tokenizer on 10 edge-case strings plus the first 2,000 ShareGPT turns. The text `[PAD248100]` must not become the filler id. The thinking-off chat prompt must end with `<|im_start|>assistant\n<think>\n\n</think>\n\n`. eos/pad must be 248046/248044. The `strata-ref.json` prompt token counts must equal Strata's `usage.prompt_tokens`, and every token Strata generated must be one vocabulary entry. |
| 6 | safe_open | `safetensors.safe_open(framework="pt")` opens every shard in the image. The keys match the raw header, and every tensor gets the right torch dtype (`float8_e4m3fn` for F8_E4M3). |

## Implementation notes (things later agents should know)

- **Memory and IO.** Each source is mmapped read-only and read one slice at a time: one GGUF tensor, one D/U
  layer, or 32 expert records (62 MB). The slice is copied out, then dropped (`madvise(DONTNEED)` +
  `posix_fadvise(DONTNEED)`). Shards are written with `pwrite` into a preallocated `.partial` file, with
  `fdatasync` + `DONTNEED` every 256 MB, then `fsync` + rename. Measured RSS: 0.15 to 0.27 GB per expert worker and
  about 0.1 GB for the dense writer. One `docker stats` sample during the 6-job full run showed 0.87 GiB for the whole
  container cgroup, page cache included. That was a single sample, not a measured peak. The full run is
  NVMe-bound: the disk sat at 96% utilisation, and the dense shard took 425 s because it ran next to six expert
  writers.
- **Expert repack** (`repack_ggml_nvfp4`). Within each 16-value sub-block with source bytes `q[0..7]`, output byte
  `m < 4` is `lo(q[2m]) | lo(q[2m+1]) << 4`, and byte `m ≥ 4` is `hi(q[2m-8]) | (q[2m-7] & 0xF0)`. The scales are
  the 4 `d` bytes of each 36-byte block, in order. A test checks this against the contract pseudo-code literally.
  The tails go to `w13_weight_scale_2[:, 0:2]` and `w2_weight_scale_2`. Measured over all 48 × 512 experts: 0 scales
  with bit 7 set, 0 scales equal to 0x7F, 0 cases of `s_gate != s_up`, `tail[3]` always 0, and **0 block scales
  below 2^-6**. So Marlin's zeroing of scales below `2^-6/factor` should not trigger, though that depends on the
  plugin's per-layer factor.
- **Byte order inside a shard.** Contract catalogue order, except that tensors whose size is not a multiple of 8 go
  last, so every other tensor stays 8-byte aligned. The only such tensors are
  `model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale` (shard 1) and the 3,072 0-dim MTP
  `weight_scale_2` / `input_scale` scalars (shard 50). Expert shard files are 990,911,208 B for layers 0–9 and
  990,911,216 B for layers 10–47 (the header is 8 B longer); the tensor data is always 990,910,464 B.
- **Tokenizer** (`hf_tokenizer.py`). The Qwen3.8 HF tokenizer has the same ids and merges as Edge0, but it lacks
  the 243 `[PAD248077]`..`[PAD248319]` fillers. So the converter takes the contract's "build from Edge0's own files"
  branch:
  - The BPE vocab is Edge0's full, dense `vocab.json`, with the fillers as plain entries (never produced by
    `encode()`).
  - The merges are Edge0's.
  - The Split regex is Edge0's `qwen35` pattern.
  - The NFC normalizer and the ByteLevel pre-tokenizer, post-processor and decoder are copied from Qwen3.8.
  - The 33 added tokens use Qwen3.8's flags. `<think>`, `</think>`, `<tool_call>`, the FIM tokens, `<|repo_name|>`
    and `<|file_sep|>` are non-special, as in Qwen3.8. Edge0's `token_type` calls the six FIM/repo tokens "control".
  - Pitfall: the specials must also be in the BPE vocab. If the vocab has an id gap, `tokenizers` re-numbers the added
    tokens to `vocab_size + k` (measured: `<|im_end|>` became 248289).
- **Harmless warning.** transformers 5.20 prints "incorrect regex pattern ... fix_mistral_regex" for any local
  checkpoint whose `config.json` has no `transformers_version` key. It changes nothing (the encode checks pass). Adding
  `"transformers_version": "5.20.2"` to `config.json` would silence it, but that changes the contract, so it is the
  coordinator's call.
- **MTP experts** (Q2_0 → NVFP4, standard ModelOpt recipe, DESIGN 1.5). The relative Frobenius error is 2.64% on
  average and 2.86% at worst (gate, up and down alike). The E2M1 codes of Q2 grids `{-d, 0, d, 2d}` come out exact;
  all of the error comes from rounding each 16-block scale to E4M3. A per-block choice of the code multiplier could
  reduce it if draft acceptance turns out to matter.
- **Config details.**
  - `edge0.source_dir` records the host path (`--source-dir-record`), also when the converter runs in docker.
  - The L4 config keeps `mtp_num_hidden_layers: 1`. The contract says "same contract" apart from the layer count and
    `layer_types`. L4 has no `mtp.*` tensors or `quantized_layers` keys.
- **L4 size.** The L4 checkpoint is 7.31 GB, not the "about 4.5 GB" estimated in DESIGN 1.8. The BF16 embedding and
  LM head alone are 2.54 GB, and the four expert shards are 3.96 GB.
