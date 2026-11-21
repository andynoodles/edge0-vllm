#!/usr/bin/env python3
"""edge0/convert/hf_tokenizer.py - build and check the HF tokenizer files of the converted checkpoint (DESIGN 1.3).

Edge0's tokenizer/ folder is not an HF tokenizer (tokenizer.json there is a 541-byte Strata summary).  The design
says: start from the coordinator's Qwen3.8 HF tokenizer (tokenizer-qwen38/), verify that id->token is identical for
all 248,320 ids and that the merges are identical; if anything differs, build tokenizer.json from Edge0's own
vocab.json + merges.txt with the qwen35 pre-tokenizer regex.

Measured 2026-11-19: the Qwen3.8 tokenizer has 248,044 BPE tokens + 33 added tokens (ids up to 248,076); Edge0's
vocab additionally has the 243 unused [PAD248077]..[PAD248319] fillers.  So the id->token check fails by design and
build() takes the "build from Edge0's own files" branch:
  - model: BPE over the whole Edge0 vocab.json (dense ids 0..248319, specials included; a gap would make tokenizers
    re-number the added tokens), INCLUDING the [PAD] fillers as plain vocab entries (no merge produces them, so they
    are never emitted by encode(), as in llama.cpp where they are type 5 "unused"; they still decode to their text),
    merges = Edge0 merges.txt.
  - added_tokens: the 33 type 3/4 tokens (248044..248076) with the Qwen3.8 reference's flags, so <think>, </think>,
    <tool_call>, ... stay non-special exactly as in the Qwen3.8 tokenizer.json (the reference also marks the six
    <|fim_*|>/<|repo_name|>/<|file_sep|> tokens non-special although Edge0's token_type calls them control).
  - normalizer / pre-tokenizer / post-processor / decoder: the Qwen3.8 structure (NFC, Split(regex, Isolated) +
    ByteLevel(use_regex=false)), with the Split regex replaced by Edge0's qwen35 pattern (tokenizer/tokenizer.json
    "pre_pattern").
  - tokenizer_config.json: the Qwen3.8 one with chat_template = Edge0 chat_template.jinja, eos <|im_end|>,
    pad <|endoftext|>, add_bos_token false, tokenizer_class Qwen2Tokenizer.
build() needs only the standard library.  check() needs `tokenizers` and `transformers` (present in the vLLM image).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

N_VOCAB = 248320
THINK_OFF_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
_BYTE_LEVEL = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}
_FALLBACK_SPECIAL_LIST = ["<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
                          "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>",
                          "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>"]


def _read_merges(path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    if lines and lines[0].startswith("#version"):
        lines = lines[1:]
    return [ln for ln in lines if ln]


def _merge_str(m) -> str:
    return m if isinstance(m, str) else " ".join(m)


def load_edge0(tok_dir) -> dict:
    tok_dir = Path(tok_dir)
    vocab = json.loads((tok_dir / "vocab.json").read_text(encoding="utf-8"))
    inv = [None] * N_VOCAB
    for t, i in vocab.items():
        inv[i] = t
    assert len(vocab) == N_VOCAB and None not in inv, "Edge0 vocab.json is not a dense 0..248319 map"
    tt = json.loads((tok_dir / "token_type.json").read_text())
    assert len(tt) == N_VOCAB
    summary = json.loads((tok_dir / "tokenizer.json").read_text())
    return {"vocab": vocab, "inv": inv, "token_type": tt, "merges": _read_merges(tok_dir / "merges.txt"),
            "pre_pattern": summary["pre_pattern"], "summary": summary,
            "chat_template": (tok_dir / "chat_template.jinja").read_text(encoding="utf-8")}


def compare_with_reference(ed: dict, ref_dir) -> dict:
    """id->token and merges comparison against an HF tokenizer directory (tokenizer.json)."""
    ref = json.loads((Path(ref_dir) / "tokenizer.json").read_text(encoding="utf-8"))
    ref_inv = {i: t for t, i in ref["model"]["vocab"].items()}
    for a in ref["added_tokens"]:
        ref_inv[a["id"]] = a["content"]
    missing = [i for i in range(N_VOCAB) if i not in ref_inv]
    differ = [i for i in range(N_VOCAB) if i in ref_inv and ref_inv[i] != ed["inv"][i]]
    extra = [i for i in ref_inv if not 0 <= i < N_VOCAB]
    ref_merges = [_merge_str(m) for m in ref["model"]["merges"]]
    rep = {"ref_dir": str(ref_dir), "ref_bpe_vocab": len(ref["model"]["vocab"]), "ref_added": len(ref["added_tokens"]),
           "ids_missing_in_ref": len(missing), "ids_missing_range": [missing[0], missing[-1]] if missing else None,
           "missing_are_edge0_pad_fillers": all(ed["inv"][i].startswith("[PAD") for i in missing),
           "ids_differing": len(differ), "differ_examples": [(i, ref_inv[i], ed["inv"][i]) for i in differ[:5]],
           "ids_extra_in_ref": len(extra), "merges_equal": ref_merges == ed["merges"],
           "merges_counts": [len(ref_merges), len(ed["merges"])]}
    rep["identical"] = not missing and not differ and not extra and rep["merges_equal"]
    return rep


def build(edge0_tok_dir, ref_dir, out_dir) -> dict:
    """Write tokenizer.json, tokenizer_config.json, vocab.json, merges.txt, special_tokens_map.json and
    chat_template.jinja into out_dir.  Returns a JSON-able report."""
    edge0_tok_dir, out_dir = Path(edge0_tok_dir), Path(out_dir)
    ed = load_edge0(edge0_tok_dir)
    ref_dir = Path(ref_dir) if ref_dir else None
    have_ref = bool(ref_dir and (ref_dir / "tokenizer.json").is_file())
    rep = {"edge0_dir": str(edge0_tok_dir)}
    ref = None
    if have_ref:
        rep["compare"] = compare_with_reference(ed, ref_dir)
        ref = json.loads((ref_dir / "tokenizer.json").read_text(encoding="utf-8"))
    else:
        rep["compare"] = None
        rep["warning"] = "no HF reference tokenizer at %s; built from Edge0 files with the built-in structure" % ref_dir

    if have_ref and rep["compare"]["identical"]:
        rep["branch"] = "copied the HF reference tokenizer.json (identical ids and merges)"
        shutil.copyfile(ref_dir / "tokenizer.json", out_dir / "tokenizer.json")
    else:
        rep["branch"] = "built tokenizer.json from Edge0 vocab.json + merges.txt with the qwen35 regex"
        tt = ed["token_type"]
        added_ids = [i for i in range(N_VOCAB) if tt[i] in (3, 4)]
        assert len(added_ids) == 33 and added_ids == list(range(248044, 248077)), added_ids
        assert all(tt[i] == 1 for i in range(248044)) and all(tt[i] == 5 for i in range(248077, N_VOCAB))
        # The BPE vocab is Edge0's full, dense id map (specials included).  It must have no id gaps: tokenizers
        # re-numbers added tokens that are missing from the model vocab as model.get_vocab_size() + k, ignoring the
        # ids in the JSON (measured: with the 33 specials left out, <|im_end|> became 248289).
        bpe_vocab = {ed["inv"][i]: i for i in range(N_VOCAB)}
        ref_added = {a["id"]: a for a in ref["added_tokens"]} if ref else {}
        added, flag_diff = [], []
        for i in added_ids:
            a = {"id": i, "content": ed["inv"][i], "single_word": False, "lstrip": False, "rstrip": False,
                 "normalized": False, "special": tt[i] == 3}
            if i in ref_added:
                # the reference's flags win (they are the base model's); they only change skip_special_tokens
                # decoding, never encode().  Measured: <|fim_*|>, <|repo_name|>, <|file_sep|> (248060-248065) are
                # token_type 3 (control) in Edge0 but special=false in Qwen3.8.
                r = ref_added[i]
                assert r["content"] == a["content"], (r, a)
                if r["special"] != a["special"]:
                    flag_diff.append([i, a["content"], r["special"]])
                for k in ("special", "single_word", "lstrip", "rstrip", "normalized"):
                    a[k] = r[k]
            added.append(a)
        rep["special_flag_taken_from_ref"] = flag_diff
        split = {"type": "Split", "pattern": {"Regex": ed["pre_pattern"]}, "behavior": "Isolated", "invert": False}
        if ref:
            normalizer, decoder, post = ref["normalizer"], ref["decoder"], ref["post_processor"]
            pre = json.loads(json.dumps(ref["pre_tokenizer"]))
            assert pre["type"] == "Sequence" and pre["pretokenizers"][0]["type"] == "Split", pre
            rep["ref_regex"] = pre["pretokenizers"][0]["pattern"]["Regex"]
            pre["pretokenizers"][0] = split
            model = {k: v for k, v in ref["model"].items() if k not in ("vocab", "merges")}
        else:
            normalizer, decoder, post = {"type": "NFC"}, dict(_BYTE_LEVEL), dict(_BYTE_LEVEL)
            pre = {"type": "Sequence", "pretokenizers": [split, dict(_BYTE_LEVEL)]}
            model = {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
                     "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False}
        model["vocab"] = bpe_vocab
        model["merges"] = ed["merges"]
        tj = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": added, "normalizer": normalizer,
              "pre_tokenizer": pre, "post_processor": post, "decoder": decoder, "model": model}
        rep["regex"] = ed["pre_pattern"]
        rep["bpe_vocab"] = len(bpe_vocab)
        rep["added_tokens"] = len(added)
        with open(out_dir / "tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(tj, f, ensure_ascii=False)

    # tokenizer_config.json
    cfg = json.loads((ref_dir / "tokenizer_config.json").read_text(encoding="utf-8")) if have_ref and (
        ref_dir / "tokenizer_config.json").is_file() else {}
    if not cfg:
        cfg = {"add_prefix_space": False, "bos_token": None, "clean_up_tokenization_spaces": False, "errors": "replace",
               "model_max_length": 262144, "split_special_tokens": False, "unk_token": None,
               "additional_special_tokens": list(_FALLBACK_SPECIAL_LIST),
               "added_tokens_decoder": {str(i): {"content": ed["inv"][i], "lstrip": False, "normalized": False,
                                                 "rstrip": False, "single_word": False,
                                                 "special": ed["token_type"][i] == 3} for i in range(248044, 248077)}}
    cfg["chat_template"] = ed["chat_template"]
    cfg["eos_token"] = "<|im_end|>"
    cfg["pad_token"] = "<|endoftext|>"
    cfg["add_bos_token"] = False
    cfg["tokenizer_class"] = "Qwen2Tokenizer"
    if "pretokenize_regex" in cfg:
        cfg["pretokenize_regex"] = ed["pre_pattern"]
    with open(out_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    stm = {"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>",
           "additional_special_tokens": cfg.get("additional_special_tokens", _FALLBACK_SPECIAL_LIST)}
    with open(out_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump(stm, f, ensure_ascii=False, indent=2)
    shutil.copyfile(edge0_tok_dir / "vocab.json", out_dir / "vocab.json")
    shutil.copyfile(edge0_tok_dir / "merges.txt", out_dir / "merges.txt")
    shutil.copyfile(edge0_tok_dir / "chat_template.jinja", out_dir / "chat_template.jinja")
    return rep


# ============================================================================== checks (need tokenizers/transformers)
SAMPLE_TEXTS = [
    "Hello, world! It's a test. They'll go; we'd see. I'M HERE, YOU'RE THERE, IT'S FINE.",
    "用繁體中文簡單說明什麼是「不當得利」，兩句話以內。",
    "請把這句話翻譯成英文：今天下午三點在會議室開會。",
    "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\n\n",
    "Numbers 1234567890 and 3.14159, 2^64 = 18446744073709551616.\r\n\r\nTabs\tand   spaces   ",
    "Émojis 😀👍🏽 and accents café naïve Ångström; Ελληνικά; русский текст; العربية; हिन्दी; 日本語のテキスト。",
    "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHello!<|im_end|>",
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
    "[PAD248100] is an unused filler and must not be matched as one token.",
    "  leading spaces, trailing newline\n",
]


def _corpus_from_sharegpt(path, max_texts: int, head_bytes: int = 16 << 20) -> list[str]:
    """The first conversations of a ShareGPT json array, decoded incrementally from the first `head_bytes` of the file
    (the whole file is ~650 MB; json.loads of it would need several GB)."""
    with open(path, "rb") as f:
        text = f.read(head_bytes).decode("utf-8", "ignore")
    dec, pos, data = json.JSONDecoder(), text.index("[") + 1, []
    while True:
        while pos < len(text) and text[pos] in " \t\r\n,":
            pos += 1
        try:
            obj, pos = dec.raw_decode(text, pos)
        except ValueError:
            break
        data.append(obj)
    out = []
    for conv in data:
        for turn in conv.get("conversations", []):
            v = turn.get("value")
            if isinstance(v, str) and v:
                out.append(v[:4000])
                if len(out) >= max_texts:
                    return out
    return out


def _bytes_to_unicode() -> dict:
    """GPT-2 byte-level alphabet (byte -> printable char), as used by ByteLevel pre-tokenizers."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def check(ckpt_dir, edge0_tok_dir, ref_dir=None, corpus=None, corpus_max: int = 2000, strata_ref=None) -> dict:
    """1.7 item 5.  Returns {"ok": bool, ...}.  Raises ImportError if tokenizers/transformers are missing."""
    from tokenizers import Tokenizer
    ckpt_dir = Path(ckpt_dir)
    ed = load_edge0(edge0_tok_dir)
    res = {"ok": True, "problems": []}

    def bad(msg):
        res["ok"] = False
        res["problems"].append(msg)

    tok = Tokenizer.from_file(str(ckpt_dir / "tokenizer.json"))
    res["vocab_size_with_added"] = tok.get_vocab_size(with_added_tokens=True)
    if res["vocab_size_with_added"] != N_VOCAB:
        bad("vocab size %d != %d" % (res["vocab_size_with_added"], N_VOCAB))
    wrong = [i for i in range(N_VOCAB) if tok.id_to_token(i) != ed["inv"][i]]
    res["id_to_token_mismatches"] = len(wrong)
    if wrong:
        bad("id->token differs for %d ids, e.g. %s" % (len(wrong), [(i, tok.id_to_token(i), ed["inv"][i])
                                                                      for i in wrong[:5]]))
    tj = json.loads((ckpt_dir / "tokenizer.json").read_text(encoding="utf-8"))
    res["merges_equal_edge0"] = [_merge_str(m) for m in tj["model"]["merges"]] == ed["merges"]
    if not res["merges_equal_edge0"]:
        bad("merges differ from Edge0 merges.txt")
    texts = list(SAMPLE_TEXTS)
    if corpus and Path(corpus).is_file():
        texts += _corpus_from_sharegpt(corpus, corpus_max)
    res["corpus_texts"] = len(texts)
    pad_ids = tok.encode("[PAD248100]", add_special_tokens=False).ids
    res["pad_filler_text_ids"] = pad_ids
    if 248100 in pad_ids:
        bad("[PAD248100] text encodes to the filler id")
    roundtrip_bad = sum(1 for t in texts if tok.decode(tok.encode(t, add_special_tokens=False).ids,
                                                         skip_special_tokens=False) != __import__("unicodedata").normalize("NFC", t))
    res["decode_roundtrip_failures"] = roundtrip_bad
    if roundtrip_bad:
        bad("%d texts do not round-trip encode/decode" % roundtrip_bad)
    if ref_dir and (Path(ref_dir) / "tokenizer.json").is_file():
        ref = Tokenizer.from_file(str(Path(ref_dir) / "tokenizer.json"))
        diff = [t for t in texts if tok.encode(t, add_special_tokens=False).ids != ref.encode(t, add_special_tokens=False).ids]
        res["encode_mismatches_vs_ref"] = len(diff)
        if diff:
            bad("encode() differs from the HF reference on %d of %d texts, e.g. %r" % (len(diff), len(texts),
                                                                                   diff[0][:120]))
    # chat template through transformers (what vLLM does)
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(str(ckpt_dir))
    msgs = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]
    prompt = hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    res["think_off_prompt"] = prompt
    if not prompt.endswith(THINK_OFF_SUFFIX):
        bad("thinking-off prompt does not end with %r" % THINK_OFF_SUFFIX)
    res["hf_len"] = len(hf)
    res["hf_eos"] = [hf.eos_token, hf.eos_token_id]
    res["hf_pad"] = [hf.pad_token, hf.pad_token_id]
    if hf.eos_token_id != 248046 or hf.pad_token_id != 248044:
        bad("eos/pad ids %s/%s" % (hf.eos_token_id, hf.pad_token_id))
    ids = hf(prompt, add_special_tokens=False)["input_ids"]
    for sp in (248045, 248046, 248068, 248069):
        if sp not in ids:
            bad("special id %d missing from the encoded chat prompt" % sp)
    res["think_off_prompt_ids"] = ids
    # Strata reference prompts (prompt token counts from the Strata server's usage) and generated tokens (every
    # logprobs token, given as raw bytes, must be one vocabulary entry of the converted tokenizer)
    if strata_ref and Path(strata_ref).is_file():
        sr = json.loads(Path(strata_ref).read_text(encoding="utf-8"))
        b2u = _bytes_to_unicode()
        gen_total, gen_missing = 0, []
        for r in sr.get("records", []):
            for c in ((r.get("logprobs") or {}).get("content") or []):
                if c.get("bytes") is None:
                    continue
                gen_total += 1
                if tok.token_to_id("".join(b2u[b] for b in c["bytes"])) is None:
                    gen_missing.append(c.get("token"))
        res["strata_generated_tokens"] = {"checked": gen_total, "not_in_vocab": len(gen_missing),
                                          "examples": gen_missing[:5]}
        if gen_missing:
            bad("%d Strata-generated tokens are not single vocabulary entries" % len(gen_missing))
        cmp = []
        for r in sr.get("records", []):
            p = hf.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            n = len(hf(p, add_special_tokens=False)["input_ids"])
            want = (r.get("usage") or {}).get("prompt_tokens")
            ids_ref = r.get("prompt_token_ids")
            ent = {"prompt_tokens_vllm": n, "prompt_tokens_strata": want}
            if ids_ref is not None:
                ent["ids_equal"] = hf(p, add_special_tokens=False)["input_ids"] == ids_ref
            cmp.append(ent)
            if (want is not None and want != n) or ent.get("ids_equal") is False:
                bad("strata-ref prompt %d: %s" % (len(cmp) - 1, ent))
        res["strata_ref"] = cmp
    else:
        res["strata_ref"] = "not available (%s)" % strata_ref
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--edge0-tokenizer", required=True)
    ap.add_argument("--ref")
    ap.add_argument("--corpus")
    ap.add_argument("--strata-ref")
    ap.add_argument("--build", action="store_true", help="(re)write the tokenizer files into --ckpt first")
    a = ap.parse_args()
    if a.build:
        os.makedirs(a.ckpt, exist_ok=True)
        print(json.dumps(build(a.edge0_tokenizer, a.ref, a.ckpt), indent=1, ensure_ascii=False)[:3000])
    r = check(a.ckpt, a.edge0_tokenizer, a.ref, a.corpus, strata_ref=a.strata_ref)
    r.pop("think_off_prompt_ids", None)
    print(json.dumps(r, indent=1, ensure_ascii=False))
