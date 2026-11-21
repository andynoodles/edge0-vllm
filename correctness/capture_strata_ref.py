#!/usr/bin/env python3
"""Capture greedy reference outputs from the Strata server (edge0-infer serve, :8124) for vLLM correctness checks.

Each prompt is sent alone (Strata's solo path, which returns logprobs), thinking off, temperature 0.
Output: JSON with the request parameters, generated text and per-token logprobs (top 5) as Strata returns them.

    python3 -I capture_strata_ref.py [out.json] [base_url]
"""
import json
import sys
import urllib.request

OUT = sys.argv[1] if len(sys.argv) > 1 else "strata-ref.json"
URL = (sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8124") + "/v1/chat/completions"
MAX_TOKENS = 48

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "用繁體中文簡單說明什麼是「不當得利」，兩句話以內。",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Compute 37 * 43 and show the steps briefly.",
    "List three differences between TCP and UDP.",
    "請把這句話翻譯成英文：今天下午三點在會議室開會。",
    "Explain in two sentences why the sky is blue.",
    "Translate to French: The weather is nice today, let's go for a walk.",
]


def ask(content):
    body = {"model": "edge0_145b", "messages": [{"role": "user", "content": content}], "max_tokens": MAX_TOKENS,
            "temperature": 0, "top_k": 1, "reasoning_effort": "none", "logprobs": True, "top_logprobs": 5,
            "stream": False}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return body, json.load(r)


records = []
for p in PROMPTS:
    body, resp = ask(p)
    ch = resp["choices"][0]
    records.append({"messages": body["messages"], "text": ch["message"].get("content"),
                    "finish_reason": ch.get("finish_reason"), "logprobs": ch.get("logprobs"),
                    "usage": resp.get("usage")})
    print(f"{len(records)}: {repr((ch['message'].get('content') or '')[:80])}")

meta = {
    "engine": "edge0-strata via edge0-infer serve (Edge0-145B-NVFP4, du=q8, ple-io=direct, MTP spec 4)",
    "sampling": {"temperature": 0, "top_k": 1, "max_tokens": MAX_TOKENS, "thinking": "off (reasoning_effort none)"},
    "vllm_equivalent": {"temperature": 0, "max_tokens": MAX_TOKENS,
                        "chat_template_kwargs": {"enable_thinking": False}, "logprobs": True, "top_logprobs": 5},
    "chat_template": "/home/andychen/models/Edge0-145B-NVFP4/tokenizer/chat_template.jinja",
    "tokenizer_hf": "/home/andychen/work/edge0-bench/tokenizer-qwen38 (identical ids to Edge0's vocab; Edge0's "
                    "tokenizer/tokenizer.json is GGUF metadata, not a HF tokenizer; vocab padded with [PADnnnnnn] to 248320)",
    "note": "Strata decoding is not bit-repeatable; expect close, not identical, agreement.",
}
json.dump({"meta": meta, "records": records}, open(OUT, "w"), ensure_ascii=False, indent=1)
print("wrote", OUT)
