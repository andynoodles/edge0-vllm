#!/usr/bin/env python3
"""Compare vLLM (edge0-vllm plugin, :8125) against Strata's greedy reference outputs (strata-ref.json).

Two comparisons per prompt, standard library only (runs on the 5090 host with the system python3):

1. Free-running greedy: the same request Strata got (temperature 0, thinking off, max_tokens 48, logprobs top 5).
   Reports the common prefix of generated tokens (overall and within the first 32), and at the first divergence both
   engines' top-5, so a near-tie (Strata is not bit-repeatable) can be told apart from a real bug.
2. Teacher-forced: the chat prompt + Strata's generated tokens are sent as one token-id prompt to /v1/completions
   with prompt_logprobs=5. For every position of Strata's continuation this gives vLLM's logprob of Strata's token
   and vLLM's own top-1, i.e. agreement without error compounding. Reports top-1 agreement, mean/max |dlogprob| of
   Strata's chosen token, and a KL estimate over the shared top-5 support.

Token identity is by raw bytes (Strata returns OpenAI 'bytes'); Strata's token bytes are mapped to ids through the
checkpoint's byte-level BPE vocab.json.

    python3 -I compare_strata.py [--ref strata-ref.json] [--url http://127.0.0.1:8125] [--vocab DIR/vocab.json]
                                 [--out result.json] [--model edge0_145b]
Exit status 0 when every prompt passes the agreement thresholds (see PASS rules at the end of main()).
"""
import argparse
import json
import math
import sys
import urllib.request

FIRST = 32


def bytes_to_unicode():
    """GPT-2 byte-level BPE byte -> printable char mapping (same as transformers / tokenizers ByteLevel)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def post(url, body, timeout=600):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def tok_bytes(entry):
    b = entry.get("bytes")
    if b is not None:
        return bytes(b)
    return entry["token"].encode("utf-8")


def show(b):
    return repr(b.decode("utf-8", errors="backslashreplace"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="/home/andychen/work/edge0-bench/strata-ref.json")
    ap.add_argument("--url", default="http://127.0.0.1:8125")
    ap.add_argument("--model", default="edge0_145b")
    ap.add_argument("--vocab", default="/home/andychen/models/Edge0-145B-vllm/vocab.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-teacher", action="store_true")
    args = ap.parse_args()

    ref = json.load(open(args.ref))
    vocab = json.load(open(args.vocab))
    b2u = bytes_to_unicode()

    def bytes_to_id(b):
        return vocab["".join(b2u[x] for x in b)]

    results = []
    for i, rec in enumerate(ref["records"]):
        msgs = rec["messages"]
        s_lp = rec["logprobs"]["content"]
        s_toks = [tok_bytes(e) for e in s_lp]

        # ---- 1. free-running greedy
        body = {"model": args.model, "messages": msgs, "max_tokens": 48, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}, "logprobs": True, "top_logprobs": 5}
        resp = post(args.url + "/v1/chat/completions", body)
        ch = resp["choices"][0]
        v_lp = ch["logprobs"]["content"]
        v_toks = [tok_bytes(e) for e in v_lp]
        n = 0
        while n < min(len(s_toks), len(v_toks)) and s_toks[n] == v_toks[n]:
            n += 1
        cmp_len = min(len(s_toks), len(v_toks))
        dl = [abs(v_lp[k]["logprob"] - s_lp[k]["logprob"]) for k in range(n)]
        r = {"i": i, "prompt": msgs[-1]["content"][:60], "strata_text": rec["text"], "vllm_text": ch["message"]["content"],
             "strata_tokens": len(s_toks), "vllm_tokens": len(v_toks), "common_prefix": n,
             "common_prefix_first32": min(n, FIRST), "compare_len": cmp_len,
             "first32_target": min(FIRST, cmp_len),
             "free_mean_abs_dlogprob": (sum(dl) / len(dl)) if dl else None, "free_max_abs_dlogprob": max(dl) if dl else None,
             "finish_strata": rec["finish_reason"], "finish_vllm": ch["finish_reason"]}
        if n < cmp_len:
            s_top = [(show(tok_bytes(e)), round(e["logprob"], 3)) for e in s_lp[n]["top_logprobs"]]
            v_top = [(show(tok_bytes(e)), round(e["logprob"], 3)) for e in v_lp[n]["top_logprobs"]]
            r["divergence"] = {"pos": n, "strata": show(s_toks[n]), "vllm": show(v_toks[n]), "strata_top5": s_top,
                               "vllm_top5": v_top,
                               "strata_margin": round(s_lp[n]["top_logprobs"][0]["logprob"] - s_lp[n]["top_logprobs"][1]["logprob"], 3),
                               "vllm_margin": round(v_lp[n]["top_logprobs"][0]["logprob"] - v_lp[n]["top_logprobs"][1]["logprob"], 3)}

        # ---- 2. teacher-forced on Strata's tokens
        if not args.no_teacher:
            tk = post(args.url + "/tokenize", {"model": args.model, "messages": msgs, "add_generation_prompt": True,
                                               "chat_template_kwargs": {"enable_thinking": False}})
            p_ids = tk["tokens"]
            g_ids = [bytes_to_id(b) for b in s_toks]
            comp = post(args.url + "/v1/completions", {"model": args.model, "prompt": p_ids + g_ids, "max_tokens": 1,
                                                       "temperature": 0, "prompt_logprobs": 5})
            plp = comp["choices"][0]["prompt_logprobs"]   # list (len prompt) of {id: {logprob, rank, decoded_token}} | None
            agree = agree_top1 = nongreedy = 0
            dls, kls = [], []
            worst = None
            for k, gid in enumerate(g_ids):
                d = plp[len(p_ids) + k]
                d = {int(t): v for t, v in d.items()}
                v_top1 = min(d.items(), key=lambda kv: kv[1].get("rank", 99))[0]
                agree += int(v_top1 == gid)
                s_top1 = bytes_to_id(tok_bytes(s_lp[k]["top_logprobs"][0]))
                agree_top1 += int(v_top1 == s_top1)
                nongreedy += int(s_top1 != gid)
                v_l = d[gid]["logprob"]
                s_l = s_lp[k]["logprob"]
                dls.append(abs(v_l - s_l))
                if worst is None or dls[-1] > worst[1]:
                    worst = (k, dls[-1], show(s_toks[k]), round(s_l, 3), round(v_l, 3))
                # KL(strata || vllm) over Strata's top-5 support (tokens vLLM did not return get vLLM's 5th logprob)
                v_by_bytes = {}
                for t, v in d.items():
                    v_by_bytes[t] = v["logprob"]
                floor = min(v_by_bytes.values())
                kl = 0.0
                for e in s_lp[k]["top_logprobs"]:
                    try:
                        tid = bytes_to_id(tok_bytes(e))
                    except KeyError:
                        continue
                    ps = math.exp(e["logprob"])
                    kl += ps * (e["logprob"] - v_by_bytes.get(tid, floor))
                kls.append(kl)
            r.update({"tf_positions": len(g_ids), "tf_top1_agree": agree, "tf_top1_agree_strata_argmax": agree_top1,
                      "strata_nongreedy_picks": nongreedy, "tf_mean_abs_dlogprob": sum(dls) / len(dls),
                      "tf_max_abs_dlogprob": max(dls), "tf_worst": worst, "tf_mean_kl_top5": sum(kls) / len(kls),
                      "prompt_tokens_vllm": len(p_ids), "prompt_tokens_strata": rec["usage"]["prompt_tokens"]})
        results.append(r)

        print(f"[{i}] {r['prompt']!r}")
        print(f"   free: common prefix {n}/{cmp_len} (first32: {r['common_prefix_first32']}/{r['first32_target']}), "
              f"mean|dlp| {r['free_mean_abs_dlogprob'] if dl else float('nan'):.4f}, "
              f"finish strata={r['finish_strata']} vllm={r['finish_vllm']}")
        if "divergence" in r:
            dv = r["divergence"]
            print(f"   diverge @{dv['pos']}: strata {dv['strata']} (margin {dv['strata_margin']}) vs vllm {dv['vllm']} "
                  f"(margin {dv['vllm_margin']})")
            print(f"      strata top5 {dv['strata_top5']}")
            print(f"      vllm   top5 {dv['vllm_top5']}")
        if "tf_positions" in r:
            print(f"   teacher-forced: vllm top1 == strata token {r['tf_top1_agree']}/{r['tf_positions']}, == strata "
                  f"argmax {r['tf_top1_agree_strata_argmax']}/{r['tf_positions']} (strata picked a non-argmax token "
                  f"{r['strata_nongreedy_picks']}x), mean|dlp| "
                  f"{r['tf_mean_abs_dlogprob']:.4f}, max {r['tf_max_abs_dlogprob']:.3f} at {r['tf_worst']}, "
                  f"KL(top5) {r['tf_mean_kl_top5']:.4f}; prompt tokens vllm={r['prompt_tokens_vllm']} "
                  f"strata={r['prompt_tokens_strata']}")
        print(f"   strata: {rec['text'][:110]!r}")
        print(f"   vllm  : {ch['message']['content'][:110]!r}")

    # PASS rules. Strata's MTP-speculative greedy decoding sometimes emits a token that is NOT the argmax of the
    # logprobs it reports (strata_nongreedy_picks), so a free-running divergence is explained when, after >= 4 equal
    # tokens, (a) Strata's pick was not its own argmax, or (b) vLLM picked Strata's argmax, or (c) it is a near-tie in
    # Strata's distribution (top-1 - top-2 < 1 nat). Teacher-forced: vLLM's top-1 equals Strata's argmax at >= 90% of
    # the positions.
    ok_all = True
    for r in results:
        ok = True
        if r["common_prefix_first32"] < r["first32_target"]:
            dv = r.get("divergence", {})
            s_argmax = dv.get("strata_top5", [("", 0)])[0][0]
            explained = (dv.get("strata") != s_argmax or dv.get("vllm") == s_argmax
                         or dv.get("strata_margin", 99) < 1.0)
            ok = r["common_prefix"] >= 4 and explained
            r["divergence_explained"] = explained
        if "tf_positions" in r and r["tf_top1_agree_strata_argmax"] < 0.9 * r["tf_positions"]:
            ok = False
        r["pass"] = ok
        ok_all &= ok
    tot = sum(r["common_prefix_first32"] for r in results)
    tgt = sum(r["first32_target"] for r in results)
    summ = {"prompts": len(results), "pass": sum(r["pass"] for r in results), "first32_common": tot,
            "first32_target": tgt}
    if results and "tf_positions" in results[0]:
        summ.update({"tf_top1_agree": sum(r["tf_top1_agree"] for r in results),
                     "tf_top1_agree_strata_argmax": sum(r["tf_top1_agree_strata_argmax"] for r in results),
                     "strata_nongreedy_picks": sum(r["strata_nongreedy_picks"] for r in results),
                     "tf_positions": sum(r["tf_positions"] for r in results),
                     "tf_mean_abs_dlogprob": sum(r["tf_mean_abs_dlogprob"] * r["tf_positions"] for r in results)
                     / sum(r["tf_positions"] for r in results),
                     "tf_mean_kl_top5": sum(r["tf_mean_kl_top5"] * r["tf_positions"] for r in results)
                     / sum(r["tf_positions"] for r in results)})
    print("SUMMARY", json.dumps(summ))
    if args.out:
        json.dump({"summary": summ, "results": results}, open(args.out, "w"), ensure_ascii=False, indent=1)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
