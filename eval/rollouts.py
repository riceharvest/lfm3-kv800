"""Rollout recall for LFM3-1.7B in vLLM, stock bf16 KV vs the 800h KV cache.

32 rollouts = seeds 0..7 x context 8K/16K x needle depth 25/75 %, built exactly like
vllm_lfm3/scripts/kv800_e2e.py:rollout_items. Step 1: the needle context alone, 1024 greedy tokens generated
(ignore_eos). Step 2: context + those 1024 tokens + the question as one prompt, 8 greedy tokens; hit = the code appears.
Prefix caching is off, so step 2 re-prefills the whole sequence in chunks of --mnbt tokens; the question chunk reads the
earlier chunks (needle included) back from the KV cache under test.
Seeds 0-1 were used while developing the kernels; seeds 2-7 are held out (reported separately).

usage: python rollouts.py --mode {stock,kv800} [--plugin] [--mnbt 2048] [--group 32] [--budget-min 25] [--tag name]
"""
import json, time

from needles import (DEV_SEEDS, base_parser, corpus, fmt_ci, gen, make_llm, rollout_items, run_info, wilson,
                     write_json)


def main():
    p = base_parser(__doc__.splitlines()[0], mnbt=2048, util=0.75, max_model_len=33792 + 1024)
    p.add_argument("--seeds", default="0-7")
    p.add_argument("--gen", type=int, default=1024)
    p.add_argument("--group", type=int, default=32, help="rollouts per batched generate call")
    p.add_argument("--budget-min", type=float, default=0,
                   help="stop starting new groups once the next group (timed like the previous one) would exceed this")
    args = p.parse_args()
    a, b = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(a, b + 1))

    llm = make_llm(args)
    tok = llm.get_tokenizer()
    IDS = corpus(tok)
    items = [it for sd in seeds for it in rollout_items(tok, IDS, sd)]
    info = run_info(args, llm)
    print(json.dumps(info), flush=True)

    t0 = time.time(); ran = []; last = 0.0
    for g in range(0, len(items), args.group):
        if args.budget_min > 0 and ran and (time.time() - t0) + 1.1 * last > args.budget_min * 60:
            print(f"budget: stopping after {len(ran)} rollouts", flush=True)
            break
        grp = items[g:g + args.group]; tg = time.time()
        g1 = gen(llm, [it["ctx"] for it in grp], args.gen)
        g2 = gen(llm, [it["ctx"] + list(o.outputs[0].token_ids) + it["q"] for it, o in zip(grp, g1)], 8)
        for it, o1, o2 in zip(grp, g1, g2):
            it["gen_head"] = tok.decode(o1.outputs[0].token_ids)[:200]
            it["out"] = tok.decode(o2.outputs[0].token_ids)
            it["hit"] = it["code"] in it["out"]
        last = time.time() - tg; ran += grp
        print(f"group {g // args.group}: {len(grp)} rollouts in {last:.0f}s, hits {sum(it['hit'] for it in grp)}", flush=True)

    hits = sum(it["hit"] for it in ran)
    res = {"info": info, "seeds": seeds, "gen_tokens": args.gen, "n": len(ran), "hits": hits,
           "wilson95": wilson(hits, len(ran)), "seconds": time.time() - t0, "per_length": {}, "dev_vs_heldout": {},
           "items": [{k: it[k] for k in ("id", "seed", "len", "depth", "code", "out", "hit", "gen_head")} for it in ran]}
    for L in sorted({it["len"] for it in ran}):
        sub = [it for it in ran if it["len"] == L]
        res["per_length"][f"{L // 1024}K"] = {"hits": sum(it["hit"] for it in sub), "n": len(sub)}
    for name, cond in (("dev seeds 0-1", lambda s: s in DEV_SEEDS["rollouts"]),
                       ("held-out seeds", lambda s: s not in DEV_SEEDS["rollouts"])):
        sub = [it for it in ran if cond(it["seed"])]
        res["dev_vs_heldout"][name] = {"hits": sum(it["hit"] for it in sub), "n": len(sub)}

    tag = (f"rollouts_{args.mode}_c{args.mnbt}" + (f"_g{args.group}" if args.group not in (32, len(items)) else "")
           + ("_zkv" if args.zero_kv else "") + (f"_{args.tag}" if args.tag else ""))
    res["group"] = args.group; res["zero_kv"] = args.zero_kv
    print(f"RESULT {tag} | recall {fmt_ci(hits, len(ran))} | per length "
          + ", ".join(f"{k} {v['hits']}/{v['n']}" for k, v in res["per_length"].items())
          + f" | {res['dev_vs_heldout']} | {res['seconds']:.0f}s", flush=True)
    for it in ran:
        if not it["hit"]:
            print(f"  miss {it['id']} code {it['code']} got {it['out']!r}", flush=True)
    write_json(tag, res)


if __name__ == "__main__":
    main()
