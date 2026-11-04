"""How many concurrent 32K-token requests fit in the B70's KV cache: stock bf16 KV vs the 800h KV cache.

1. Capacity (both modes): build the engine at --util (0.85), read vLLM's KV cache config and compute, with vLLM's own
   function (get_max_concurrency_for_kv_cache_config, the number behind the "Maximum concurrency" log line), the
   concurrency for requests of --ctx + --new tokens (32768 + 64). Also reported: the steady-state number, where every
   request holds its full-attention pages and only its windows of the sliding/exact layers (vLLM frees blocks that fall
   out of a window; its admission gate reserves window + one prefill chunk per request, which is why its stated number
   is lower), and the KV bytes per token / per request of each layer type.
2. Confirmation (--confirm, kv800): N requests with a --ctx-token prompt, all submitted at once. vLLM decodes a request as
   soon as its prefill ends, so with only 64 new tokens the first request would finish before the last one is prefilled.
   Each request therefore generates G = new + ceil(N * ctx / mnbt) + 64 tokens: enough that all N are resident with their
   full 32K prompt at the same time while every one of them is still decoding. The scheduler is instrumented: per step
   the number of running requests whose prompt is fully in the cache, KV usage, and every preemption.

usage: python concurrency.py --mode {stock,kv800} [--plugin] [--confirm none|stated|steady|<N>] [--tag name]
"""
import json, math, time

from needles import base_parser, corpus, engine_core, make_llm, run_info, write_json


def cdiv(a, b):
    return -(-a // b)


def group_rows(vc, kvc, L, in_flight):
    """Per KV cache group: kind, layers, page bytes, admission and steady-state blocks per request of L tokens."""
    rows = []
    for g in kvc.kv_cache_groups:
        s = g.kv_cache_spec; name = type(s).__name__
        page = s.page_size_bytes; bs = getattr(s, "block_size", None)
        real = getattr(s, "real_page_size_bytes", page)
        win = getattr(s, "sliding_window", None)
        if name == "MambaSpec":
            kind = "conv state"; adm = steady = cdiv(s.max_memory_usage_bytes(vc), page)
        elif win:
            kind = f"window {win}"
            adm = s.max_admission_blocks_per_request(max_in_flight_tokens=in_flight, max_model_len=L)
            steady = s.max_admission_blocks_per_request(max_in_flight_tokens=1, max_model_len=L)
        else:
            dt = str(getattr(s, "dtype", ""))
            kind = "800h quant" if "uint8" in dt else "full attention bf16"
            adm = steady = cdiv(L, bs)
        rows.append({"spec": name, "kind": kind, "layers": list(g.layer_names), "n_layers": len(g.layer_names),
                     "block_size": bs, "page_bytes": page, "content_page_bytes": real, "window": win,
                     "admission_blocks": adm, "steady_blocks": steady})
    return rows


def layer_index(name):
    for part in name.split("."):
        if part.isdigit():
            return int(part)
    return -1


def capacity(args, llm):
    from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config
    core = engine_core(llm); vc = core.vllm_config; kvc = core.scheduler.kv_cache_config
    L = args.ctx + args.new
    old = vc.model_config.max_model_len
    in_flight = vc.max_in_flight_tokens
    rows = group_rows(vc, kvc, L, in_flight)
    try:                                   # vLLM's own formula, evaluated for L-token requests
        vc.model_config.max_model_len = L
        stated = get_max_concurrency_for_kv_cache_config(vc, kvc); stated_src = "vllm"
    except Exception as e:                 # same formula by hand
        print(f"vLLM concurrency function failed ({e!r}); using num_blocks / admission blocks", flush=True)
        stated = kvc.num_blocks / sum(r["admission_blocks"] for r in rows); stated_src = "manual"
    finally:
        vc.model_config.max_model_len = old
    pool = max(t.offset + t.size for t in kvc.kv_cache_tensors)
    per_block = pool / kvc.num_blocks
    adm = sum(r["admission_blocks"] for r in rows); steady = sum(r["steady_blocks"] for r in rows)
    # steady state with the admission gate: N-1 requests at steady state + the newest one passing the gate
    n_steady = (kvc.num_blocks - adm) // steady + 1
    kinds = {}
    for r in rows:                         # per layer type, content bytes (allocated bytes in brackets)
        k = kinds.setdefault(r["kind"], {"layers": [], "page_bytes": r["page_bytes"],
                                         "content_page_bytes": r["content_page_bytes"], "block_size": r["block_size"],
                                         "window": r["window"]})
        k["layers"] += sorted(layer_index(n) for n in r["layers"])
    for k, v in kinds.items():
        n = len(v["layers"])
        if k in ("800h quant", "full attention bf16"):
            v["bytes_per_token_per_layer"] = v["content_page_bytes"] / v["block_size"]
            v["alloc_bytes_per_token_per_layer"] = v["page_bytes"] / v["block_size"]
            v["bytes_per_token_all_layers"] = n * v["bytes_per_token_per_layer"]
            v["alloc_bytes_per_token_all_layers"] = n * v["alloc_bytes_per_token_per_layer"]
        elif k.startswith("window"):
            per_tok = v["content_page_bytes"] / v["block_size"]
            v["bytes_per_token_per_layer"] = per_tok
            v["fixed_bytes_per_request_all_layers"] = n * v["window"] * per_tok
        else:
            v["fixed_bytes_per_request_all_layers"] = n * v["content_page_bytes"]
            v["alloc_bytes_per_request_all_layers"] = n * v["page_bytes"]
    res = {"request_tokens": L, "num_blocks": kvc.num_blocks, "pool_bytes": pool, "pool_GiB": pool / 2 ** 30,
           "bytes_per_block_id": per_block, "max_in_flight_tokens": in_flight,
           "vllm_logged_max_concurrency": getattr(vc.cache_config, "kv_cache_max_concurrency", None),
           "vllm_logged_kv_tokens": getattr(vc.cache_config, "kv_cache_size_tokens", None),
           "vllm_logged_for_max_model_len": old,
           "stated_concurrency": stated, "stated_source": stated_src, "stated_N": math.floor(stated),
           "admission_blocks_per_request": adm, "steady_blocks_per_request": steady,
           "admission_bytes_per_request": adm * per_block, "steady_bytes_per_request": steady * per_block,
           "steady_N": int(n_steady), "groups": rows, "layer_types": kinds}
    return res


def confirm(args, llm, tok, cap, N):
    import random
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    core = engine_core(llm); sched = core.scheduler
    if args.stagger:     # request i is prefilled ~i-th (FCFS): it only has to outlast the N-i prefills after it
        Gs = [args.new + cdiv((N - i) * args.ctx, args.mnbt - N) + 64 + N // 8 for i in range(N)]
    else:
        Gs = [args.new + cdiv(N * args.ctx, args.mnbt) + 64] * N
    G = max(Gs)
    assert args.ctx + G <= args.max_model_len, f"max_model_len {args.max_model_len} < {args.ctx + G}"
    IDS = corpus(tok); rnd = random.Random(7)
    prompts = [IDS[(st := rnd.randint(0, len(IDS) - args.ctx - 1)):st + args.ctx] for _ in range(N)]
    steps = []; pre = []
    orig_sched, orig_pre = sched.schedule, sched._preempt_request

    def schedule(*a, **k):
        out = orig_sched(*a, **k)
        full = sum(1 for r in sched.running if r.num_computed_tokens >= args.ctx)
        steps.append((time.time(), len(sched.running), len(sched.waiting), full, sched.kv_cache_manager.usage))
        return out

    def preempt(request, *a, **k):
        pre.append((time.time(), request.request_id, request.num_computed_tokens))
        return orig_pre(request, *a, **k)

    sched.schedule, sched._preempt_request = schedule, preempt
    sp = [SamplingParams(temperature=0, max_tokens=g, ignore_eos=True) for g in Gs]
    t0 = time.time()
    try:
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, use_tqdm=False)
    finally:
        sched.schedule, sched._preempt_request = orig_sched, orig_pre
    wall = time.time() - t0
    full_max = max(s[3] for s in steps)
    all_steps = [s for s in steps if s[3] == N]
    usage_peak = max(s[4] for s in steps)
    n_gen = [len(o.outputs[0].token_ids) for o in outs]
    try:
        import torch
        free, total = torch.xpu.mem_get_info()
        mem = {"device_free_GiB_after": free / 2 ** 30, "device_total_GiB": total / 2 ** 30}
    except Exception as e:
        mem = {"err": repr(e)}
    res = {"N": N, "prompt_tokens": args.ctx, "new_tokens_per_request": G, "new_tokens_min": min(Gs), "stagger": args.stagger, "wall_s": wall,
           "preemptions": len(pre), "preempted": pre[:20], "max_running": max(s[1] for s in steps),
           "max_resident_with_full_prompt": full_max, "steps_with_all_N_resident": len(all_steps),
           "seconds_with_all_N_resident": (all_steps[-1][0] - all_steps[0][0]) if all_steps else 0.0,
           "kv_usage_peak": usage_peak, "kv_usage_when_all_N_resident": max((s[4] for s in all_steps), default=None),
           "all_finished_with_G_tokens": n_gen == Gs, "min_tokens_generated": min(n_gen),
           "steps": len(steps), "memory": mem,
           "pass": len(pre) == 0 and full_max == N and n_gen == Gs}
    return res


def sweep_point(args, llm, tok, N):
    """Throughput at concurrency N: N x ctx-token prompts submitted at once, staggered generation lengths so all N are
    resident and decoding together for >= --window steps. Measured: prefill rate until all N have their first token,
    TTFT from the burst start, and aggregate decode tok/s / TPOT over the steps where all N are decoding."""
    import random, statistics
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    core = engine_core(llm); sched = core.scheduler
    Gs = [args.new + cdiv((N - i) * args.ctx, args.mnbt - N) + 64 + N // 8 + args.window for i in range(N)]
    assert args.ctx + max(Gs) <= args.max_model_len, f"max_model_len {args.max_model_len} < {args.ctx + max(Gs)}"
    IDS = corpus(tok); rnd = random.Random(1000 + N)
    prompts = [IDS[(st := rnd.randint(0, len(IDS) - args.ctx - 1)):st + args.ctx] for _ in range(N)]
    steps, pre, first = [], [], {}
    orig_sched, orig_pre = sched.schedule, sched._preempt_request
    t0 = [0.0]

    def schedule(*a, **k):
        out = orig_sched(*a, **k)
        now = time.time(); full = 0
        for r in sched.running:
            if r.num_computed_tokens >= args.ctx:
                full += 1
                if r.request_id not in first:
                    first[r.request_id] = now - t0[0]
        steps.append((now, len(sched.running), len(sched.waiting), full, sched.kv_cache_manager.usage))
        return out

    def preempt(request, *a, **k):
        pre.append(request.request_id)
        return orig_pre(request, *a, **k)

    sched.schedule, sched._preempt_request = schedule, preempt
    sp = [SamplingParams(temperature=0, max_tokens=g, ignore_eos=True) for g in Gs]
    t0[0] = time.time()
    try:
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, use_tqdm=False)
    finally:
        sched.schedule, sched._preempt_request = orig_sched, orig_pre
    wall = time.time() - t0[0]
    win = [s for s in steps if s[3] == N and s[1] == N]
    ttft = sorted(first.values())
    n_gen = [len(o.outputs[0].token_ids) for o in outs]
    res = {"N": N, "prompt_tokens": args.ctx, "new_tokens_min": min(Gs), "new_tokens_max": max(Gs), "wall_s": wall,
           "preemptions": len(pre), "all_tokens_returned": n_gen == Gs, "steps": len(steps),
           "window_steps": len(win), "kv_usage_peak": max(s[4] for s in steps)}
    if len(ttft) == N:
        res.update(ttft_median_s=statistics.median(ttft), ttft_max_s=ttft[-1], ttft_min_s=ttft[0],
                   prefill_tok_s=N * args.ctx / ttft[-1])
    if len(win) >= 2:
        dt = win[-1][0] - win[0][0]
        res.update(window_s=dt, decode_tok_s=N * (len(win) - 1) / dt, tpot_ms=dt / (len(win) - 1) * 1e3,
                   kv_usage_window=max(s[4] for s in win),
                   context_at_window=[args.ctx + args.new + 64 + N // 8, args.ctx + max(Gs)])
    res["pass"] = (len(pre) == 0 and n_gen == Gs and len(win) >= 2 and len(ttft) == N)
    return res


def main():
    p = base_parser(__doc__.splitlines()[0], mnbt=8192, util=0.85, max_model_len=32768 + 4096)
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--new", type=int, default=64)
    p.add_argument("--confirm", default="none", help="none | stated | steady | <N>")
    p.add_argument("--stagger", action="store_true",
                   help="per-request generation length (request i: new + its share of the remaining prefill) instead "
                        "of one length for all; same all-resident condition, shorter decode tail")
    p.add_argument("--sweep", default="", help="comma list of concurrencies for the throughput sweep (kv800)")
    p.add_argument("--window", type=int, default=128, help="sweep: extra decode tokens per request (all-decoding window)")
    p.add_argument("--steady-margin", type=float, default=0.9, help="fraction of steady_N used by --confirm steady")
    p.set_defaults(max_num_seqs=256)        # same for both modes (the sampler warm-up memory scales with it)
    args = p.parse_args()

    llm = make_llm(args)
    tok = llm.get_tokenizer()
    info = run_info(args, llm)
    cap = capacity(args, llm)
    print(json.dumps({k: v for k, v in cap.items() if k not in ("groups",)}, indent=1), flush=True)
    res = {"info": info, "capacity": cap}
    tag = f"concurrency_{args.mode}" + (f"_{args.tag}" if args.tag else "")
    print(f"RESULT {tag} capacity | {cap['request_tokens']}-token requests | pool {cap['pool_GiB']:.2f} GiB, "
          f"{cap['num_blocks']} blocks x {cap['bytes_per_block_id']:.0f} B | vLLM stated {cap['stated_concurrency']:.2f}x "
          f"(admission {cap['admission_blocks_per_request']} blocks/request) | steady-state {cap['steady_N']} "
          f"({cap['steady_blocks_per_request']} blocks/request)", flush=True)
    write_json(tag, res)                    # capacity is saved even if the confirmation crashes

    if args.confirm != "none":
        if args.confirm == "stated":
            N = cap["stated_N"]
        elif args.confirm == "steady":
            N = int(cap["steady_N"] * args.steady_margin)
        else:
            N = int(args.confirm)
        assert N <= args.max_num_seqs, "raise --max-num-seqs"
        c = confirm(args, llm, tok, cap, N)
        res["confirm"] = c
        print(f"RESULT {tag} confirm N={N} | {'PASS' if c['pass'] else 'FAIL'} | preemptions {c['preemptions']} | "
              f"max resident with full {args.ctx}-token prompt {c['max_resident_with_full_prompt']}/{N} for "
              f"{c['steps_with_all_N_resident']} steps ({c['seconds_with_all_N_resident']:.0f}s) | KV usage then "
              f"{c['kv_usage_when_all_N_resident']} | {c['new_tokens_min']}-{c['new_tokens_per_request']} new tokens each | wall {c['wall_s']:.0f}s",
              flush=True)
        write_json(tag, res)

    if args.sweep:
        res["sweep"] = []
        for N in (cap["stated_N"] if x == "stated" else int(x) for x in args.sweep.split(",")):
            assert N <= args.max_num_seqs and N < args.mnbt, "raise --max-num-seqs"
            r = sweep_point(args, llm, tok, N); res["sweep"].append(r)
            print(f"RESULT {tag} sweep c={N} | {'PASS' if r['pass'] else 'FAIL'} | decode {r.get('decode_tok_s', 0):.0f} tok/s "
                  f"aggregate, TPOT {r.get('tpot_ms', 0):.1f} ms over {r['window_steps']} all-decoding steps | prefill "
                  f"{r.get('prefill_tok_s', 0):.0f} tok/s, TTFT median {r.get('ttft_median_s', 0):.1f}s max "
                  f"{r.get('ttft_max_s', 0):.1f}s | KV usage {r.get('kv_usage_window', 0) * 100:.1f}% | preemptions "
                  f"{r['preemptions']} | wall {r['wall_s']:.0f}s", flush=True)
            write_json(tag, res)


if __name__ == "__main__":
    main()
