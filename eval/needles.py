"""Needle-in-a-haystack recall for LFM3-1.7B in vLLM, stock bf16 KV vs the 800h KV cache.

96 needles = seeds 3..14 x lengths 1K/4K/16K/32K x depth 25/75 %, built exactly like
vllm_lfm3/scripts/kv800_e2e.py:needle_items (same RNG sequence, same haystack, same question).
Greedy, 8 new tokens, hit = the 4-digit code appears in the decoded continuation.
Seeds 3-5 were used while developing the kernels; seeds 6-14 are held out (reported separately).

This file also holds the helpers shared with rollouts.py and concurrency.py (engine setup, corpus, Wilson).

usage: python needles.py --mode {stock,kv800} [--plugin] [--mnbt 2048] [--budget-min 40] [--tag name]
"""
import argparse, json, math, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SNAP = ("/mnt/ssd/huggingface/hub/models--LiquidAI--LFM3-1.7B-Base/snapshots/"
        "7bca5038823c0d0f14efe06439109354a10e014c")
SRC_FILES = [f"/usr/lib64/python3.14/{f}.py" for f in ("argparse", "difflib", "inspect", "typing")]
KEYS = ["blue door", "red train", "old lighthouse", "green kettle", "silver bike", "quiet library", "paper boat", "iron gate",
        "copper lamp", "north tower", "wooden bridge", "glass jar", "yellow kite", "stone well", "black piano", "white tent",
        "orange van", "brass key", "tin drum", "velvet chair", "marble stair", "cedar box", "linen map", "amber clock",
        "frozen lake", "hidden cave", "rusty anchor", "golden bell", "purple scarf", "broken compass", "narrow alley", "clay pot"]
DEV_SEEDS = {"needles": (3, 4, 5), "rollouts": (0, 1)}       # seen during kernel development


# ---------------------------------------------------------------- shared helpers
def base_parser(desc, mnbt, util, max_model_len):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--mode", choices=("stock", "kv800"), required=True, help="stock bf16 KV or the 800h KV cache")
    p.add_argument("--plugin", action="store_true",
                   help="rely on the installed package's vllm.general_plugins entry point instead of the in-tree "
                        "vllm_lfm3.register() call")
    p.add_argument("--mnbt", type=int, default=mnbt, help="max_num_batched_tokens (= prefill chunk size)")
    p.add_argument("--util", type=float, default=util, help="gpu_memory_utilization")
    p.add_argument("--max-model-len", type=int, default=max_model_len)
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument("--attn-backend", default="", help="vLLM attention_backend, e.g. TRITON_ATTN (stock on this image)")
    p.add_argument("--tag", default="", help="suffix for the output file name")
    p.add_argument("--zero-kv", action="store_true",
                   help="zero every KV cache tensor before each generate call (no request is alive then), so no request "
                        "reads blocks recycled from an earlier call; isolates a stale-block dependence of the KV path")
    return p


def setup_env(args):
    os.environ["LFM3_KV800"] = "1" if args.mode == "kv800" else "0"
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")       # in-process engine (scheduler introspection)


def make_llm(args):
    setup_env(args)
    ZERO_KV[0] = args.zero_kv
    if not args.plugin:
        import vllm_lfm3
        vllm_lfm3.register()
    from vllm import LLM
    llm = LLM(model=SNAP, dtype="bfloat16", enforce_eager=True, enable_prefix_caching=False,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.util, trust_remote_code=True,
              max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.mnbt, compilation_config={"mode": 0},
              disable_log_stats=False, **({"attention_backend": args.attn_backend} if args.attn_backend else {}))
    PATH_INFO.update(model_path(llm))
    if args.plugin:                          # never fall back to the in-tree code silently
        bad = [m for m in ("vllm_lfm3", "lfm3kv") if m in sys.modules]
        if bad or not PATH_INFO["model_module"].startswith("lfm3_kv800."):
            raise RuntimeError(f"--plugin: model not served by the lfm3_kv800 package: {PATH_INFO}, in-tree modules {bad}")
    print(f"model path: {PATH_INFO}", flush=True)
    return llm


PATH_INFO = {}


def model_path(llm):
    """Which code serves the model: the class of the loaded model and the package file it came from."""
    info = {}
    try:
        runner = engine_core(llm).model_executor.driver_worker.worker.model_runner
        model = runner.get_model() if hasattr(runner, "get_model") else runner.model
        info["model_module"] = type(model).__module__
        info["model_file"] = sys.modules[type(model).__module__].__file__
    except Exception as e:                   # fall back to the registry entry
        from vllm import ModelRegistry
        info["registry_error"] = repr(e)
        try:
            info["model_module"] = ModelRegistry.models["Lfm3ForCausalLM"].module_name
        except Exception as e2:
            info["model_module"] = f"unknown ({e2!r})"
    info["in_tree_modules_loaded"] = sorted(m for m in ("vllm_lfm3", "lfm3kv") if m in sys.modules)
    try:
        import lfm3_kv800
        info["lfm3_kv800_file"] = lfm3_kv800.__file__
    except ImportError:
        info["lfm3_kv800_file"] = None
    return info


def engine_core(llm):
    return llm.llm_engine.engine_core.engine_core


def corpus(tok):
    src = "".join(open(f).read() for f in SRC_FILES)
    return tok(src, add_special_tokens=False)["input_ids"]


ZERO_KV = [False]


def zero_kv(llm):
    """Zero all KV cache tensors bound to the model's attention/conv layers (in-process engine)."""
    import torch
    ctx = engine_core(llm).vllm_config.compilation_config.static_forward_context
    seen, nbytes = set(), 0

    def z(t):
        nonlocal nbytes
        if isinstance(t, torch.Tensor):
            if t.numel() and t.data_ptr() not in seen:
                seen.add(t.data_ptr()); t.zero_(); nbytes += t.numel() * t.element_size()
        elif isinstance(t, (list, tuple)):
            for x in t:
                z(x)
    for m in ctx.values():
        z(getattr(m, "kv_cache", None))
    torch.xpu.synchronize()
    if not nbytes:
        raise RuntimeError("--zero-kv: no KV cache tensor found")
    return nbytes


def gen(llm, prompts, n):
    if ZERO_KV[0]:
        print(f"zeroed {zero_kv(llm) / 2 ** 30:.2f} GiB of KV views", flush=True)
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    sp = SamplingParams(temperature=0, max_tokens=n, ignore_eos=True)
    return llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, use_tqdm=False)


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt_ci(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.1f}% (95% CI {100 * lo:.1f}-{100 * hi:.1f}%)" if n else "0/0"


def run_info(args, llm):
    import vllm, torch
    info = {"mode": args.mode, "plugin": args.plugin, "mnbt": args.mnbt, "util": args.util,
            "max_model_len": args.max_model_len, "max_num_seqs": args.max_num_seqs, "attn_backend": args.attn_backend or "default", "vllm": vllm.__version__,
            "torch": torch.__version__, "backend": os.environ.get("LFM3KV_BACKEND", "sycl") if args.mode == "kv800" else "vllm attn",
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "model_path": dict(PATH_INFO)}
    try:
        kvc = engine_core(llm).scheduler.kv_cache_config
        info["kv_num_blocks"] = kvc.num_blocks
        info["kv_groups"] = [f"{type(g.kv_cache_spec).__name__}(bs={getattr(g.kv_cache_spec, 'block_size', '?')}, "
                             f"page={getattr(g.kv_cache_spec, 'page_size_bytes', '?')}, "
                             f"win={getattr(g.kv_cache_spec, 'sliding_window', None)}) x{len(g.layer_names)}"
                             for g in kvc.kv_cache_groups]
    except Exception as e:  # informational only
        info["kv_groups"] = f"not readable: {e!r}"
    return info


def write_json(name, obj):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"wrote {path}", flush=True)


def _haystack_item(tok, IDS, rnd, key, n, d):
    # identical to kv800_e2e.needle_items / rollout_items (body of the loop)
    code, start = str(rnd.randint(1000, 9999)), rnd.randint(0, len(IDS) - n - 1)
    body = IDS[start:start + n]; cut = int(d * n)
    text = tok.decode(body[:cut]) + f"\n# The secret number of the {key} is {code}.\n" + tok.decode(body[cut:])
    q = f"\n\nQuestion: What is the secret number of the {key}?\nAnswer: The secret number of the {key} is"
    return code, tok(text)["input_ids"], tok(q, add_special_tokens=False)["input_ids"]


def needle_items(tok, IDS, seed, lengths=(1024, 4096, 16384, 32768)):
    rnd = random.Random(seed); out = []; keys = rnd.sample(KEYS, 8)
    for i, (n, d) in enumerate([(n, d) for n in lengths for d in (0.25, 0.75)]):
        code, ctx, q = _haystack_item(tok, IDS, rnd, keys[i], n, d)
        out.append({"id": f"s{seed}:{n // 1024}K@{int(d * 100)}%", "seed": seed, "len": n, "depth": d, "code": code,
                    "ctx": ctx, "q": q})
    return out


def rollout_items(tok, IDS, seed, lengths=(8192, 16384)):
    rnd = random.Random(100 + seed); out = []; keys = rnd.sample(KEYS, 8)
    for i, (n, d) in enumerate([(n, d) for n in lengths for d in (0.25, 0.75)]):
        code, ctx, q = _haystack_item(tok, IDS, rnd, keys[i], n, d)
        out.append({"id": f"s{seed}:{n // 1024}K@{int(d * 100)}%", "seed": seed, "len": n, "depth": d, "code": code,
                    "ctx": ctx, "q": q})
    return out


def balanced_subset(items, k):
    """First k items in seed order (k rounded down to even: both depths of a seed stay together)."""
    k = max(0, min(len(items), k - k % 2))
    return items[:k]


# ---------------------------------------------------------------- needles
def main():
    p = base_parser(__doc__.splitlines()[0], mnbt=2048, util=0.75, max_model_len=33792 + 1024)
    p.add_argument("--seeds", default="3-14")
    p.add_argument("--budget-min", type=float, default=0,
                   help="time budget; length buckets run 1K->32K and a bucket is cut to a balanced subset when the "
                        "quadratic extrapolation from the previous bucket would exceed it (0 = no budget, one batch)")
    p.add_argument("--min-per-length", type=int, default=12)
    p.add_argument("--group", type=int, default=0,
                   help="needles per generate call (0 = all in one batch; 1 = one request at a time, so no step mixes "
                        "one request's decode with another's prefill)")
    args = p.parse_args()
    a, b = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(a, b + 1))

    llm = make_llm(args)
    tok = llm.get_tokenizer()
    IDS = corpus(tok)
    items = [it for sd in seeds for it in needle_items(tok, IDS, sd)]
    lengths = sorted({it["len"] for it in items})
    info = run_info(args, llm)
    print(json.dumps(info), flush=True)

    t0 = time.time(); ran = []; timing = {}
    if args.budget_min <= 0:
        grp = args.group or len(items)
        for g in range(0, len(items), grp):
            outs = gen(llm, [it["ctx"] + it["q"] for it in items[g:g + grp]], 8)
            for it, o in zip(items[g:g + grp], outs):
                it["out"] = tok.decode(o.outputs[0].token_ids)
        ran = items; timing["all"] = time.time() - t0
    else:
        prev = None
        for L in lengths:
            bucket = [it for it in items if it["len"] == L]
            if prev is not None:
                pl, pt, pn = prev
                est = pt / pn * (L / pl) ** 2                        # per item, attention cost ~ length^2
                left = args.budget_min * 60 - (time.time() - t0)
                k = max(args.min_per_length, int(left / est)) if est > 0 else len(bucket)
                bucket = balanced_subset(bucket, k)
                print(f"bucket {L}: est {est:.1f}s/item, {left:.0f}s left -> {len(bucket)} items", flush=True)
            tb = time.time()
            outs = gen(llm, [it["ctx"] + it["q"] for it in bucket], 8)
            for it, o in zip(bucket, outs):
                it["out"] = tok.decode(o.outputs[0].token_ids)
            dt = time.time() - tb; timing[str(L)] = dt; prev = (L, dt, len(bucket)); ran += bucket
            print(f"bucket {L}: {len(bucket)} items in {dt:.0f}s", flush=True)

    for it in ran:
        it["hit"] = it["code"] in it["out"]
    hits = sum(it["hit"] for it in ran)
    res = {"info": info, "seeds": seeds, "n": len(ran), "hits": hits, "wilson95": wilson(hits, len(ran)),
           "seconds": time.time() - t0, "timing": timing, "per_length": {}, "per_depth": {}, "dev_vs_heldout": {},
           "items": [{k: it[k] for k in ("id", "seed", "len", "depth", "code", "out", "hit")} |
                     {"prompt_tokens": len(it["ctx"]) + len(it["q"])} for it in ran]}
    for L in lengths:
        sub = [it for it in ran if it["len"] == L]
        res["per_length"][f"{L // 1024}K"] = {"hits": sum(it["hit"] for it in sub), "n": len(sub),
                                               "wilson95": wilson(sum(it["hit"] for it in sub), len(sub))}
    for d in (0.25, 0.75):
        sub = [it for it in ran if it["depth"] == d]
        res["per_depth"][f"{int(d * 100)}%"] = {"hits": sum(it["hit"] for it in sub), "n": len(sub)}
    for name, cond in (("dev seeds 3-5", lambda s: s in DEV_SEEDS["needles"]),
                       ("held-out seeds", lambda s: s not in DEV_SEEDS["needles"])):
        sub = [it for it in ran if cond(it["seed"])]
        res["dev_vs_heldout"][name] = {"hits": sum(it["hit"] for it in sub), "n": len(sub)}

    tag = (f"needles_{args.mode}_c{args.mnbt}" + (f"_g{args.group}" if args.group else "") + ("_zkv" if args.zero_kv else "")
           + (f"_{args.tag}" if args.tag else ""))
    res["zero_kv"] = args.zero_kv
    res["group"] = args.group
    print(f"RESULT {tag} | recall {fmt_ci(hits, len(ran))} | per length "
          + ", ".join(f"{k} {v['hits']}/{v['n']}" for k, v in res["per_length"].items())
          + f" | {res['dev_vs_heldout']} | {res['seconds']:.0f}s", flush=True)
    for it in ran:
        if not it["hit"]:
            print(f"  miss {it['id']} code {it['code']} got {it['out']!r}", flush=True)
    write_json(tag, res)


if __name__ == "__main__":
    main()
