# %%
"""NVIDIA check for the lfm3-kv800 package (800h paged KV cache for LiquidAI LFM3-1.7B).

Made for a free Kaggle (T4 x2) or Colab (T4) GPU. Runs top to bottom in about 5-25 minutes and needs no paid hardware.
  1. environment report (GPU, CUDA, torch, triton, bf16/fp8 support, tiny Triton feature probes)
  2. kernel parity: the 12 cases of run/kv800_check.py for the triton and torch backends, against a self-contained
     torch reference (copied from lfm3kv.qkv.reference; nothing from lfm3kv is imported)
  3. decode speed: B=64/16/1 x 32K, quantized layer and window layer, against torch SDPA decode in bf16 and fp16
  4. optional end-to-end run: HF LFM3-1.7B-Base with the 800h cache patched into its 6 full-attention layers for decode.
     8 needles at each of 1K, 4K and 16K, compared with stock. Downloads about 3.4 GB from the Hub.
Results print as they come, followed by a summary table. They are also written to nvidia_check_results.json.

Usage:  python check_nvidia.py [--e2e] [--backends triton,torch] [--n 32768] [--device cuda|cpu]
The package must be importable as `lfm3_kv800` (pip install it first; see README_NVIDIA.md)."""
import argparse, copy, gc, inspect, json, math, os, platform, random, sys, tempfile, time, traceback

# A fresh Triton cache per run, so a kernel compiled in an earlier run (for example with the fp8 shim below) is never
# reused under another configuration. Set TRITON_CACHE_DIR yourself to keep compiled kernels between runs.
os.environ.setdefault("TRITON_CACHE_DIR", tempfile.mkdtemp(prefix="kv800_triton_"))

import torch
import torch.nn.functional as F

try:
    import triton, triton.language as tl
except Exception as _e:  # noqa: BLE001  (P100 / CPU-only images may lack triton)
    triton, tl, TRITON_IMPORT_ERROR = None, None, repr(_e)
else:
    TRITON_IMPORT_ERROR = None

MODEL, PIN = "LiquidAI/LFM3-1.7B-Base", "7bca5038823c0d0f14efe06439109354a10e014c"
QL, WL = [6, 10, 14, 22], [2, 18]          # 800h plan: quantized full-attention layers / sinks + window-256 layers
CFG = dict(device="cuda", backends=None, n=32768, batches=(64, 16, 1), e2e=False, e2e_dtype="bf16",
           lengths=(1024, 4096, 16384), needles=8, model=MODEL, revision=PIN, out="nvidia_check_results.json",
           sweep_warps=True)
RES = {}
SHIPPED = {}          # as-shipped parity verdicts; survives a re-run after a shim patched the package in this session


def reset_results():
    RES.clear(); RES.update(env={}, probes={}, parity=[], speed=[], sdpa=[], e2e={}, notes=[])


reset_results()


def log(*a):
    print(*a, flush=True)


def note(s):
    RES["notes"].append(s); log("NOTE:", s)


# %%
# ---------------------------------------------------------------- package adapter (the only place that touches the API)
# Assumed API (lfm3_kv800.kv800, mirroring lfm3kv/kv800.py): PagedStore(batch, max_len, device, window=0, ...) with
# .H, .append(k, v), .paged(), .qpool/.epool/.sk/.sv/.N/.NQ/.n/.nq; decode(P, q, cos_t, sin_t, H[, spb, target]);
# quantize_blocks(k, v, H); constants BLK/LAG/NSINK/WIN/SUB/*_O/F8; backend picked by set_backend(name) if the package
# has one, else the module attribute BACKEND ("sycl" / "triton" / "torch" / "auto"), else env LFM3KV_BACKEND.
CONST_DEFAULTS = dict(BLK=256, LAG=128, NSINK=4, WIN=256, SUB=51200, KC_O=0, VB_O=32768, VLO_O=49152, VSC_O=49664,
                      KLO_O=50176, KSC_O=50688)


class Pkg:
    def __init__(self):
        try:
            import lfm3_kv800
        except Exception as e:
            raise RuntimeError(f"cannot import lfm3_kv800 ({e!r}); install the package first (README_NVIDIA.md)")
        self.root = lfm3_kv800
        try:
            from lfm3_kv800 import kv800
        except Exception as e:
            raise RuntimeError(f"cannot import lfm3_kv800.kv800 ({e!r}); package exports: {dir(lfm3_kv800)}")
        self.kv = kv800
        miss = [n for n in ("decode", "PagedStore") if not hasattr(kv800, n)]
        if miss:
            raise RuntimeError(f"lfm3_kv800.kv800 lacks {miss}; it has: {[n for n in dir(kv800) if not n.startswith('__')]}")
        self.C = {k: getattr(kv800, k, v) for k, v in CONST_DEFAULTS.items()}
        for k, v in CONST_DEFAULTS.items():
            if getattr(kv800, k, v) != v:
                note(f"package constant {k}={getattr(kv800, k)} differs from the reference value {v}")
        self.takes_H = "H" in inspect.signature(kv800.decode).parameters
        self.version = getattr(lfm3_kv800, "__version__", "?")

    def set_backend(self, name):
        for obj in (self.kv, self.root):
            f = getattr(obj, "set_backend", None)
            if callable(f):
                f(name); break
        else:
            self.kv.BACKEND = name
        os.environ["LFM3KV_BACKEND"] = name; os.environ["LFM3_KV800_BACKEND"] = name

    def backend(self):
        for obj in (self.kv, self.root):
            f = getattr(obj, "get_backend", None) or getattr(obj, "active_backend", None)
            if callable(f):
                try: return str(f())
                except Exception: pass
        return str(getattr(self.kv, "BACKEND", "?"))

    def store(self, B, max_len, window):
        return self.kv.PagedStore(B, max_len, DEV, window=window)

    def decode(self, st, q, cos_t, sin_t):
        P = st.paged()
        return self.kv.decode(P, q, cos_t, sin_t, st.H) if self.takes_H else self.kv.decode(P, q, cos_t, sin_t)

    def triton_modules(self):
        """Modules of the package that hold the Triton decode kernels (for the fp8 shim and the warps sweep)."""
        mods = [m for k, m in list(sys.modules.items()) if k.startswith("lfm3_kv800") and m is not None]
        return [m for m in mods if hasattr(m, "_qdec") or hasattr(m, "_xdec")]


def clone_store(st):
    """Snapshot of a PagedStore (the package has no clone()): shallow copy, tensors cloned, Hadamard shared."""
    c = copy.copy(st)
    names = list(getattr(st, "__dict__", {})) + [s for k in type(st).__mro__ for s in getattr(k, "__slots__", ())]
    for k in names:
        v = getattr(st, k, None)
        if torch.is_tensor(v) and k != "H": setattr(c, k, v.clone())
    return c


# %%
# ---------------------------------------------------------------- reference math (copied from lfm3kv.qkv / surgery_gpu)
F8 = torch.float8_e4m3fn
BLK, LAG, NSINK, WIN = 256, 128, 4, 256
f8 = lambda t: t.to(F8).to(t.dtype)


def hadamard(n=128, device=None):
    H = torch.ones(1, 1)
    while H.shape[0] < n: H = torch.cat((torch.cat((H, H), 1), torch.cat((H, -H), 1)), 0)
    return (H / n ** 0.5).to(device)


def _aq(x, bits, dim, fp8s=True):
    """Asymmetric min/max quantize-dequantize along `dim`, fp8 e4m3 lo/scale (surgery_gpu._aq mode "mm")."""
    lo, hi = x.amin(dim, keepdim=True), x.amax(dim, keepdim=True)
    sc = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-8)
    if fp8s: lo, sc = f8(lo), f8(sc).clamp_min(1e-8)
    return ((x - lo) / sc).round().clamp(0, 2 ** bits - 1) * sc + lo


def reference(q, Kh, Vh, cos_full, sin_full, window=0):
    """Torch reference with the store's semantics (lfm3kv.qkv.reference).
    q (B,16,128) bf16 post-RoPE; Kh (B,n,4,128) pre-RoPE bf16; Vh (B,n,4,128) bf16; cos/sin_full (n,128) bf16 (HF).
    -> (B,16,128) fp32."""
    B, n = Kh.shape[:2]; H = hadamard(128, q.device)
    nq = 0 if window else max(0, (n - LAG) // BLK)
    K, V = Kh.float().clone(), Vh.float().clone()
    if nq:
        x = Kh[:, :nq * BLK].float().reshape(B, nq, BLK, 512)
        K[:, :nq * BLK] = _aq(x, 2, 2).view(B, nq * BLK, 4, 128).to(torch.bfloat16).float()
        y = (Vh[:, :nq * BLK].float() @ H).reshape(B, nq * BLK, 2, 256)
        V[:, :nq * BLK] = (_aq(y, 1, -1).view(B, nq * BLK, 4, 128) @ H)
    cs, sn = cos_full[:n].float()[None, :, None], sin_full[:n].float()[None, :, None]
    rot = torch.cat((-K[..., 64:], K[..., :64]), -1)
    Kr = (K * cs + rot * sn).to(torch.bfloat16).float()
    Kr, V = Kr.repeat_interleave(4, 2), V.repeat_interleave(4, 2)
    sc = torch.einsum("bhd,bnhd->bhn", q.float(), Kr) * 128 ** -0.5
    if window:
        p = torch.arange(n, device=q.device)
        sc = sc.masked_fill(~((p >= n - window) | (p < NSINK))[None, None], float("-inf"))
    return torch.einsum("bhn,bnhd->bhd", sc.softmax(-1), V)


def rope_table(inv_freq, max_pos, device):
    """cos/sin of the 32 rotating pairs as Lfm3RotaryEmbedding computes them (fp32 product, bf16), (max_pos, 32)."""
    assert inv_freq.numel() == 64 and bool((inv_freq[32:] == 0).all()) and bool((inv_freq[:32] > 0).all()), \
        "full-attention inv_freq must be 32 nonzero + 32 zero frequencies (wrong transformers version?)"
    fr = torch.arange(max_pos, device=device, dtype=torch.float32)[:, None] * inv_freq[None, :32].to(device, torch.float32)
    return fr.cos().to(torch.bfloat16).contiguous(), fr.sin().to(torch.bfloat16).contiguous()


def test_tables(max_pos=40000, full=8192):
    inv = 1.0 / (1e6 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 128))
    c32, s32 = rope_table(torch.cat((inv, torch.zeros(32))), max_pos, DEV)
    fr = torch.arange(full, device=DEV, dtype=torch.float32)[:, None] * inv.to(DEV)[None]
    c64 = torch.cat((fr.cos(), torch.ones(full, 32, device=DEV)), 1); s64 = torch.cat((fr.sin(), torch.zeros(full, 32, device=DEV)), 1)
    return c32, s32, torch.cat((c64, c64), 1).bfloat16(), torch.cat((s64, s64), 1).bfloat16()


# %%
# ---------------------------------------------------------------- timing / memory helpers
GPU_PEAK_GBS = {"T4": 320, "P100": 732, "L4": 300, "V100": 900, "A10": 600, "A100": 1555, "H100": 3350, "A4000": 448,
                "RTX 4090": 1008, "RTX 3090": 936}


def sync():
    if DEV == "cuda": torch.cuda.synchronize()


def bench(fn, warm=3, iters=20):
    for _ in range(warm): fn()
    sync()
    if DEV == "cuda":
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters): fn()
        b.record(); b.synchronize()
        return a.elapsed_time(b) / 1e3 / iters
    t = time.perf_counter()
    for _ in range(iters): fn()
    return (time.perf_counter() - t) / iters


def free_bytes():
    if DEV != "cuda": return 8 << 30
    gc.collect(); torch.cuda.empty_cache()
    return torch.cuda.mem_get_info()[0]


def peak_gbs():
    if DEV != "cuda": return None
    name = torch.cuda.get_device_name(0)
    for k, v in GPU_PEAK_GBS.items():
        if k in name: return v
    return None


def kernels_seen(fn):
    """Names of the CUDA kernels one call launches (tells which backend really ran), or None if profiling fails."""
    if DEV != "cuda": return None
    try:
        from torch.profiler import profile, ProfilerActivity
        fn(); sync()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn(); sync()
        return sorted({e.key for e in prof.key_averages()})
    except Exception as e:  # noqa: BLE001
        return [f"<profiler failed: {e!r}>"]


def ran_triton(names):
    return None if names is None else any(("_qdec" in n) or ("_xdec" in n) for n in names)


def errtext(e):
    """str of an exception and its causes (Triton wraps the real compile error)."""
    out, seen = [], set()
    while e is not None and id(e) not in seen:
        seen.add(id(e)); out.append(f"{type(e).__name__}: {e}"); e = e.__cause__ or e.__context__
    return " <- ".join(out)


def short(e, n=300):
    """One line: the exception, or for wrapped errors (Triton CompilationError) the innermost cause."""
    root, seen = e, {id(e)}
    while root.__cause__ is not None and id(root.__cause__) not in seen:
        root = root.__cause__; seen.add(id(root))
    s = f"{type(e).__name__}: {e}" if root is e else f"{type(e).__name__} <- {type(root).__name__}: {root}"
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + " ..."


# %%
# ---------------------------------------------------------------- 1. environment report + Triton feature probes
if triton is not None:
    @triton.jit
    def _probe_bf16(X, Y, N: tl.constexpr):
        i = tl.arange(0, N)
        tl.store(Y + i, (tl.load(X + i).to(tl.float32) * 1.1).to(tl.bfloat16))   # 1.1: the product needs rounding

    @triton.jit
    def _probe_fp8(X, Y, N: tl.constexpr):
        i = tl.arange(0, N)
        tl.store(Y + i, tl.load(X + i).to(tl.float8e4nv, bitcast=True).to(tl.float32))

    @triton.jit
    def _f8_sw(p):
        """fp8 e4m3fn byte -> fp32 with integer ops only (exact; no NaN handling, the cache never stores NaN).
        Drop-in for the package's `_f8` on GPUs whose Triton rejects float8e4nv (compute capability < 8.9)."""
        b = tl.load(p).to(tl.int32)
        e = (b >> 3) & 15
        m = b & 7
        s = (b >> 7) & 1
        nrm = ((s << 31) | ((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
        sub = m.to(tl.float32) * 0.001953125
        return tl.where(e == 0, tl.where(s == 1, -sub, sub), nrm)

    @triton.jit
    def _probe_f8sw(X, Y, N: tl.constexpr):
        i = tl.arange(0, N)
        tl.store(Y + i, _f8_sw(X + i))

    @triton.jit
    def _probe_noop(X, N: tl.constexpr):
        i = tl.arange(0, N)
        tl.store(X + i, tl.load(X + i) + 1.0)


def env_report():
    reset_results()                                   # a re-run starts a clean report
    E = RES["env"]
    E.update(python=sys.version.split()[0], platform=platform.platform(), torch=torch.__version__,
             cuda=torch.version.cuda, triton=getattr(triton, "__version__", None), triton_import_error=TRITON_IMPORT_ERROR,
             kaggle=bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")), colab="google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG")))
    try:
        import transformers; E["transformers"] = transformers.__version__
    except Exception:
        E["transformers"] = None
    if DEV == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("No CUDA GPU visible. Kaggle: Settings > Accelerator > GPU T4 x2. Colab: Runtime > Change runtime type > T4 GPU.")
        p = torch.cuda.get_device_properties(0); cap = torch.cuda.get_device_capability(0)
        E.update(gpu=p.name, n_gpus=torch.cuda.device_count(), capability=f"{cap[0]}.{cap[1]}", sm=cap[0] * 10 + cap[1],
                 vram_gb=round(p.total_memory / 2**30, 1), free_gb=round(torch.cuda.mem_get_info()[0] / 2**30, 1),
                 arch_list=torch.cuda.get_arch_list(), cudnn=torch.backends.cudnn.version(), peak_gbs=peak_gbs())
        E["bf16_native"] = cap >= (8, 0)
        try:
            E["torch_bf16_supported_no_emulation"] = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:
            E["torch_bf16_supported_no_emulation"] = torch.cuda.is_bf16_supported()
        try:
            x = torch.randn(64, device=DEV); y = x.to(F8).float(); E["torch_fp8_cast"] = bool(torch.isfinite(y).all())
        except Exception as e:
            E["torch_fp8_cast"] = short(e)
        import re
        nums = [(a.split("_")[0], int(re.match(r"\D+_(\d+)", a).group(1))) for a in E["arch_list"] if re.match(r"(sm|compute)_\d+", a)]
        runs = any(k == "sm" and v // 10 == cap[0] and v % 10 <= cap[1] for k, v in nums) or \
            any(k == "compute" and v <= E["sm"] for k, v in nums)
        if not runs:
            note(f"this torch build has no sm_{E['sm']} kernels (arch list {E['arch_list']}); expect 'no kernel image' errors. "
                 "On Kaggle pick 'GPU T4 x2', not P100.")
    else:
        E.update(gpu="cpu", sm=0)
    for k, v in E.items(): log(f"  {k:36s} {v}")
    run_probes()


def run_probes():
    P = RES["probes"]
    if DEV != "cuda" or triton is None:
        P["skipped"] = "no CUDA or no triton"; log("  triton probes skipped"); return
    if RES["env"]["sm"] < 70:
        P["skipped"] = f"sm_{RES['env']['sm']} < sm_70: Triton does not support this GPU"; log("  " + P["skipped"]); return
    N = 256
    def probe(name, fn):
        try:
            P[name] = fn() or "ok"
        except Exception as e:  # noqa: BLE001
            exp = "fp8e4nv" in name and RES["env"]["sm"] < 89
            P[name] = ("unsupported here (expected below sm_89): " if exp else "FAIL ") + short(e, 200)
        log(f"  triton probe {name:28s} {P[name]}")
    def bf16():
        x = torch.randn(N, device=DEV).bfloat16(); y = torch.empty_like(x); _probe_bf16[(1,)](x, y, N=N)
        return None if torch.equal(y, (x.float() * 1.1).bfloat16()) else "WRONG RESULT (fp32->bf16 rounding differs from torch)"
    def fp8():
        x = torch.arange(N, device=DEV, dtype=torch.int32).to(torch.uint8); y = torch.empty(N, device=DEV)
        _probe_fp8[(1,)](x, y, N=N); r = x.view(F8).float(); ok = ~torch.isnan(r)
        return None if torch.equal(y[ok], r[ok]) else "WRONG RESULT"
    def f8sw():
        x = torch.arange(N, device=DEV, dtype=torch.int32).to(torch.uint8); y = torch.empty(N, device=DEV)
        _probe_f8sw[(1,)](x, y, N=N); r = x.view(F8).float(); ok = ~torch.isnan(r)
        return None if torch.equal(y[ok], r[ok]) else "WRONG RESULT"
    def ws():
        x = torch.zeros(N, device=DEV); _probe_noop[(1,)](x, N=N, num_warps=1, warp_size=16)
        return None if bool((x == 1).all()) else "WRONG RESULT"
    probe("bf16 load/round", bf16)
    probe("fp8e4nv bitcast (needs sm89)", fp8)
    probe("fp8 integer decode (shim)", f8sw)
    probe("launch kwarg warp_size=16", ws)


# %%
# ---------------------------------------------------------------- 2. kernel parity (the 12 cases of run/kv800_check.py)
PARITY_CASES = [(0, n) for n in (50, 383, 384, 700, 1500, 5000)] + [(WIN, n) for n in (3, 10, 200, 259, 260, 1500)]


def apply_f8_shim(pkg):
    """Swap the package's `_f8` helper for the integer decoder; returns a description or None if not applicable."""
    done = []
    for m in pkg.triton_modules():
        if hasattr(m, "_f8"):
            m._f8 = _f8_sw
            for v in list(vars(m).values()):             # drop cached hashes so the kernels recompile with the shim
                if triton is not None and isinstance(v, triton.runtime.JITFunction):
                    v.hash = None; v.used_global_vals = {}
            done.append(m.__name__)
    return ("_f8 -> integer e4m3 decode in " + ", ".join(done)) if done else None


def apply_kw_shim(pkg, drop=("warp_size",)):
    """Wrap the Triton kernels so unknown launch kwargs (XPU-only warp_size) are dropped. For old Triton on CUDA."""
    class _Drop:
        def __init__(self, k): self.k = k
        def __getitem__(self, grid):
            f = self.k[grid]
            return lambda *a, **kw: f(*a, **{x: y for x, y in kw.items() if x not in drop})
    done = []
    for m in pkg.triton_modules():
        for n in ("_qdec", "_xdec"):
            if hasattr(m, n) and not isinstance(getattr(m, n), _Drop):
                setattr(m, n, _Drop(getattr(m, n))); done.append(f"{m.__name__}.{n}")
    return ("dropped " + ",".join(drop) + " from " + ", ".join(done)) if done else None


def parity_case(pkg, win, n, tables, qdtype=torch.bfloat16, seed=0):
    c32, s32, cf, sf = tables
    torch.manual_seed(seed)
    B = 2
    Kh = (torch.randn(B, n, 4, 128, device=DEV) + 3 * torch.randn(1, 1, 4, 128, device=DEV)).bfloat16()
    Vh = torch.randn(B, n, 4, 128, device=DEV).bfloat16()
    q = (2 * torch.randn(B, 16, 128, device=DEV)).bfloat16()
    st = pkg.store(B, n + 8, win)
    pre = max(1, n - 5); st.append(Kh[:, :pre], Vh[:, :pre])
    for i in range(pre, n): st.append(Kh[:, i:i + 1], Vh[:, i:i + 1])        # the last tokens one at a time, as in decode
    o = pkg.decode(st, q.to(qdtype), c32, s32).float()
    r = reference(q, Kh, Vh, cf, sf, window=win)
    rel = ((o - r).norm() / r.norm()).item()
    return rel, (o - r).abs().max().item(), getattr(st, "nq", None), (lambda: pkg.decode(st, q.to(qdtype), c32, s32))


def parity_backend(pkg, be, tables, label=None):
    label = label or be
    try:
        pkg.set_backend(be)
    except Exception as e:  # noqa: BLE001
        log(f"  [{label}] cannot select backend: {short(e)}"); RES["parity"].append(dict(backend=label, error=short(e)))
        return False, e
    ok, first_err, rows = True, None, []
    for win, n in PARITY_CASES:
        row = dict(backend=label, win=win, n=n)
        try:
            rel, mx, nq, call = parity_case(pkg, win, n, tables)
            good = math.isfinite(rel) and rel < 1e-2
            row.update(rel=rel, maxabs=mx, nq=nq, ok=good)
            if win == 0 and n == 5000 and DEV == "cuda":
                ks = kernels_seen(call); row["kernels"] = ks; row["triton_ran"] = ran_triton(ks)
            log(f"  [{label}] parity win={win:3d} n={n:5d} nq={nq}: rel {rel:.2e} maxabs {mx:.2e} {'ok' if good else 'FAIL'}")
        except Exception as e:  # noqa: BLE001
            good = False; first_err = first_err or e; row.update(ok=False, error=short(e))
            log(f"  [{label}] parity win={win:3d} n={n:5d}: ERROR {short(e)}")
        ok &= good; rows.append(row)
        if first_err is not None and len(rows) == 1: break      # a compile error repeats for every case; stop early
    # informational, not part of PASS: fp16 query (T4-friendly dtype); the kernels upcast q to fp32 on load
    if ok:
        for win, n in ((0, 1500), (WIN, 1500)):
            try:
                rel, mx, _, _ = parity_case(pkg, win, n, tables, qdtype=torch.float16)
                log(f"  [{label}] info: fp16 query win={win} n={n}: rel {rel:.2e}")
                rows.append(dict(backend=label, win=win, n=n, qdtype="fp16", rel=rel, info=True))
            except Exception as e:  # noqa: BLE001
                log(f"  [{label}] info: fp16 query not accepted ({short(e, 120)})")
                rows.append(dict(backend=label, win=win, n=n, qdtype="fp16", error=short(e), info=True))
    if ok and be == "triton":
        tr = [r.get("triton_ran") for r in rows if "triton_ran" in r]
        if tr and tr[0] is False:
            note(f"[{label}] parity passed but no _qdec/_xdec CUDA kernel ran: the package fell back to another backend")
    RES["parity"].extend(rows)
    return ok, first_err


def run_parity(pkg):
    tables = test_tables()
    sm = RES["env"].get("sm", 0)
    backends = list(CFG["backends"] or (["triton", "torch"] if DEV == "cuda" else ["torch"]))
    status, final = {}, {}
    pre = []                                          # shims already applied earlier in this Python session
    for m in pkg.triton_modules():
        if "_f8_sw" in globals() and getattr(m, "_f8", None) is globals()["_f8_sw"] and "f8shim" not in pre: pre.append("f8shim")
        if any(type(getattr(m, k, None)).__name__ == "_Drop" for k in ("_qdec", "_xdec")) and "kwshim" not in pre: pre.append("kwshim")
    for be in backends:
        if be == "triton" and (triton is None or DEV != "cuda" or sm < 70):
            status[be] = final[be] = "SKIP (needs triton on a CUDA GPU >= sm_70)"; log(f"  [triton] {status[be]}"); continue
        t = time.time(); shims = list(pre) if be == "triton" else []
        label = "triton+" + "+".join(shims) if shims else be
        if shims:
            status[be] = SHIPPED.get(be, "not re-tested")
            note(f"the package was already patched ({', '.join(shims)}) earlier in this session; testing {label}. "
                 "Restart the runtime for a fresh as-shipped result.")
        ok, err = parity_backend(pkg, be, tables, label=label)
        while be == "triton" and not ok and err is not None:
            # known CUDA gaps of the shipped kernels -> report the failure, patch in place, re-run under a new label
            msg = errtext(err)
            if ("fp8e4nv" in msg or "not supported in this architecture" in msg) and "f8shim" not in shims:
                fix, why, tag = apply_f8_shim(pkg), f"Triton has no float8e4nv below sm_89 (this GPU: sm_{sm})", "f8shim"
            elif "warp_size" in msg and "unrecognised" in msg and "kwshim" not in shims:
                fix, why, tag = apply_kw_shim(pkg), "this Triton rejects the XPU launch kwarg warp_size", "kwshim"
            else:
                break
            if not fix:
                break
            if not shims: status[be] = SHIPPED[be] = f"FAIL as shipped: {why}"
            shims.append(tag); status[label] = status.get(label) if label == be else "FAIL"
            label = "triton+" + "+".join(shims)
            log(f"  [triton] {why}; retrying as {label} ({fix})")
            ok, err = parity_backend(pkg, be, tables, label=label)
        status[label] = "PASS" if ok else "FAIL"
        if label != be and be not in status: status[be] = "FAIL as shipped"
        final[be] = status[label]
        if ok and be == "triton": RES["active_triton_label"] = label
        log(f"  [{label}] parity {status[label]} ({time.time() - t:.0f}s incl. compile)")
    # what does the automatic chain pick on this machine?
    try:
        pkg.set_backend("auto")
        st = pkg.store(1, 1024, 0); st.append(torch.randn(1, 1000, 4, 128, device=DEV).bfloat16(), torch.randn(1, 1000, 4, 128, device=DEV).bfloat16())
        q = torch.randn(1, 16, 128, device=DEV).bfloat16()
        ks = kernels_seen(lambda: pkg.decode(st, q, tables[0], tables[1]))
        RES["auto_backend"] = dict(reported=pkg.backend(), triton_kernels_ran=ran_triton(ks))
        log(f"  auto backend: reports {pkg.backend()!r}, triton kernels ran: {ran_triton(ks)}")
    except Exception as e:  # noqa: BLE001
        RES["auto_backend"] = dict(error=short(e)); log(f"  auto backend check failed: {short(e)}")
    RES["parity_status"], RES["parity_final"] = status, final
    return status


# %%
# ---------------------------------------------------------------- 3. decode speed microbench
def fill_random(pkg, st, n, win):
    """Random but valid cache contents (fp8 lo/scale in sane ranges), lengths set to n."""
    C = pkg.C
    if not win:
        v = st.qpool.view(-1, C["SUB"]); v.random_(0, 256)
        for a, b in ((C["VLO_O"], C["VSC_O"]), (C["KLO_O"], C["KSC_O"])):
            v[:, a:b] = (torch.randn(v.shape[0], b - a, device=DEV) * 0.5).to(F8).view(torch.uint8)
        for a, b in ((C["VSC_O"], C["KLO_O"]), (C["KSC_O"], C["SUB"])):
            v[:, a:b] = (torch.rand(v.shape[0], b - a, device=DEV) * 0.5 + 0.1).to(F8).view(torch.uint8)
    for t in (st.epool, getattr(st, "sk", None), getattr(st, "sv", None)):
        if t is not None: t.normal_()
    st.n, st.nq = n, 0 if win else (n - C["LAG"]) // C["BLK"]
    st.N.fill_(n); st.NQ.fill_(st.nq)


def store_bytes(B, n, win, ebs=64, nring=10):
    return (0 if win else B * (n // 256 + 1) * 51200) + B * nring * 4 * ebs * 256 * 2


def fit_n(B, n, bytes_fn, frac=0.6):
    """Largest n' <= n (halving) whose estimate fits in `frac` of free memory; 0 if even 1K does not fit."""
    free = free_bytes()
    while n >= 1024 and bytes_fn(B, n) > frac * free: n //= 2
    return n if n >= 1024 else 0


def run_speed(pkg):
    tables = test_tables(max_pos=CFG["n"] + 64)
    c32, s32 = tables[0], tables[1]
    status = RES.get("parity_status", {})
    tri = RES.get("active_triton_label")
    backends = [b for b in (CFG["backends"] or ["triton", "torch"]) if b != "triton" or tri]
    if "triton" in (CFG["backends"] or ["triton"]) and not tri:
        note("triton speed skipped: triton parity did not pass (or triton was not run) on this machine")
    for be in list(backends):
        if be != "triton" and status.get(be) not in ("PASS", None):
            backends.remove(be); note(f"{be} speed skipped: parity {status.get(be)}")
    pk = peak_gbs()
    for B in CFG["batches"]:
        for win in (0, WIN):
            n = fit_n(B, CFG["n"], lambda b, nn: store_bytes(b, nn, win))
            if not n:
                log(f"  speed {'window' if win else 'quant'} B={B}: skipped (does not fit)"); continue
            if n != CFG["n"]: note(f"speed B={B} {'window' if win else 'quant'}: scaled n {CFG['n']} -> {n} to fit free memory")
            try:
                st = pkg.store(B, n, win); fill_random(pkg, st, n, win)
            except Exception as e:  # noqa: BLE001
                log(f"  speed B={B} win={win}: store alloc failed {short(e)}"); continue
            q = torch.randn(B, 16, 128, device=DEV).bfloat16()
            byt = B * (st.nq * 256 * 200 + ((n - st.nq * 256) if not win else WIN + NSINK) * 2048)
            for be in backends:
                pkg.set_backend(be); label = tri if be == "triton" else be
                try:
                    s = bench(lambda: pkg.decode(st, q, c32, s32))
                    gbs = byt / s / 1e9
                    row = dict(backend=label, layer="window" if win else "quant", B=B, n=n, us=s * 1e6, gbs=gbs,
                               pct_peak=(100 * gbs / pk) if pk else None)
                    log(f"  [{label}] speed {row['layer']:6s} B={B:3d} n={n}: {s * 1e6:9.0f} us/call  {gbs:7.1f} GB/s of cache"
                        + (f"  ({row['pct_peak']:.0f}% of {pk} GB/s peak)" if pk else ""))
                except torch.cuda.OutOfMemoryError as e:
                    row = dict(backend=label, layer="window" if win else "quant", B=B, n=n, error="OOM"); log(f"  [{label}] B={B} win={win}: OOM")
                except Exception as e:  # noqa: BLE001
                    row = dict(backend=label, layer="window" if win else "quant", B=B, n=n, error=short(e)); log(f"  [{label}] B={B} win={win}: {short(e)}")
                RES["speed"].append(row)
                gc.collect()
                if DEV == "cuda": torch.cuda.empty_cache()
            if CFG["sweep_warps"] and tri and B == 16 and not win:
                sweep_warps(pkg, st, q, c32, s32, byt, tri)
            del st; gc.collect()
            if DEV == "cuda": torch.cuda.empty_cache()
    run_sdpa()


def sweep_warps(pkg, st, q, c32, s32, byt, label):
    """num_warps 1/2/4 for the Triton kernels (module global QW, as in lfm3kv/kv800.py). The kernels were tuned on
    Intel (one 16-lane sub-group); this shows how far that is from a CUDA-friendly launch. Informational."""
    mods = [m for m in pkg.triton_modules() if hasattr(m, "QW")]
    if not mods:
        log("  warps sweep skipped: package has no module-level QW"); return
    m = mods[0]; old = m.QW; pkg.set_backend("triton")
    try:
        for qw in (1, 2, 4):
            m.QW = qw
            try:
                s = bench(lambda: pkg.decode(st, q, c32, s32))
                log(f"  [{label}] warps sweep quant B=16: num_warps={qw}: {s * 1e6:9.0f} us/call  {byt / s / 1e9:7.1f} GB/s")
                RES["speed"].append(dict(backend=label, layer="quant", B=16, n=st.n, num_warps=qw, us=s * 1e6, gbs=byt / s / 1e9, sweep=True))
            except Exception as e:  # noqa: BLE001
                log(f"  warps sweep num_warps={qw}: {short(e, 150)}")
    finally:
        m.QW = old


def run_sdpa():
    """Stock decode attention for scale: one query token over a 4-kv-head 16-bit cache (GQA 16/4), torch SDPA."""
    n0 = CFG["n"]; pk = peak_gbs()
    dts = [("bf16", torch.bfloat16), ("fp16", torch.float16)] if DEV == "cuda" else [("bf16", torch.bfloat16)]
    for name, dt in dts:
        for B in CFG["batches"]:
            n = fit_n(B, n0, lambda b, nn: 2 * b * 4 * nn * 128 * 2)
            if not n:
                log(f"  sdpa {name} B={B}: skipped (does not fit)"); continue
            if n != n0: note(f"SDPA {name} B={B}: scaled n {n0} -> {n} to fit free memory")
            try:
                q = torch.randn(B, 16, 1, 128, device=DEV, dtype=dt)
                k = torch.randn(B, 4, n, 128, device=DEV, dtype=dt); v = torch.randn_like(k)
                try:
                    s = bench(lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=True)); how = "enable_gqa"
                except (TypeError, RuntimeError) as e:
                    if 2 * B * 16 * n * 128 * 2 > 0.6 * free_bytes():
                        raise RuntimeError(f"enable_gqa unsupported ({short(e, 80)}) and the expanded cache does not fit")
                    kk, vv = k.repeat_interleave(4, 1), v.repeat_interleave(4, 1)
                    s = bench(lambda: F.scaled_dot_product_attention(q, kk, vv)); how = "expanded to 16 heads"
                    del kk, vv
                byt = B * n * 2048; gbs = byt / s / 1e9
                row = dict(dtype=name, B=B, n=n, us=s * 1e6, gbs=gbs, how=how, pct_peak=(100 * gbs / pk) if pk else None)
                log(f"  [sdpa {name}] decode B={B:3d} n={n}: {s * 1e6:9.0f} us/call  {gbs:7.1f} GB/s of 16-bit cache ({how})")
            except Exception as e:  # noqa: BLE001
                row = dict(dtype=name, B=B, n=n, error=short(e)); log(f"  [sdpa {name}] B={B}: {short(e)}")
            RES["sdpa"].append(row)
            q = k = v = None; gc.collect()
            if DEV == "cuda": torch.cuda.empty_cache()


# %%
# ---------------------------------------------------------------- 4. optional end to end: HF LFM3 with the 800h cache
KEYS = ["blue door", "red train", "old lighthouse", "green kettle", "silver bike", "quiet library", "paper boat", "iron gate",
        "copper lamp", "north tower", "wooden bridge", "glass jar", "yellow kite", "stone well", "black piano", "white tent",
        "orange van", "brass key", "tin drum", "velvet chair", "marble stair", "cedar box", "linen map", "amber clock",
        "frozen lake", "hidden cave", "rusty anchor", "golden bell", "purple scarf", "broken compass", "narrow alley", "clay pot"]
CH = 4096


def patch_transformers():
    """Guarded copies of lfm3kv/model.py's config patches. Some transformers 5.x releases rename the `conv` layer type
    to `linear_attention` and flatten the nested per-layer-type rope_parameters; both break LFM3. No-ops where not needed."""
    done = []
    try:
        import transformers.configuration_utils as cu
        remap = getattr(cu, "_LEGACY_LAYER_TYPE_REMAP", None)
        if isinstance(remap, dict) and "conv" in remap:
            remap.pop("conv"); done.append("conv remap removed")
        al = getattr(cu, "ALLOWED_ATTN_LAYER_TYPES", None)
        if isinstance(al, tuple) and "conv" not in al:
            cu.ALLOWED_ATTN_LAYER_TYPES = al + ("conv",); done.append("conv allowed")
    except Exception as e:  # noqa: BLE001
        done.append(f"layer-type patch skipped ({e!r})")
    try:
        from transformers.modeling_rope_utils import RotaryEmbeddingConfigMixin as R
        if hasattr(R, "convert_rope_params_to_dict") and hasattr(R, "standardize_rope_params") and not getattr(R, "_kv800_patched", False):
            orig = R.convert_rope_params_to_dict
            def conv(self, **kw):
                rp = getattr(self, "rope_parameters", None)
                if isinstance(rp, dict) and any(isinstance(v, dict) for v in rp.values()):
                    self._rope_type_labels = getattr(self, "_rope_type_labels", getattr(self, "_attention_layer_types", None))
                    self.standardize_rope_params(); return kw
                return orig(self, **kw)
            R.convert_rope_params_to_dict = conv; R._kv800_patched = True; done.append("nested rope_parameters kept")
    except Exception as e:  # noqa: BLE001
        done.append(f"rope patch skipped ({e!r})")
    return done


def load_model(dtype):
    import importlib.metadata, transformers
    inst = importlib.metadata.version("transformers")
    if inst != transformers.__version__:
        raise RuntimeError(f"transformers {inst} is installed but {transformers.__version__} is already imported in this "
                           "Python session: restart the runtime (Colab: Runtime > Restart session; Kaggle: Run > Restart) and Run all")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    patches = patch_transformers()
    log(f"  transformers patches: {patches}")
    kw = dict(trust_remote_code=True, attn_implementation="sdpa")
    if not os.path.exists(CFG["model"]): kw["revision"] = CFG["revision"]
    tok = AutoTokenizer.from_pretrained(CFG["model"], trust_remote_code=True,
                                        **({"revision": CFG["revision"]} if "revision" in kw else {}))
    try:
        m = AutoModelForCausalLM.from_pretrained(CFG["model"], dtype=dtype, **kw)
    except TypeError:                                            # transformers 4.x spelling
        m = AutoModelForCausalLM.from_pretrained(CFG["model"], torch_dtype=dtype, **kw)
    m = m.to(DEV); m.requires_grad_(False).eval()
    lm = m.model
    lm.rotary_emb = type(lm.rotary_emb)(m.config, device=DEV)    # fp32 inv_freq whatever the weight dtype
    lt = list(m.config.layer_types)
    bad = [l for l in QL + WL if lt[l] != "full_attention"]
    nz_f = int((lm.rotary_emb.full_attention_inv_freq != 0).sum()); nz_s = int((lm.rotary_emb.sliding_attention_inv_freq != 0).sum())
    log(f"  layer_types[0..3]={lt[:4]}  full-attn rope nonzero freqs {nz_f}/64 (want 32), sliding {nz_s}/64 (want 64)")
    if bad or lt[0] != "conv" or nz_f != 32 or nz_s != 64:
        raise RuntimeError(f"model config was not parsed as expected (layer types {lt}, rope {nz_f}/{nz_s}); "
                           "install the pinned transformers version (README_NVIDIA.md)")
    return tok, m, lm, m.lm_head.weight


class KVLayers:
    """Port of lfm3kv/qkv_run.py's KVLayers onto the package's PagedStore: layers 6/10/14/22 keep 2-bit K / 1-bit V
    blocks + an exact tail, layers 2/18 keep 4 sinks + the last 256 tokens. Prefill attention stays exact (eager fp32),
    every decode step (question, answer, teacher-forced code) reads the cache through pkg.decode."""

    def __init__(self, pkg, lm, max_pos):
        self.pkg = pkg
        self.cos_t, self.sin_t = rope_table(lm.rotary_emb.full_attention_inv_freq, max_pos, DEV)
        self.mods = {l: lm.layers[l].self_attn for l in QL + WL}
        self.M = sys.modules[type(self.mods[2]).__module__]
        self.st = {}
        for l, a in self.mods.items(): a.forward = self._fwd(a, l)

    def reset(self, max_len):
        self.st = {l: self.pkg.store(1, max_len, WIN if l in WL else 0) for l in self.mods}

    def snapshot(self):
        return {l: clone_store(s) for l, s in self.st.items()}

    def remove(self):
        for a in self.mods.values(): a.__dict__.pop("forward", None)

    def _fwd(self, attn, l):
        M = self.M

        def fwd(hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kw):
            B, T, _ = hidden_states.shape; hsh = (B, T, -1, 128); st = self.st[l]
            q = attn.q_norm(attn.q_proj(hidden_states).view(hsh)).transpose(1, 2)
            kp = attn.k_norm(attn.k_proj(hidden_states).view(hsh))                 # (B, T, 4, 128) pre-RoPE
            v = attn.v_proj(hidden_states).view(hsh)
            qr, k = M.apply_rotary_pos_emb(q, kp.transpose(1, 2), *position_embeddings)
            kf, vf = past_key_values.update(k, v.transpose(1, 2), l)              # exact copy: positions + prefill
            st.append(kp, v)
            if T == 1:
                o = self.pkg.decode(st, qr[:, :, 0].to(torch.bfloat16).contiguous(), self.cos_t, self.sin_t)
                o = o.to(hidden_states.dtype)[:, None]                             # (B, 1, 16, 128)
            else:
                o = self._prefill(attn, qr, kf, vf, st.win).transpose(1, 2)
            gate = torch.sigmoid(attn.gate_proj(hidden_states)).unsqueeze(-1)
            return attn.o_proj((o * gate).reshape(B, T, -1)), None
        return fwd

    @staticmethod
    def _prefill(attn, q, k, v, win):
        """Exact causal attention (quantized layers) or 4 sinks + window 256 (window layers), eager fp32, chunked."""
        T, S = q.shape[2], k.shape[2]
        k, v = k.repeat_interleave(4, 1).float(), v.repeat_interleave(4, 1).float()
        qp = torch.arange(S - T, S, device=q.device)[:, None]; kp = torch.arange(S, device=q.device)[None]
        m = kp <= qp
        if win: m = m & ((kp > qp - win) | (kp < NSINK))
        o = torch.empty_like(q)
        for i in range(0, T, 512):
            sc = (q[:, :, i:i + 512].float() @ k.transpose(-1, -2)) * attn.scaling
            o[:, :, i:i + 512] = (sc.masked_fill(~m[None, None, i:i + 512], float("-inf")).softmax(-1) @ v).to(q.dtype)
        return o


def haystack_ids(tok, need):
    """Python stdlib source as filler (argparse, difflib, inspect, typing of the running interpreter)."""
    import argparse as a1, difflib as a2, inspect as a3, typing as a4
    src = "".join(open(inspect.getsourcefile(mod)).read() for mod in (a1, a2, a3, a4))
    ids = tok(src, add_special_tokens=False)["input_ids"]
    while len(ids) < need + 64: ids = ids + ids
    return ids


def needle_items(tok, seed=3):
    """`needles` items per context length; needle depths spread evenly over (0, 1); fixed seed -> same items for both runs."""
    L, k = CFG["lengths"], CFG["needles"]
    ids = haystack_ids(tok, max(L)); rnd = random.Random(seed); out = []
    for n in L:
        keys = rnd.sample(KEYS, k) if k <= len(KEYS) else rnd.choices(KEYS, k=k)
        for i in range(k):
            d = (i + 0.5) / k
            code, start = str(rnd.randint(1000, 9999)), rnd.randint(0, len(ids) - n - 1)
            body = ids[start:start + n]; cut = int(d * n)
            text = tok.decode(body[:cut]) + f"\n# The secret number of the {keys[i]} is {code}.\n" + tok.decode(body[cut:])
            qs = f"\n\nQuestion: What is the secret number of the {keys[i]}?\nAnswer: The secret number of the {keys[i]} is"
            out.append((n, f"{n // 1024}K@{int(d * 100)}%", code, tok(text)["input_ids"], tok(qs, add_special_tokens=False)["input_ids"]))
    return out


class Runner:
    def __init__(self, m, lm, W, kv):
        self.m, self.lm, self.W, self.kv = m, lm, W, kv
        self.prefill_s = self.prefill_tok = self.dec_s = self.dec_n = 0

    def step(self, cache, t):
        sync(); t0 = time.perf_counter()
        h = self.lm(input_ids=torch.tensor([[t]], device=DEV), past_key_values=cache, use_cache=True).last_hidden_state[0, -1]
        sync(); self.dec_s += time.perf_counter() - t0; self.dec_n += 1
        return h

    def prefill(self, ids, extra):
        from transformers import DynamicCache
        if self.kv: self.kv.reset(len(ids) + extra + 16)
        cache = DynamicCache(config=self.m.config); sync(); t = time.perf_counter()
        for i in range(0, len(ids), CH):
            h = self.lm(input_ids=torch.tensor([ids[i:i + CH]], device=DEV), past_key_values=cache, use_cache=True).last_hidden_state[0, -1]
        sync(); self.prefill_s += time.perf_counter() - t; self.prefill_tok += len(ids)
        return cache, h

    def logits(self, h):
        return (h @ self.W.t()).float()

    def item(self, ctx, q, code_ids, n_ans=8):
        cache, h = self.prefill(ctx, len(q) + len(code_ids) + n_ans + 8)
        for t in q: h = self.step(cache, t)
        snap = (copy.deepcopy(cache), self.kv.snapshot() if self.kv else None)
        ans, hh = [], h
        for _ in range(n_ans):
            nxt = int(self.logits(hh).argmax()); ans.append(nxt); hh = self.step(cache, nxt)
        cache = snap[0]
        if self.kv: self.kv.st = snap[1]
        lp, hh = 0.0, h
        for i, c in enumerate(code_ids):
            lp += torch.log_softmax(self.logits(hh), -1)[c].item()
            if i + 1 < len(code_ids): hh = self.step(cache, c)
        return ans, lp


def lossless_check(pkg, tok, m, lm, W):
    """Short context (< 384 tokens): no quantized blocks yet and the window covers everything, so the 800h path must
    match stock up to bf16 rounding. Catches wiring mistakes before the long needles."""
    from transformers import DynamicCache
    ids = haystack_ids(tok, 200)[:200]          # <= 256 tokens: window layers see all, quantized layers have no blocks
    def run(kv):
        if kv: kv.reset(256)
        c = DynamicCache(config=m.config)
        h = lm(input_ids=torch.tensor([ids[:-4]], device=DEV), past_key_values=c, use_cache=True).last_hidden_state[0, -1]
        outs = []
        for t in ids[-4:]:
            h = lm(input_ids=torch.tensor([[t]], device=DEV), past_key_values=c, use_cache=True).last_hidden_state[0, -1]
            outs.append((h @ W.t()).float())
        return torch.stack(outs)
    with torch.no_grad():
        a = run(None)
        kv = KVLayers(pkg, lm, 1024)
        try: b = run(kv)
        finally: kv.remove()
    d = (a - b).abs().max().item(); agree = bool((a.argmax(-1) == b.argmax(-1)).all())
    rel = ((a - b).norm() / a.norm()).item()
    log(f"  lossless regime (196-token prompt, 4 decode steps): max |dlogit| {d:.3f}, rel {rel:.2e}, argmax agree {agree}"
        " (only bf16/fp32 rounding differs here; a wiring bug gives rel ~ 1)")
    RES["e2e"]["lossless"] = dict(max_dlogit=d, rel=rel, argmax_agree=agree)
    return rel < 0.1


def e2e_one(name, m, lm, W, tok, kv, items):
    r = Runner(m, lm, W, kv); hits, lps, per = [], [], {}
    t0 = time.time()
    with torch.no_grad():
        for n, nm, code, ctx, q in items:
            c = tok(" " + code, add_special_tokens=False)["input_ids"]
            ans, lp = r.item(ctx, q, c); hit = code in tok.decode(ans)
            hits.append(hit); lps.append(lp); per.setdefault(n, []).append(hit)
            log(f"    [{name}] {nm:9s} code {code}  answer {tok.decode(ans)!r:40s} {'HIT' if hit else 'miss'}  logp {lp:.2f}")
    out = dict(recall=f"{sum(hits)}/{len(hits)}", mean_logp=sum(lps) / len(lps),
               per_length={f"{n // 1024}K": f"{sum(v)}/{len(v)}" for n, v in per.items()},
               prefill_tok_s=r.prefill_tok / max(r.prefill_s, 1e-9), decode_ms_per_token=1e3 * r.dec_s / max(r.dec_n, 1),
               wall_s=time.time() - t0)
    log(f"  {name}: recall {out['recall']} per length {out['per_length']}  mean logp {out['mean_logp']:.3f}  "
        f"harness prefill {out['prefill_tok_s']:.0f} tok/s, decode {out['decode_ms_per_token']:.1f} ms/token (B=1, HF eager harness)")
    return out


def run_e2e(pkg):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[CFG["e2e_dtype"]]
    if dtype is torch.bfloat16 and DEV == "cuda" and not RES["env"].get("bf16_native"):
        note("e2e runs the model in bf16 on a GPU without native bf16 (sm < 80): matmuls are emulated, expect it slow; "
             "--e2e-dtype fp16 is faster but LFM3 was not validated in fp16")
    tok, m, lm, W = load_model(dtype)
    # stock smoke forward; if the model's own Triton conv kernel fails on this GPU, fall back to its torch path
    try:
        with torch.no_grad(): lm(input_ids=torch.tensor([[1, 2, 3, 4]], device=DEV))
    except Exception as e:  # noqa: BLE001
        Mmod = sys.modules[type(lm).__module__]
        if hasattr(Mmod, "can_use_lfm3_conv"):
            Mmod.can_use_lfm3_conv = lambda *a, **k: False
            note(f"model's Triton conv kernel failed here ({short(e, 120)}); using its torch path")
            with torch.no_grad(): lm(input_ids=torch.tensor([[1, 2, 3, 4]], device=DEV))
        else:
            raise
    tri = RES.get("active_triton_label")
    be = "triton" if tri else "torch"
    pkg.set_backend(be)
    RES["e2e"]["backend"] = tri or "torch"; RES["e2e"]["dtype"] = CFG["e2e_dtype"]
    log(f"  800h decode backend for e2e: {tri or 'torch'}")
    if not lossless_check(pkg, tok, m, lm, W):
        note("e2e lossless check disagrees with stock: the needle numbers below are not trustworthy")
    items = needle_items(tok)
    log(f"  {len(items)} needles ({CFG['needles']} per length at {list(CFG['lengths'])})")
    RES["e2e"]["stock"] = e2e_one("stock", m, lm, W, tok, None, items)
    kv = KVLayers(pkg, lm, max(len(it[3]) for it in items) + 512)
    try:
        RES["e2e"]["kv800"] = e2e_one("800h", m, lm, W, tok, kv, items)
    finally:
        kv.remove()
    del m, lm, W; gc.collect()
    if DEV == "cuda": torch.cuda.empty_cache()


# %%
# ---------------------------------------------------------------- summary + main
def summary():
    log("\n================ SUMMARY ================")
    E = RES["env"]
    log(f"GPU {E.get('gpu')} (sm_{E.get('sm')}, {E.get('vram_gb')} GB), torch {E.get('torch')}, CUDA {E.get('cuda')}, "
        f"triton {E.get('triton')}, native bf16: {E.get('bf16_native')}")
    for k, v in RES.get("probes", {}).items(): log(f"  probe {k:30s} {v}")
    for k, v in RES.get("parity_status", {}).items(): log(f"  parity {k:22s} {v}")
    if "auto_backend" in RES: log(f"  auto backend: {RES['auto_backend']}")
    rows = [r for r in RES["speed"] if "us" in r and not r.get("sweep")]
    if rows:
        log(f"  {'backend':16s} {'layer':6s} {'B':>4s} {'n':>6s} {'us/call':>10s} {'GB/s':>8s}")
        for r in rows: log(f"  {r['backend']:16s} {r['layer']:6s} {r['B']:4d} {r['n']:6d} {r['us']:10.0f} {r['gbs']:8.1f}")
    for r in RES["sdpa"]:
        if "us" in r: log(f"  sdpa {r['dtype']:11s} {'dense':6s} {r['B']:4d} {r['n']:6d} {r['us']:10.0f} {r['gbs']:8.1f}  ({r['how']})")
    for k in ("stock", "kv800"):
        if k in RES["e2e"]:
            e = RES["e2e"][k]; log(f"  e2e {k:6s} recall {e['recall']} {e['per_length']} logp {e['mean_logp']:.3f} decode {e['decode_ms_per_token']:.1f} ms/tok")
    for s in RES["notes"]: log("  note: " + s)
    try:
        with open(CFG["out"], "w") as f: json.dump(RES, f, indent=1, default=str)
        log(f"results written to {os.path.abspath(CFG['out'])}")
    except Exception as e:  # noqa: BLE001
        log(f"could not write {CFG['out']}: {e!r}")


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--device", default=CFG["device"], choices=["cuda", "cpu"], help="cpu = torch backend dry run")
    p.add_argument("--backends", default=None, help="comma list, default triton,torch on cuda / torch on cpu")
    p.add_argument("--n", type=int, default=CFG["n"], help="context length for the speed bench (default 32768)")
    p.add_argument("--batches", default=",".join(map(str, CFG["batches"])))
    p.add_argument("--skip-speed", action="store_true")
    p.add_argument("--e2e", action="store_true", help="also run the HF LFM3 end-to-end needles (downloads ~3.4 GB)")
    p.add_argument("--e2e-dtype", default=CFG["e2e_dtype"], choices=["bf16", "fp16", "fp32"])
    p.add_argument("--lengths", default=",".join(map(str, CFG["lengths"])))
    p.add_argument("--needles", type=int, default=CFG["needles"], help="needles per context length")
    p.add_argument("--model", default=CFG["model"]); p.add_argument("--revision", default=CFG["revision"])
    p.add_argument("--no-sweep", action="store_true"); p.add_argument("--out", default=CFG["out"])
    a = p.parse_args(argv)
    CFG.update(device=a.device, backends=a.backends.split(",") if a.backends else None, n=a.n,
               batches=tuple(int(x) for x in a.batches.split(",")), e2e=a.e2e, e2e_dtype=a.e2e_dtype,
               lengths=tuple(int(x) for x in a.lengths.split(",")), needles=a.needles, model=a.model,
               revision=a.revision, sweep_warps=not a.no_sweep, out=a.out)
    return a


DEV = CFG["device"]


def main(argv=None):
    global DEV
    a = parse(argv); DEV = CFG["device"]
    t0 = time.time()
    log("== 1. environment"); env_report()
    try:
        pkg = Pkg()
    except Exception as e:
        log(f"PACKAGE NOT USABLE: {e}"); RES["notes"].append(str(e)); summary(); return 2
    RES["env"]["package_version"] = pkg.version; log(f"  lfm3_kv800 {pkg.version}, decode takes H: {pkg.takes_H}")
    log("== 2. kernel parity vs torch reference")
    status = run_parity(pkg)
    final = RES.get("parity_final", {})        # last attempt per backend (after any shim)
    ok = any(v == "PASS" for v in final.values()) and not any(v == "FAIL" for v in final.values())
    if not a.skip_speed:
        log("== 3. decode speed")
        try: run_speed(pkg)
        except Exception as e:  # noqa: BLE001
            note(f"speed section crashed: {short(e)}"); traceback.print_exc()
    if CFG["e2e"]:
        log("== 4. end to end (HF LFM3-1.7B-Base)")
        try: run_e2e(pkg)
        except Exception as e:  # noqa: BLE001
            note(f"e2e section crashed: {short(e)}"); traceback.print_exc()
    RES["wall_s"] = time.time() - t0
    summary()
    shipped_fail = any(str(v).startswith("FAIL as shipped") for v in RES.get("parity_status", {}).values())
    return 1 if not ok else (3 if shipped_fail else 0)     # 3 = correct only with the in-script shim


if __name__ == "__main__":
    sys.exit(main())
