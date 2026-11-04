"""Tests of lfm3_kv800.train (the training-side 800h hook).

    PYTHONPATH=<repo root> HF_HUB_OFFLINE=1 LFM3_MODEL=<LFM3-1.7B-Base dir> python -m pytest -q -s tests/test_train_hook.py

CPU-only tests (quantizer parity with the cache's quantize_blocks -> dequantize_blocks, mask / group alignment under
padding and packing) always run. Model tests need the LFM3-1.7B-Base weights (LFM3_MODEL, or the HF cache) and run on
XPU / CUDA if present, else CPU. The CE reference test needs the local text files of the RECOVERY PROBE eval passages
(Fedora: python3.14 stdlib, libreoffice-data, man-db) and reproduces its step-0 numbers (stock 1.582, 800h 1.735)."""
import glob, os, re

import pytest
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from lfm3_kv800 import train as kt  # noqa: E402

DEV = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
REF_STOCK, REF_800H = 1.582, 1.735                     # RESULTS.md RECOVERY PROBE, step 0 (rule "last128")


def _model_dir():
    if os.environ.get("LFM3_MODEL"):
        return os.environ["LFM3_MODEL"]
    roots = [os.environ.get("HF_HOME", ""), os.path.expanduser("~/.cache/huggingface"), "/mnt/ssd/huggingface"]
    for r in roots:
        hits = sorted(glob.glob(os.path.join(r, "hub/models--LiquidAI--LFM3-1.7B-Base/snapshots/*/config.json")))
        if hits:
            return os.path.dirname(hits[0])
    return None


MODEL_DIR = _model_dir()
needs_model = pytest.mark.skipif(MODEL_DIR is None, reason="LFM3-1.7B-Base weights not found (set LFM3_MODEL)")


# ---------------------------------------------------------------- reference quantizer (kv800.py, copied verbatim)
BLK, SUB = 256, 51200
KC_O, VB_O, VLO_O, VSC_O, KLO_O, KSC_O = 0, 32768, 49152, 49664, 50176, 50688
F8 = torch.float8_e4m3fn


def quantize_blocks(k, v, H):
    """k, v: (M, 256, 4, 128) bf16 (pre-RoPE K, unrotated V) -> (M, 51200) uint8 sub-blocks."""
    M = k.shape[0]
    out = torch.empty(M, SUB, dtype=torch.uint8, device=k.device)
    x = k.float()
    lo, hi = x.amin(1), x.amax(1)                                       # (M, 4, 128)
    lo8, sc8 = lo.to(F8), ((hi - lo) / 3).clamp_min(1e-8).to(F8)
    c = ((x - lo8.float()[:, None]) / sc8.float().clamp_min(1e-8)[:, None]).round().clamp(0, 3).to(torch.uint8)
    c = c.view(M, BLK, 4, 4, 32)
    out[:, KC_O:VB_O] = (c[..., 0, :] | (c[..., 1, :] << 2) | (c[..., 2, :] << 4) | (c[..., 3, :] << 6)).reshape(M, -1)
    out[:, KLO_O:KSC_O] = lo8.view(torch.uint8).reshape(M, -1)
    out[:, KSC_O:SUB] = sc8.view(torch.uint8).reshape(M, -1)
    y = (v.float() @ H).view(M, BLK, 2, 256)
    lo, hi = y.amin(-1), y.amax(-1)
    lo8, sc8 = lo.to(F8), (hi - lo).clamp_min(1e-8).to(F8)
    bb = ((y - lo8.float()[..., None]) / sc8.float().clamp_min(1e-8)[..., None]).round().clamp(0, 1).to(torch.uint8)
    bb = bb.view(M, BLK, 4, 8, 16)
    out[:, VB_O:VLO_O] = sum(bb[..., i, :] << i for i in range(8)).to(torch.uint8).reshape(M, -1)
    out[:, VLO_O:VSC_O] = lo8.view(torch.uint8).reshape(M, -1)
    out[:, VSC_O:KLO_O] = sc8.view(torch.uint8).reshape(M, -1)
    return out


def dequantize_blocks(sb, H):
    """(M, 51200) uint8 -> pre-RoPE K (M, 256, 4, 128) bf16 (as the kernel sees it) and unrotated V (M, 256, 4, 128) fp32."""
    M = sb.shape[0]
    kc = sb[:, KC_O:VB_O].reshape(M, BLK, 4, 32)
    codes = torch.stack([(kc >> (2 * i)) & 3 for i in range(4)], 3).reshape(M, BLK, 4, 128).float()
    klo = sb[:, KLO_O:KSC_O].contiguous().view(F8).float().view(M, 1, 4, 128)
    ksc = sb[:, KSC_O:SUB].contiguous().view(F8).float().view(M, 1, 4, 128)
    k = (codes * ksc + klo).to(torch.bfloat16)
    vb = sb[:, VB_O:VLO_O].reshape(M, BLK, 4, 16)
    bits = torch.stack([(vb >> i) & 1 for i in range(8)], 3).reshape(M, BLK, 4, 128).float()
    vlo = sb[:, VLO_O:VSC_O].contiguous().view(F8).float().view(M, BLK, 2, 1).repeat_interleave(2, 2)
    vsc = sb[:, VSC_O:KLO_O].contiguous().view(F8).float().view(M, BLK, 2, 1).repeat_interleave(2, 2)
    v = (bits * vsc + vlo) @ H
    return k, v


def _probe_mask(T, kind):
    """lfm3kv/qat_probe.py _mask (R = 128), copied."""
    i = torch.arange(T)[:, None]; j = torch.arange(T)[None]
    d = i - j
    if kind == "quant":
        return torch.cat(((d >= 0) & (d < 128), d >= 128), 1)
    return (d >= 0) & ((d < 256) | (j < 4))


# ---------------------------------------------------------------- CPU tests (no model)
def test_quantizer_matches_cache():
    g = torch.Generator().manual_seed(0)
    k = (torch.randn(3, 256, 4, 128, generator=g) * torch.rand(1, 1, 4, 128, generator=g) * 4 + 0.3).to(torch.bfloat16)
    v = (torch.randn(3, 256, 4, 128, generator=g) * 2).to(torch.bfloat16)
    H = kt.hadamard(128)
    kr, vr = dequantize_blocks(quantize_blocks(k, v, H), H)
    kq = kt.fake_quant_k(k.float().reshape(3, 256, 512)).reshape(3, 256, 4, 128).to(torch.bfloat16)
    assert torch.equal(kq, kr)
    vq = kt.fake_quant_v(v.float(), H)
    assert torch.allclose(vq, vr, atol=1e-5, rtol=0)
    # the layout path (any T, pads to a multiple of 256) gives the same K on aligned groups
    lay = kt._Layout(1, 700, "cpu", "cache")
    kk = k.reshape(1, 768, 4, 128)[:, :700]
    out = lay.fq_k(kk)
    assert torch.equal(out[:, :512], kr.reshape(1, 768, 4, 128)[:, :512])


@pytest.mark.parametrize("T", [512, 1024])
def test_last128_masks_equal_probe(T):
    lay = kt._Layout(1, T, "cpu", "last128")
    assert torch.equal(lay.mask("quant")[0, 0], _probe_mask(T, "quant"))
    assert torch.equal(lay.mask("win")[0, 0], _probe_mask(T, "win"))


def test_cache_rule_semantics():
    """Query at real index i reads key j quantized iff j < 256 * ((i + 1 - 128) // 256) (vLLM flush rule)."""
    T = 1100
    m = kt._Layout(1, T, "cpu", "cache").mask("quant")[0, 0]
    ex, qu = m[:, :T], m[:, T:]
    assert not (ex & qu).any()
    assert torch.equal(ex | qu, torch.ones(T, T, dtype=torch.bool).tril())
    for i in (0, 382, 383, 384, 638, 639, 1099):
        nq = max(0, i + 1 - 128) // 256
        assert int(qu[i].sum()) == 256 * nq and (not nq or bool(qu[i, 256 * nq - 1]))
        assert int(ex[i].sum()) == i + 1 - 256 * nq


@pytest.mark.parametrize("rule", ["cache", "last128"])
def test_padding_and_packing_alignment(rule):
    """Left padding and packed documents give the same masks and quantized K, per sequence, as unpadded rows."""
    g = torch.Generator().manual_seed(1)
    P, T = 37, 600
    am = torch.ones(2, P + T, dtype=torch.long); am[1, :P] = 0                      # row 1 left-padded
    lay = kt._Layout(2, P + T, "cpu", rule, attention_mask=am)
    ref = kt._Layout(1, T, "cpu", rule)
    for kind in ("quant", "win"):
        mb, mr = lay.mask(kind), ref.mask(kind)
        n = mb.shape[-1] // (P + T)
        sub = mb[1, 0, P:].reshape(T, n, P + T)[:, :, P:].reshape(T, n * T)
        assert torch.equal(sub, mr[0, 0])
        assert not mb[1, 0, P:].reshape(T, n, P + T)[:, :, :P].any()                # pads never visible
        assert bool(mb.any(-1).all())                                                # no all-masked query row
    k = torch.randn(1, T, 4, 128, generator=g).to(torch.bfloat16)
    kp = torch.cat((torch.randn(1, P, 4, 128, generator=g).to(torch.bfloat16), k), 1)
    out = lay.fq_k(torch.cat((torch.cat((k, k[:, :P]), 1), kp), 0))
    assert torch.equal(out[1, P:], ref.fq_k(k)[0])
    # packing: docs of 400 and 200 tokens in one row via restarting position_ids
    pos = torch.cat((torch.arange(400), torch.arange(200)))[None]
    lp = kt._Layout(1, 600, "cpu", rule, position_ids=pos)
    for kind in ("quant", "win"):
        mb = lp.mask(kind)[0, 0]
        n = mb.shape[-1] // 600
        mb = mb.reshape(600, n, 600)
        for s, e in ((0, 400), (400, 600)):
            r = kt._Layout(1, e - s, "cpu", rule).mask(kind)[0, 0].reshape(e - s, n, e - s)
            assert torch.equal(mb[s:e, :, s:e], r)
        assert not mb[400:, :, :400].any()


# ---------------------------------------------------------------- model tests
def eval_passages(tok):
    """lfm3kv/qat_probe.py eval_passages, copied."""
    srcs = {"code:inspect": "/usr/lib64/python3.14/inspect.py", "code:typing": "/usr/lib64/python3.14/typing.py",
            "prose:lo-license": "/usr/share/licenses/libreoffice-data/LICENSE",
            "prose:man-db": "/usr/share/doc/man-db/man-db-manual.txt"}
    if not all(os.path.exists(p) for p in srcs.values()):
        pytest.skip("RECOVERY PROBE eval passage files not on this machine")
    out = []
    for k, p in srcs.items():
        t = open(p).read()
        if k == "prose:man-db":
            t = re.sub(r"\x1b\[[0-9;]*m", "", t)
        ids = tok(t, add_special_tokens=False)["input_ids"]; st = min(2000, max(0, len(ids) - 4096))
        out.append((k, ids[st:st + 4096]))
    return out


@pytest.fixture(scope="session")
def base():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kt.apply_transformers_compat()
    dt = torch.float32 if DEV == "cpu" else torch.bfloat16
    m = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dt, device_map=DEV, trust_remote_code=True,
                                             attn_implementation="sdpa")
    m.requires_grad_(False)
    m.model.rotary_emb = type(m.model.rotary_emb)(m.config, device=DEV)     # fp32 inv_freq (as the probe's loader)
    m.eval()
    tok = AutoTokenizer.from_pretrained(os.environ.get("LFM3_TOKENIZER", MODEL_DIR), trust_remote_code=True)
    return m, tok


@pytest.fixture(scope="session")
def peft_model(base):
    """The probe measured step 0 on the PEFT-wrapped model (LoRA r32 on attention q/k/v/o, B = 0)."""
    from peft import LoraConfig, get_peft_model
    m, tok = base
    lt = m.config.layer_types
    tm = [f"layers.{l}.self_attn.{p}" for l in range(len(lt)) if lt[l] != "conv" for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
    pm = get_peft_model(m, LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0, target_modules=tm))
    pm.eval()
    return pm, tok


def _ids(tok, n, which=0):
    srcs = ["/usr/lib64/python3.14/typing.py", "/usr/lib64/python3.14/inspect.py"]
    srcs = [p for p in srcs if os.path.exists(p)]
    text = open(srcs[which % len(srcs)]).read() if srcs else ("def f(x):\n    return x + 1\n" * 2000)
    return tok(text, add_special_tokens=False)["input_ids"][1000:1000 + n]


@needs_model
@torch.no_grad()
def test_stock_mode_equals_unpatched(base):
    m, tok = base
    x = torch.tensor([_ids(tok, 1024), _ids(tok, 1024, 1)], device=DEV)
    ref = m(input_ids=x, use_cache=False).logits
    hook = kt.patch_for_training(m, mode="stock")
    try:
        out = m(input_ids=x, use_cache=False).logits
    finally:
        hook.remove()
    dmax = (out.float() - ref.float()).abs().max().item()
    print(f"\n[stock vs unpatched] bitwise equal {torch.equal(out, ref)}, max |dlogit| {dmax:.3g}")
    assert torch.equal(out, ref) or dmax < 0.02                # same code path; tolerance only for kernel nondeterminism
    for l in kt.QUANT_LAYERS + kt.WINDOW_LAYERS:
        assert "forward" not in m.model.layers[l].self_attn.__dict__
    assert (m(input_ids=x, use_cache=False).logits.float() - ref.float()).abs().max().item() <= max(dmax, 0.02)


@torch.no_grad()
def _ce(model, ps):
    res = []
    for _, ids in ps:
        x = torch.tensor([ids], device=DEV)
        logits = model(input_ids=x, use_cache=False).logits[0, -2049:-1].float()
        res.append(F.cross_entropy(logits, x[0, -2048:]).item())
    return sum(res) / len(res)


@needs_model
def test_800h_ce_matches_probe_reference(peft_model):
    pm, tok = peft_model
    ps = eval_passages(tok)
    hook = kt.patch_for_training(pm, mode="stock", rule="last128")
    try:
        s = _ce(pm, ps)
        hook.set_mode("800h")
        q = _ce(pm, ps)
    finally:
        hook.remove()
    hook = kt.patch_for_training(pm, rule="cache")
    try:
        c = _ce(pm, ps)
    finally:
        hook.remove()
    print(f"\n[CE, 4 eval passages, last 2K of 4K] stock {s:.4f} (ref {REF_STOCK}) | 800h last128 {q:.4f} "
          f"(ref {REF_800H}) | 800h cache rule (sampler) {c:.4f}")
    assert abs(s - REF_STOCK) < 0.01
    assert abs(q - REF_800H) < 0.01
    assert s - 0.01 < c < q + 0.01


@needs_model
def test_padding_packing_prefill_and_odd_lengths(peft_model):
    pm, tok = peft_model
    pad = pm.config.pad_token_id if pm.config.pad_token_id is not None else 0
    a, b = _ids(tok, 1000), _ids(tok, 700, 1)
    x = torch.tensor([a, [pad] * 300 + b], device=DEV)
    am = torch.ones_like(x); am[1, :300] = 0

    def lp(logits, ids):                                       # log p(next token), (T - 1,)
        return logits.float().log_softmax(-1)[:-1].gather(-1, ids[1:, None])[:, 0]

    hook = kt.patch_for_training(pm)
    try:
        diffs = {}
        with torch.no_grad():
            for mode in ("stock", "800h"):
                hook.set_mode(mode)
                lb = pm(input_ids=x, attention_mask=am, use_cache=False).logits
                assert torch.isfinite(lb[0]).all() and torch.isfinite(lb[1, 300:]).all()
                l0 = pm(input_ids=x[:1], use_cache=False).logits[0]
                l1 = pm(input_ids=x[1:, 300:], use_cache=False).logits[0]
                xa, xb = x[0], x[1, 300:]
                d0 = (lp(lb[0], xa) - lp(l0, xa)).abs().mean().item()
                d1 = (lp(lb[1, 300:], xb) - lp(l1, xb)).abs().mean().item()
                # packed: the same two sequences in one row, position_ids restarting, no attention mask
                xp = torch.cat((xb, xa[:300]))[None]
                pos = torch.cat((torch.arange(700), torch.arange(300))).to(DEV)[None]
                lpk = pm(input_ids=xp, position_ids=pos, use_cache=False).logits[0]
                l2 = pm(input_ids=xa[None, :300], use_cache=False).logits[0]
                d2 = (lp(lpk[:700], xb) - lp(l1, xb)).abs().mean().item()
                d3 = (lp(lpk[700:], xa[:300]) - lp(l2, xa[:300])).abs().mean().item()
                diffs[mode] = (d0, d1, d2, d3)
            print(f"\n[batch/pad/pack vs single] mean |dlogp| (row0, left-padded row1, packed doc1, packed doc2): "
                  f"stock {['%.4f' % d for d in diffs['stock']]} | 800h {['%.4f' % d for d in diffs['800h']]}")
            for ds, dq in zip(diffs["stock"], diffs["800h"]):
                assert dq <= 3 * ds + 0.02
            # prefill boundary: columns < 383 never see quantized keys; prefill columns >= 383 now see exact ones
            hook.set_mode("800h")
            with hook.prefill(768):
                lpf = pm(input_ids=x[:1], use_cache=False).logits[0].float()
            l0 = l0.float()
            early = (lpf[:383] - l0[:383]).abs().mean().item()
            mid = (lpf[400:768] - l0[400:768]).abs().mean().item()
            print(f"[prefill 768] mean |dlogit| cols < 383: {early:.3g} | cols 400-767: {mid:.3g}")
            assert early < 0.01 and mid > 5 * early + 1e-3
        # backward through padding (pad query rows must not produce NaNs), odd T
        pm.train()
        labels = x.masked_fill(am == 0, -100)
        loss = pm(input_ids=x, attention_mask=am, labels=labels, use_cache=False).loss
        loss.backward()
        grads = [p.grad for n, p in pm.named_parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        pm.zero_grad(set_to_none=True)
    finally:
        pm.eval()
        hook.remove()


@needs_model
def test_backward_finite_lora_grads(peft_model):
    pm, tok = peft_model
    hook = kt.patch_for_training(pm, mode="800h")
    try:
        pm.train()
        x = torch.tensor([_ids(tok, 600)], device=DEV)
        loss = pm(input_ids=x, labels=x, use_cache=False).loss
        loss.backward()
        named = [(n, p) for n, p in pm.named_parameters() if p.requires_grad]
        assert named and all("lora_" in n for n, _ in named)
        bad = [n for n, p in named if p.grad is None or not torch.isfinite(p.grad).all()]
        assert not bad, bad[:5]
        for l in kt.QUANT_LAYERS + kt.WINDOW_LAYERS:
            gb = [p.grad.abs().sum().item() for n, p in named if f"layers.{l}.self_attn" in n and "lora_B" in n]
            assert gb and max(gb) > 0, f"layer {l}: no LoRA-B gradient"
        print(f"\n[backward] 800h loss {loss.item():.3f}, {len(named)} LoRA tensors, all grads finite")
        pm.zero_grad(set_to_none=True)
    finally:
        pm.eval()
        hook.remove()
