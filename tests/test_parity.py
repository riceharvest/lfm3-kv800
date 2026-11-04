"""Parity of lfm3_kv800.kv800.decode against a plain torch reference, per backend (12 cases: quantized layer at
n = 50 .. 5000, window + sinks layer at n = 3 .. 1500; B = 2; the last 5 tokens appended one at a time). Pass: relative
error < 1e-2 on every case.

    python tests/test_parity.py [--device xpu|cuda|cpu] [sycl] [triton] [torch]   (strict: a missing backend fails)
    pytest tests/test_parity.py                                                (skips backends this machine lacks)

The reference quantizes K per (256-token block, head, channel) to 2 bits and Hadamard-rotated V per (token, 2-head
group) to 1 bit, min/max with fp8 e4m3 lo / scale, applies RoPE to the dequantized pre-RoPE keys in fp32 with a bf16
cast, and runs full softmax attention (window layers: 4 sinks + the last 256 tokens)."""
import argparse, math, sys

import torch

from lfm3_kv800 import kv800

QUANT_NS = (50, 383, 384, 700, 1500, 5000)
WINDOW_NS = (3, 10, 200, 259, 260, 1500)


def default_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------- reference
def rope_table(inv_freq, max_pos, device):
    """cos / sin (max_pos, 32) bf16 of the 32 rotating frequencies (fp32 product, bf16 cast)."""
    fr = torch.arange(max_pos, device=device, dtype=torch.float32)[:, None] * inv_freq[None, :32].to(device, torch.float32)
    return fr.cos().to(torch.bfloat16).contiguous(), fr.sin().to(torch.bfloat16).contiguous()


def _aq(x, bits, dim):
    """Asymmetric min/max quantize-dequantize along `dim`, lo and scale rounded to fp8 e4m3."""
    f8 = lambda t: t.to(torch.float8_e4m3fn).to(t.dtype)
    lo, hi = x.amin(dim, keepdim=True), x.amax(dim, keepdim=True)
    sc = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-8)
    lo, sc = f8(lo), f8(sc).clamp_min(1e-8)
    return ((x - lo) / sc).round().clamp(0, 2 ** bits - 1) * sc + lo


def reference(q, Kh, Vh, cos_full, sin_full, window=0):
    """q (B,16,128) bf16 post-RoPE; Kh (B,n,4,128) pre-RoPE bf16; Vh (B,n,4,128) bf16; cos/sin_full (n,128) bf16.
    -> (B,16,128) fp32."""
    B, n = Kh.shape[:2]; H = kv800.hadamard(128, q.device)
    nq = 0 if window else max(0, (n - kv800.LAG) // kv800.BLK)
    K, V = Kh.float().clone(), Vh.float().clone()
    if nq:
        x = Kh[:, :nq * kv800.BLK].float().reshape(B, nq, kv800.BLK, 512)
        K[:, :nq * kv800.BLK] = _aq(x, 2, 2).view(B, nq * kv800.BLK, 4, 128).to(torch.bfloat16).float()
        y = (Vh[:, :nq * kv800.BLK].float() @ H).reshape(B, nq * kv800.BLK, 2, 256)
        V[:, :nq * kv800.BLK] = _aq(y, 1, -1).view(B, nq * kv800.BLK, 4, 128) @ H
    cs, sn = cos_full[:n].float()[None, :, None], sin_full[:n].float()[None, :, None]
    rot = torch.cat((-K[..., 64:], K[..., :64]), -1)
    Kr = (K * cs + rot * sn).to(torch.bfloat16).float()
    Kr, V = Kr.repeat_interleave(4, 2), V.repeat_interleave(4, 2)
    sc = torch.einsum("bhd,bnhd->bhn", q.float(), Kr) * 128 ** -0.5
    if window:
        p = torch.arange(n, device=q.device)
        sc = sc.masked_fill(~((p >= n - window) | (p < kv800.NSINK))[None, None], float("-inf"))
    return torch.einsum("bhn,bnhd->bhd", sc.softmax(-1), V)


def tables(device):
    inv = 1.0 / (1e6 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 128))      # LFM3 full layers: theta 1e6
    c32, s32 = rope_table(torch.cat((inv, torch.zeros(32))), 8192, device)
    fr = torch.arange(8192, device=device, dtype=torch.float32)[:, None] * inv.to(device)[None]
    c64 = torch.cat((fr.cos(), torch.ones(8192, 32, device=device)), 1)
    s64 = torch.cat((fr.sin(), torch.zeros(8192, 32, device=device)), 1)
    return c32, s32, torch.cat((c64, c64), 1).bfloat16(), torch.cat((s64, s64), 1).bfloat16()


def cases():
    return [(0, n) for n in QUANT_NS] + [(kv800.WIN, n) for n in WINDOW_NS]


def run_case(backend, device, win, n, tabs, seed=0):
    """-> (relative error, backend that ran)."""
    c32, s32, cf, sf = tabs
    g = torch.Generator(device="cpu").manual_seed(seed * 100003 + n * 7 + win)
    B = 2
    Kh = (torch.randn(B, n, 4, 128, generator=g) + 3 * torch.randn(1, 1, 4, 128, generator=g)).bfloat16().to(device)
    Vh = torch.randn(B, n, 4, 128, generator=g).bfloat16().to(device)
    q = (2 * torch.randn(B, 16, 128, generator=g)).bfloat16().to(device)
    st = kv800.PagedStore(B, n + 8, device, window=win)
    pre = max(1, n - 5); st.append(Kh[:, :pre], Vh[:, :pre])
    for i in range(pre, n):
        st.append(Kh[:, i:i + 1], Vh[:, i:i + 1])
    kv800.set_backend(backend)
    o = kv800.decode(st.paged(), q, c32, s32, st.H).float()
    ran = kv800.get_backend()
    r = reference(q, Kh, Vh, cf, sf, window=win)
    return ((o - r).norm() / r.norm()).item(), ran


def check(backend, device):
    tabs = tables(device)
    ok = True
    for win, n in cases():
        rel, ran = run_case(backend, device, win, n, tabs)
        good = rel < 1e-2 and math.isfinite(rel) and ran == backend
        ok &= good
        print(f"[{backend} on {device}, ran {ran}] parity win={win} n={n}: rel {rel:.2e} {'ok' if good else 'FAIL'}",
              flush=True)
    return ok


# ---------------------------------------------------------------- pytest
try:
    import pytest

    @pytest.mark.parametrize("backend", kv800.BACKENDS)
    def test_parity(backend):
        dev = default_device()
        ok, why = kv800._available(backend, dev)
        if not ok:
            pytest.skip(f"{backend} not available on {dev}: {why}")
        try:
            assert check(backend, dev)
        finally:
            kv800.set_backend(None)
except ImportError:
    pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("backends", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = a.device or default_device()
    bes = a.backends or [b for b in kv800.BACKENDS if kv800._available(b, dev)[0]]
    allok = True
    for be in bes:
        ok = check(be, dev)
        print(f"PARITY {be} on {dev}: {'OK' if ok else 'FAILED'}", flush=True)
        allok &= ok
    sys.exit(0 if allok else 1)
