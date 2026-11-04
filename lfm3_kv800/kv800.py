"""Paged 800h KV cache (the layout vLLM pages hold) + Triton decode kernels + torch quantize/prefill helpers.

Quantized layer state = 256-token SUB-BLOCKS of 51,200 bytes each, addressed by a per-sequence table of byte offsets
(BOFF[b, j] = byte offset of sub-block j of sequence b from the quant pool's base pointer). Sub-block layout:
  [0, 32768)      kc   [256, 4, 32]  planar 2-bit K (byte j = dims j, j+32, j+64, j+96 at bits 0-1/2-3/4-5/6-7)
  [32768, 49152)  vb   [256, 4, 16]  planar 1-bit V after per-head Hadamard (byte j = dims j + 16i at bit i)
  [49152, 49664)  vlo  [256, 2] fp8   per (token, 2-head group)
  [49664, 50176)  vsc  [256, 2] fp8
  [50176, 50688)  klo  [4, 128] fp8   per (head, channel) over the 256 tokens, PRE-RoPE
  [50688, 51200)  ksc  [4, 128] fp8
Exact tokens (pre-RoPE K, unrotated V, bf16) live in pages of EBS tokens addressed by EOFF[b, p // EBS] (element offset
of the page in the exact pool); inside a page: K of (head h, slot t) at + h*e_hs + t*e_ts, V at + e_v from K.
Sinks (window layers): K/V of positions 0..3 at element offsets SKOFF[b] / SVOFF[b] in their pools, layout [4, 4, 128].
Semantics: sub-block j of a full layer is quantized once n >= 256 j + 384 (the newest 128..383 tokens stay exact);
window layers = 4 sinks + the last 256 tokens, all exact.

decode() picks a backend per device type (logged once): xpu: sycl (prebuilt op) -> triton -> torch; cuda: triton ->
torch; other devices: torch. Env LFM3KV_BACKEND=sycl|triton|torch|auto (or set_backend()) forces one; a forced backend
that is unavailable raises instead of falling back."""
import math, os
import torch
from . import get_logger

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:                                    # CPU-only installs: torch backend only
    triton = tl = None
    HAVE_TRITON = False

BLK, LAG, NSINK, WIN = 256, 128, 4, 256
SUB = 51200
KC_O, VB_O, VLO_O, VSC_O, KLO_O, KSC_O = 0, 32768, 49152, 49664, 50176, 50688
F8 = torch.float8_e4m3fn
LOG2E = 1.4426950408889634
QBT, XBT, QW, QWS, TARGET = 16, 16, 1, 16, 1024      # QW / QWS (num_warps / warp_size): XPU launches only
QW_OTHER = None                                      # num_warps on CUDA etc. (None = Triton default)
BACKENDS = ("sycl", "triton", "torch")
_log = get_logger()


def hadamard(n=128, device=None):
    H = torch.ones(1, 1)
    while H.shape[0] < n: H = torch.cat((torch.cat((H, H), 1), torch.cat((H, -H), 1)), 0)
    return (H / n ** 0.5).to(device)


def flat(t):
    """1-D view of t's storage starting at t's first element (t may be a strided view into a shared buffer)."""
    n = t.untyped_storage().nbytes() // t.element_size() - t.storage_offset()
    return torch.as_strided(t, (n,), (1,), t.storage_offset())


# ---------------------------------------------------------------- torch quantizer
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


# ---------------------------------------------------------------- Triton decode kernels (SIMT, one 16-lane sub-group)
if HAVE_TRITON:
    @triton.jit
    def _f8(p):
        """fp8 e4m3fn byte -> fp32, exact (no NaN bytes in the scales). Integer decode: Triton's fp8e4nv type needs
        sm_89+ on CUDA."""
        b = tl.load(p).to(tl.int32); e = (b >> 3) & 15; m = b & 7; s = (b >> 7) & 1
        nrm = ((s << 31) | ((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
        sub = m.to(tl.float32) * 0.001953125
        return tl.where(e == 0, tl.where(s == 1, -sub, sub), nrm)


    @triton.jit
    def _pick(x, rr, r: tl.constexpr):
        return tl.sum(tl.where(rr[:, None] == r, x, 0.0), 0)


    @triton.jit
    def _pv(p, vs, alpha, rr, r: tl.constexpr, bits3, acc):
        pr = _pick(p, rr, r) * vs
        a = tl.sum(tl.where(rr == r, alpha, 0.0), 0)
        return acc * a + tl.sum(pr[:, None, None] * bits3, 0)


    @triton.jit
    def _qdec(Q, QP, BOFF, boff_stride, NQ, COS, SIN, MO, LO_, AO, n_split, spb, scale, BT: tl.constexpr):
        """Program (b*4 + kv head, split): flushed sub-blocks [s*spb, min((s+1)*spb, NQ[b])) -> partial (m, l, acc_rot)."""
        bh = tl.program_id(0); s = tl.program_id(1); b = bh // 4; h = bh % 4
        rr = tl.arange(0, 4); j = tl.arange(0, 32); t = tl.arange(0, BT); j16 = tl.arange(0, 16); i8 = tl.arange(0, 8)
        qp = Q + (b * 16 + h * 4 + rr[:, None]) * 128 + j[None, :]
        q0 = tl.load(qp).to(tl.float32)[:, None, :]; q1 = tl.load(qp + 32).to(tl.float32)[:, None, :]
        q2 = tl.load(qp + 64).to(tl.float32)[:, None, :]; q3 = tl.load(qp + 96).to(tl.float32)[:, None, :]
        m_i = tl.full([4], -1e30, tl.float32); l_i = tl.zeros([4], tl.float32); lv = tl.zeros([4], tl.float32)
        a0 = tl.zeros([8, 16], tl.float32); a1 = tl.zeros([8, 16], tl.float32)
        a2 = tl.zeros([8, 16], tl.float32); a3 = tl.zeros([8, 16], tl.float32)
        sh8 = i8[None, :, None]
        nq = tl.load(NQ + b)
        j0 = s * spb; j1 = tl.minimum((s + 1) * spb, nq)
        for jb in range(j0, j1):
            base = tl.load(BOFF + b * boff_stride + jb)                                   # int64 byte offset
            so = base + h * 128
            sc0 = _f8(QP + so + 50688 + j)[None, :]; lo0 = _f8(QP + so + 50176 + j)[None, :]
            sc2 = _f8(QP + so + 50688 + 64 + j)[None, :]; lo2 = _f8(QP + so + 50176 + 64 + j)[None, :]
            sc1 = _f8(QP + so + 50688 + 32 + j)[None, None, :]; lo1 = _f8(QP + so + 50176 + 32 + j)[None, None, :]
            sc3 = _f8(QP + so + 50688 + 96 + j)[None, None, :]; lo3 = _f8(QP + so + 50176 + 96 + j)[None, None, :]
            qs1 = q1 * sc1; qs3 = q3 * sc3
            c13 = tl.sum(tl.sum(q1 * lo1 + q3 * lo3, 2), 1)
            for t0 in range(0, 256, BT):
                pos = jb * 256 + t0 + t
                tr = t0 + t                                                               # token row in the sub-block
                kb = tl.load(QP + base + (tr[:, None] * 4 + h) * 32 + j[None, :]).to(tl.int32)
                cs = tl.load(COS + pos[:, None] * 32 + j[None, :]).to(tl.float32)
                sn = tl.load(SIN + pos[:, None] * 32 + j[None, :]).to(tl.float32)
                k0 = (kb & 3).to(tl.float32) * sc0 + lo0
                k2 = ((kb >> 4) & 3).to(tl.float32) * sc2 + lo2
                kr0 = k0 * cs - k2 * sn
                kr2 = k2 * cs + k0 * sn
                c1 = ((kb >> 2) & 3).to(tl.float32); c3 = ((kb >> 6) & 3).to(tl.float32)
                sc = tl.sum(q0 * kr0[None, :, :] + q2 * kr2[None, :, :] + qs1 * c1[None, :, :] + qs3 * c3[None, :, :], 2) + c13[:, None]
                s2 = sc * scale
                m_new = tl.maximum(m_i, tl.max(s2, 1))
                alpha = tl.exp2(m_i - m_new)
                p = tl.exp2(s2 - m_new[:, None])
                l_i = l_i * alpha + tl.sum(p, 1); m_i = m_new
                vl = _f8(QP + base + 49152 + tr * 2 + h // 2); vs = _f8(QP + base + 49664 + tr * 2 + h // 2)
                lv = lv * alpha + tl.sum(p * vl[None, :], 1)
                vb = tl.load(QP + base + 32768 + (tr[:, None] * 4 + h) * 16 + j16[None, :]).to(tl.int32)
                bits3 = ((vb[:, None, :] >> sh8) & 1).to(tl.float32)
                a0 = _pv(p, vs, alpha, rr, 0, bits3, a0); a1 = _pv(p, vs, alpha, rr, 1, bits3, a1)
                a2 = _pv(p, vs, alpha, rr, 2, bits3, a2); a3 = _pv(p, vs, alpha, rr, 3, bits3, a3)
        o = b * (n_split + 1) + s
        row = o * 16 + h * 4 + rr
        tl.store(MO + row, m_i); tl.store(LO_ + row, l_i)
        ap = AO + (o * 16 + h * 4) * 128 + i8[:, None] * 16 + j16[None, :]
        tl.store(ap, a0 + tl.sum(tl.where(rr == 0, lv, 0.0), 0)); tl.store(ap + 128, a1 + tl.sum(tl.where(rr == 1, lv, 0.0), 0))
        tl.store(ap + 256, a2 + tl.sum(tl.where(rr == 2, lv, 0.0), 0)); tl.store(ap + 384, a3 + tl.sum(tl.where(rr == 3, lv, 0.0), 0))


    @triton.jit
    def _xtile(q0, q1, q2, q3, KB, VB_, kof, vof, pos, valid, COS, SIN, m_i, l_i, a0, a1, a2, a3, scale, rr, j, d):
        """SIMT tile of exact bf16 tokens; kof / vof (BT,) = element offsets of each token's 128-dim K / V row."""
        mk = valid[:, None]
        kp = KB + kof[:, None] + j[None, :]
        k0 = tl.load(kp, mask=mk, other=0.0).to(tl.float32); k1 = tl.load(kp + 32, mask=mk, other=0.0).to(tl.float32)
        k2 = tl.load(kp + 64, mask=mk, other=0.0).to(tl.float32); k3 = tl.load(kp + 96, mask=mk, other=0.0).to(tl.float32)
        cs = tl.load(COS + pos[:, None] * 32 + j[None, :], mask=mk, other=1.0).to(tl.float32)
        sn = tl.load(SIN + pos[:, None] * 32 + j[None, :], mask=mk, other=0.0).to(tl.float32)
        kr0 = (k0 * cs - k2 * sn).to(tl.bfloat16).to(tl.float32)        # HF: RoPE in fp32, keys cast to bf16
        kr2 = (k2 * cs + k0 * sn).to(tl.bfloat16).to(tl.float32)
        sc = tl.sum(q0 * kr0[None, :, :] + q1 * k1[None, :, :] + q2 * kr2[None, :, :] + q3 * k3[None, :, :], 2)
        s2 = tl.where(valid[None, :], sc * scale, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s2, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s2 - m_new[:, None])
        v = tl.load(VB_ + vof[:, None] + d[None, :], mask=mk, other=0.0).to(tl.float32)   # (BT, 128)
        a0 = a0 * tl.sum(tl.where(rr == 0, alpha, 0.0), 0) + tl.sum(_pick(p, rr, 0)[:, None] * v, 0)
        a1 = a1 * tl.sum(tl.where(rr == 1, alpha, 0.0), 0) + tl.sum(_pick(p, rr, 1)[:, None] * v, 0)
        a2 = a2 * tl.sum(tl.where(rr == 2, alpha, 0.0), 0) + tl.sum(_pick(p, rr, 2)[:, None] * v, 0)
        a3 = a3 * tl.sum(tl.where(rr == 3, alpha, 0.0), 0) + tl.sum(_pick(p, rr, 3)[:, None] * v, 0)
        return m_new, l_i * alpha + tl.sum(p, 1), a0, a1, a2, a3


    @triton.jit
    def _xdec(Q, EP, EOFF, eoff_stride, ebs, e_hs, e_ts, e_v, SKP, SVP, SKOFF, SVOFF, N, NQ, COS, SIN, MO, LO_, AO,
              n_split, scale, WINC: tl.constexpr, NSK: tl.constexpr, BT: tl.constexpr):
        """Program b*4 + kv head: sinks (positions < NSK below the window) + exact positions [lo_e, N[b])."""
        bh = tl.program_id(0); b = bh // 4; h = bh % 4
        rr = tl.arange(0, 4); j = tl.arange(0, 32); t = tl.arange(0, BT); d = tl.arange(0, 128)
        qp = Q + (b * 16 + h * 4 + rr[:, None]) * 128 + j[None, :]
        q0 = tl.load(qp).to(tl.float32)[:, None, :]; q1 = tl.load(qp + 32).to(tl.float32)[:, None, :]
        q2 = tl.load(qp + 64).to(tl.float32)[:, None, :]; q3 = tl.load(qp + 96).to(tl.float32)[:, None, :]
        m_i = tl.full([4], -1e30, tl.float32); l_i = tl.zeros([4], tl.float32)
        a0 = tl.zeros([128], tl.float32); a1 = tl.zeros([128], tl.float32)
        a2 = tl.zeros([128], tl.float32); a3 = tl.zeros([128], tl.float32)
        n = tl.load(N + b); lo_e = tl.load(NQ + b) * 256
        if WINC > 0:
            lo_e = tl.maximum(lo_e, n - WINC)
        if NSK > 0:
            ts = tl.arange(0, BT)
            sv = (ts < NSK) & (ts < lo_e) & (ts < n)
            ko = tl.load(SKOFF + b) + (ts * 4 + h) * 128
            vo = tl.load(SVOFF + b) + (ts * 4 + h) * 128
            m_i, l_i, a0, a1, a2, a3 = _xtile(q0, q1, q2, q3, SKP, SVP, ko, vo, ts, sv, COS, SIN, m_i, l_i, a0, a1, a2, a3, scale, rr, j, d)
        for p0 in range(lo_e, n, BT):
            pos = p0 + t
            valid = pos < n
            pg = tl.load(EOFF + b * eoff_stride + pos // ebs, mask=valid, other=0)
            ko = pg + h * e_hs + (pos % ebs) * e_ts
            m_i, l_i, a0, a1, a2, a3 = _xtile(q0, q1, q2, q3, EP, EP, ko, ko + e_v, pos, valid, COS, SIN, m_i, l_i, a0, a1, a2, a3, scale, rr, j, d)
        o = b * (n_split + 1) + n_split
        row = o * 16 + h * 4 + rr
        tl.store(MO + row, m_i); tl.store(LO_ + row, l_i)
        ap = AO + (o * 16 + h * 4) * 128 + d
        tl.store(ap, a0); tl.store(ap + 128, a1); tl.store(ap + 256, a2); tl.store(ap + 384, a3)


def combine(MO, LO, AO, ns, H):
    w = torch.exp2(MO - MO.amax(1, keepdim=True))
    L = (w * LO).sum(1)
    o = w[:, ns, :, None] * AO[:, ns]
    if ns: o = o + (w[:, :ns, :, None] * AO[:, :ns]).sum(1) @ H
    return (o / L[..., None]).to(torch.bfloat16)


class Paged:
    """Everything the decode kernels read for one layer and one batch of decode sequences.
    qpool: uint8 tensor whose data_ptr is the byte base of BOFF; epool: bf16 tensor, base of EOFF (elements);
    sk/sv: bf16 tensors, bases of SKOFF/SVOFF (elements). N: seq lengths incl. the current token; NQ: flushed
    sub-blocks per sequence; nqmax: host int >= max(NQ). win: 0 = quantized layer, 256 = window+sinks layer."""
    __slots__ = ("qpool", "boff", "epool", "eoff", "ebs", "e_hs", "e_ts", "e_v", "sk", "sv", "skoff", "svoff",
                 "N", "NQ", "nqmax", "win")

    def __init__(self, **kw):
        for k in self.__slots__: setattr(self, k, kw.get(k))


# ---------------------------------------------------------------- backend selection
_FORCED = None          # set_backend(); None = env LFM3KV_BACKEND (default auto)
_CHOSEN = {}            # device type -> backend in use
_VERIFIED = set()       # device types whose auto-chosen backend has completed one decode
_LAST = [None]


def _requested():
    v = _FORCED if _FORCED is not None else os.environ.get("LFM3KV_BACKEND", "auto")
    v = (v or "auto").strip().lower()
    if v not in BACKENDS + ("auto",):
        raise ValueError(f"LFM3KV_BACKEND={v!r}: expected one of sycl, triton, torch, auto")
    return v


def set_backend(name=None):
    """Force a decode backend ('sycl' | 'triton' | 'torch'), or 'auto' / None to pick per device (env LFM3KV_BACKEND
    is read when this is None)."""
    global _FORCED
    _FORCED = None if name is None else str(name).strip().lower()
    _requested()
    _CHOSEN.clear(); _VERIFIED.clear()


def get_backend(device_type=None):
    """Backend used by the last decode() (or the one in use on `device_type`)."""
    return _CHOSEN.get(device_type) if device_type else _LAST[0]


def _available(name, dt):
    """(ok, detail) for backend `name` on device type `dt`."""
    if name == "torch":
        return True, "pure torch"
    if name == "triton":
        if not HAVE_TRITON:
            return False, "triton is not installed"
        if dt not in ("xpu", "cuda"):
            return False, f"no Triton backend for device type {dt!r}"
        return True, f"triton {triton.__version__}"
    if name == "sycl":
        if dt != "xpu":
            return False, f"the SYCL op runs on xpu only (device type {dt!r})"
        from . import kv800_sycl
        try:
            kv800_sycl.ops()
        except ImportError as e:
            return False, str(e)
        return True, kv800_sycl.SOURCE
    raise ValueError(name)


def _chain(dt):
    return {"xpu": ("sycl", "triton", "torch"), "cuda": ("triton", "torch")}.get(dt, ("torch",))


def _resolve(dt):
    if dt in _CHOSEN:
        return _CHOSEN[dt]
    req = _requested()
    if req != "auto":
        ok, why = _available(req, dt)
        if not ok:
            raise RuntimeError(f"800h decode backend {req!r} was requested (LFM3KV_BACKEND / set_backend) but is not "
                               f"available on {dt}: {why}")
        _log.info("decode backend on %s: %s (forced; %s)", dt, req, why)
        _CHOSEN[dt] = req
        return req
    skipped = []
    for name in _chain(dt):
        ok, why = _available(name, dt)
        if ok:
            _log.info("decode backend on %s: %s (%s)%s", dt, name, why,
                      "; skipped " + "; ".join(skipped) if skipped else "")
            _CHOSEN[dt] = name
            return name
        skipped.append(f"{name}: {why}")
    raise RuntimeError("no decode backend")          # unreachable: torch is always available


def decode(P, q, cos_t, sin_t, H, spb=None, target=None):
    """q: (B, 16, 128) bf16 post-RoPE -> (B, 16, 128) bf16. cos_t / sin_t: (P, 32) bf16 tables of the 32 rotating
    frequencies (dims d and d + 64 share frequency d, d < 32); H: 128x128 normalized Hadamard (kv800.hadamard)."""
    dt = q.device.type
    be = _resolve(dt)
    _LAST[0] = be
    if dt in _VERIFIED or _requested() != "auto":
        return _run(be, P, q, cos_t, sin_t, H, spb, target)
    try:                                             # auto mode: the first call may demote to the next backend
        out = _run(be, P, q, cos_t, sin_t, H, spb, target)
    except Exception as e:
        ch = _chain(dt)
        if be == ch[-1]:
            raise
        nxt = ch[ch.index(be) + 1]
        _log.warning("decode backend %s failed on %s (%s: %s); using %s", be, dt, type(e).__name__, str(e)[:300], nxt)
        _CHOSEN[dt] = nxt
        return decode(P, q, cos_t, sin_t, H, spb, target)
    _VERIFIED.add(dt)
    return out


def _run(be, P, q, cos_t, sin_t, H, spb, target):
    if be == "sycl":
        from . import kv800_sycl
        return kv800_sycl.decode(P, q, cos_t, sin_t, spb, target)
    if be == "torch":
        from . import torch_decode
        return torch_decode.decode(P, q, cos_t, sin_t, H)
    return _triton_decode(P, q, cos_t, sin_t, H, spb, target)


def _triton_decode(P, q, cos_t, sin_t, H, spb=None, target=None):
    B = q.shape[0]; q = q.contiguous()
    nqmax = 0 if P.win else P.nqmax
    target = target or TARGET
    if spb is None: spb = max(1, math.ceil(nqmax * B * 4 / target)) if nqmax else 1
    ns = math.ceil(nqmax / spb) if nqmax else 0
    MO = torch.empty(B, ns + 1, 16, device=q.device, dtype=torch.float32)
    LO = torch.empty_like(MO); AO = torch.empty(B, ns + 1, 16, 128, device=q.device, dtype=torch.float32)
    scale = 128 ** -0.5 * LOG2E
    if q.device.type == "xpu":                       # one 16-lane sub-group per program (XPU-only launch kwargs)
        kw = dict(num_warps=QW, warp_size=QWS)
    else:
        kw = {} if QW_OTHER is None else dict(num_warps=QW_OTHER)
    if ns:
        _qdec[(B * 4, ns)](q, P.qpool, P.boff, P.boff.stride(0), P.NQ, cos_t, sin_t, MO, LO, AO, ns, spb, scale,
                           BT=QBT, **kw)
    sk = P.sk if P.sk is not None else P.epool; sv = P.sv if P.sv is not None else P.epool
    skoff = P.skoff if P.skoff is not None else P.N; svoff = P.svoff if P.svoff is not None else P.N
    _xdec[(B * 4,)](q, P.epool, P.eoff, P.eoff.stride(0), P.ebs, P.e_hs, P.e_ts, P.e_v, sk, sv, skoff, svoff,
                    P.N, P.NQ, cos_t, sin_t, MO, LO, AO, ns, scale,
                    WINC=P.win, NSK=NSINK if (P.win and P.sk is not None) else 0, BT=XBT, **kw)
    return combine(MO, LO, AO, ns, H)


# ---------------------------------------------------------------- harness store with the paged layout
class PagedStore:
    """Test/bench store with the vLLM-style paged layout for `batch` equal-length sequences. Exact pages form a ring of
    `nring` pages of `ebs` tokens per sequence (the window manager's job in vLLM). Same flush rule as the vLLM cache."""

    def __init__(self, batch, max_len, device, window=0, ebs=64, nring=10):
        self.B, self.win, self.dev, self.ebs, self.nring = batch, window, device, ebs, nring
        self.H = hadamard(128, device)
        nsub = 0 if window else max_len // BLK + 1
        self.qpool = torch.zeros(max(1, batch * nsub), SUB, dtype=torch.uint8, device=device)
        self.boff = (torch.arange(batch * max(nsub, 1), device=device, dtype=torch.int64) * SUB).view(batch, max(nsub, 1))
        # exact pages: [B * nring, 4 heads, ebs tokens, K 128 | V 128] bf16
        self.epool = torch.zeros(batch * nring, 4, ebs, 256, dtype=torch.bfloat16, device=device)
        npg = max_len // ebs + 2
        pg = torch.arange(npg, device=device) % nring
        self.eoff = ((torch.arange(batch, device=device)[:, None] * nring + pg[None]) * (4 * ebs * 256)).to(torch.int64)
        self.e_hs, self.e_ts, self.e_v = ebs * 256, 256, 128
        self.sk = torch.zeros(batch, NSINK, 4, 128, dtype=torch.bfloat16, device=device)
        self.sv = torch.zeros_like(self.sk)
        self.skoff = torch.arange(batch, device=device, dtype=torch.int64) * NSINK * 512
        self.N = torch.zeros(batch, dtype=torch.int32, device=device); self.NQ = torch.zeros_like(self.N)
        self.n, self.nq = 0, 0
        assert nring * ebs >= 512 + ebs

    def paged(self):
        return Paged(qpool=self.qpool, boff=self.boff, epool=self.epool, eoff=self.eoff // 1, ebs=self.ebs,
                     e_hs=self.e_hs, e_ts=self.e_ts, e_v=self.e_v, sk=self.sk if self.win else None,
                     sv=self.sv if self.win else None, skoff=self.skoff, svoff=self.skoff, N=self.N, NQ=self.NQ,
                     nqmax=self.nq, win=self.win)

    def _exact_index(self, pos):
        """element offsets (B, len(pos)) of the K rows of positions `pos` for each head -> (B, P, 4)."""
        pg = self.eoff[:, pos // self.ebs]                                            # (B, P)
        return pg[..., None] + torch.arange(4, device=self.dev) * self.e_hs + (pos % self.ebs)[None, :, None] * self.e_ts

    def append(self, k, v):
        """k: pre-RoPE keys, v: values, (B, T, 4, 128)."""
        T = k.shape[1]; ef = flat(self.epool)
        for s in range(0, T, 128):
            kk, vv = k[:, s:s + 128].to(torch.bfloat16), v[:, s:s + 128].to(torch.bfloat16); t = kk.shape[1]
            pos = torch.arange(self.n, self.n + t, device=self.dev)
            idx = self._exact_index(pos)[..., None] + torch.arange(128, device=self.dev)   # (B, t, 4, 128)
            ef[idx] = kk; ef[idx + self.e_v] = vv
            if self.win and self.n < NSINK:
                m = pos < NSINK
                self.sk[:, pos[m]], self.sv[:, pos[m]] = kk[:, m], vv[:, m]
            self.n += t
            while not self.win and (self.n - LAG) // BLK > self.nq:
                j = self.nq
                pos = torch.arange(j * BLK, (j + 1) * BLK, device=self.dev)
                idx = self._exact_index(pos)[..., None] + torch.arange(128, device=self.dev)
                sb = quantize_blocks(ef[idx], ef[idx + self.e_v], self.H)                    # (B, 51200)
                qf = flat(self.qpool)
                qf[self.boff[:, j][:, None] + torch.arange(SUB, device=self.dev)] = sb
                self.nq += 1
        self.N.fill_(self.n); self.NQ.fill_(self.nq)
