"""Pure-torch 800h decode: same inputs (kv800.Paged), semantics and output as the Triton / SYCL kernels, on any device
(cpu, cuda, xpu). Written for correctness, not speed: one host sync and a python loop over the batch rows.

Per row b (n = N[b], nq = NQ[b]):
  quantized layer (win == 0): sub-blocks j < nq (byte offsets BOFF[b, j] into qpool) dequantized with
      kv800.dequantize_blocks; exact positions [256 nq, n) from the exact pages.
  window layer (win > 0): exact positions [max(256 nq, n - win), n), plus the sinks (positions < min(4, lo, n)) from
      SKOFF / SVOFF when Paged.sk is set.
K is pre-RoPE: RoPE on the dim pairs (d, d + 64), d < 32, from the (P, 32) cos / sin tables, in fp32, then rounded
to bf16 (as HF does for keys). One fp32 softmax over all positions, scale 1/sqrt(128), 4 query heads per kv head."""
import torch

from . import kv800


def _rope_bf16(K, pos, cos_t, sin_t):
    """K (S, 4, 128) float pre-RoPE at positions pos (S,) -> rotated, rounded to bf16, as fp32."""
    cs = cos_t[pos].float()[:, None, :]
    sn = sin_t[pos].float()[:, None, :]
    k0, k2 = K[..., :32], K[..., 64:96]
    Kr = K.clone()
    Kr[..., :32] = k0 * cs - k2 * sn
    Kr[..., 64:96] = k2 * cs + k0 * sn
    return Kr.to(torch.bfloat16).float()


def decode(P, q, cos_t, sin_t, H):
    """q: (B, 16, 128) bf16 post-RoPE -> (B, 16, 128) bf16."""
    B, dev = q.shape[0], q.device
    H = H.to(dev, torch.float32)
    win = int(P.win or 0)
    nqmax = 0 if win else int(P.nqmax or 0)
    sinks = bool(win) and P.sk is not None
    N = P.N.tolist()
    NQ = P.NQ.tolist()
    ef = kv800.flat(P.epool)
    qf = kv800.flat(P.qpool) if nqmax else None
    if sinks:
        skf, svf = kv800.flat(P.sk), kv800.flat(P.sv)
        skoff, svoff = P.skoff.tolist(), P.svoff.tolist()
    ar128 = torch.arange(128, device=dev)
    h4 = torch.arange(4, device=dev)
    sub = torch.arange(kv800.SUB, device=dev)
    out = torch.zeros(B, 16, 128, dtype=torch.bfloat16, device=dev)
    for b in range(B):
        n, nq_b = int(N[b]), int(NQ[b])
        nq = min(nq_b, nqmax)
        lo = nq_b * kv800.BLK
        if win:
            lo = max(lo, n - win)
        Ks, Vs, Ps = [], [], []
        if nq:
            idx = P.boff[b, :nq].to(dev, torch.long)[:, None] + sub
            k, v = kv800.dequantize_blocks(qf[idx], H)                       # (nq, 256, 4, 128) bf16 / fp32
            Ks.append(k.reshape(-1, 4, 128).float()); Vs.append(v.reshape(-1, 4, 128).float())
            Ps.append(torch.arange(nq * kv800.BLK, device=dev))
        if sinks:
            cnt = min(kv800.NSINK, lo, n)
            if cnt > 0:
                t = torch.arange(cnt, device=dev)
                rel = ((t[:, None] * 4 + h4) * 128)[..., None] + ar128          # (cnt, 4, 128)
                Ks.append(skf[int(skoff[b]) + rel].float()); Vs.append(svf[int(svoff[b]) + rel].float())
                Ps.append(t)
        if n > lo:
            pos = torch.arange(lo, n, device=dev)
            page = P.eoff[b].to(dev, torch.long)[pos // P.ebs]
            kidx = (page + (pos % P.ebs) * P.e_ts)[:, None] + h4 * P.e_hs       # (S, 4)
            kidx = kidx[..., None] + ar128                                    # (S, 4, 128)
            Ks.append(ef[kidx].float()); Vs.append(ef[kidx + P.e_v].float())
            Ps.append(pos)
        if not Ks:
            continue
        pos = torch.cat(Ps)
        K = _rope_bf16(torch.cat(Ks), pos, cos_t, sin_t)                     # (S, 4, 128)
        V = torch.cat(Vs)
        qb = q[b].float().view(4, 4, 128)                                    # (kv head, q row, dim)
        sc = torch.einsum("hrd,shd->hrs", qb, K) * 128 ** -0.5
        o = torch.einsum("hrs,shd->hrd", sc.softmax(-1), V)
        out[b] = o.reshape(16, 128).to(torch.bfloat16)
    return out
