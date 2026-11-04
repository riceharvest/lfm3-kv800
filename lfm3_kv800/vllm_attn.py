"""vLLM (0.32) integration of the 800h KV cache for LFM3's full-attention layers.

Per quantized layer (6, 10, 14, 22) two vLLM-allocated caches:
  <attn prefix>.kv800_quant  FullAttentionSpec, block 1280 tokens: 5 x 51,200 B sub-blocks (kv800.py layout) padded
                             to 262,144 B pages = 204.8 B/token; the 6,144 B padding of a request's first page holds the 4
                             sink K or V rows of a window layer.
  <attn prefix>.kv800_exact  SlidingWindowSpec (window 640): exact bf16 pre-RoPE K | unrotated V of the newest tokens.
Per window layer (2, 18) only the exact window cache; its 4 sink tokens live in the aux of layers (6, 10) / (14, 22).
= 4 x 204.8 B = 819 B/token + per-request windows (exact 640 x 6 layers) and conv state.
Decode rows go through kv800.decode (SYCL / Triton / torch backend); prefill chunks are computed in torch
(dequantized history + exact window + the chunk, masked SDPA). Needs enforce_eager, prefix caching off, compile off
(the plugin sets these when LFM3_KV800=1).
Env: LFM3KV_FLASH_PREFILL=0 uses the fp32 reference prefill; LFM3KV_PREFILL_CHECK=0 skips the one-time check of the
bf16 SDPA prefill against it."""
from dataclasses import dataclass, field
from typing import ClassVar

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (AttentionBackend, AttentionCGSupport, AttentionMetadataBuilder,
                                       CommonAttentionMetadata, MultipleOf)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec, SlidingWindowSpec

import os

from . import get_logger, kv800

_log = get_logger()
QUANT_LAYERS, WINDOW_LAYERS = (6, 10, 14, 22), (2, 18)
SINK_HOST = {2: (6, 10), 18: (14, 22)}          # window layer -> (layer whose aux holds sink K, ... sink V)
EXACT_WINDOW = 640                               # >= 383 exact tokens + one exact block of slack
QBLOCK, QCONTENT, QPAGE = 1280, 25, 262144         # 4 heads x 1280 tokens x (25 + 25) B = 256,000 B of sub-blocks


class KV800Backend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "KV800"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("KV800 caches are driven by KV800Attention, not an AttentionImpl")

    @staticmethod
    def get_builder_cls():
        return KV800Builder

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        if kv_cache_spec is not None and getattr(kv_cache_spec, "dtype", None) == torch.uint8:
            return [MultipleOf(256)]
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [QCONTENT, 128]


@dataclass
class KV800Meta:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    positions: torch.Tensor
    num_reqs: int
    num_actual_tokens: int
    num_decodes: int
    cache: dict = field(default_factory=dict)        # per-step values shared by all layers of the group


class KV800Builder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER
    reorder_batch_threshold: int | None = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool = False):
        c = common_attn_metadata
        nd, _, _, _ = split_decodes_and_prefills(c, decode_threshold=1)
        R = c.num_reqs
        qsl = c.query_start_loc_cpu[:R + 1]
        sl_cpu = c.seq_lens_cpu_upper_bound[:R] if c.seq_lens_cpu_upper_bound is not None else c.seq_lens[:R].cpu()
        T = c.num_actual_tokens
        if c.positions is not None:
            pos = c.positions[:T]
        else:                                    # token i of request r sits at seq_len_r - qlen_r + (i - qsl_r)
            dev = c.seq_lens.device; Tq = int(qsl[R])
            qlen = (qsl[1:] - qsl[:-1]).to(dev)
            start = c.seq_lens[:R] - qlen
            pos = torch.zeros(T, dtype=torch.long, device=dev)
            pos[:Tq] = torch.repeat_interleave(start - qsl[:-1].to(dev), qlen, output_size=Tq) + torch.arange(Tq, device=dev)
        return KV800Meta(c.block_table_tensor, c.slot_mapping, c.seq_lens[:R], sl_cpu, qsl, pos.long(), R, T, nd)


class _CacheLayer(nn.Module, AttentionLayerBase):
    def __init__(self, name: str, spec_fn):
        super().__init__()
        self.layer_name, self._spec_fn = name, spec_fn
        self.kv_cache = torch.tensor([])
        ctx = get_current_vllm_config().compilation_config.static_forward_context
        if name in ctx:
            raise ValueError(f"Duplicate layer name: {name}")
        ctx[name] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return self._spec_fn(vllm_config)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return KV800Backend

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self.kv_cache = kv_cache

    def forward(self):
        ...


def _quant_spec(cfg: VllmConfig) -> KVCacheSpec:
    # 1280 tokens = 5 sub-blocks = 256,000 B, padded to 262,144 = 2 x the 64-token bf16 page (131,072 B) of the
    # sliding/exact groups, so vLLM's page unification only doubles their block size (no padding of the quant pages).
    # The 6,144 padding bytes are the aux area (sinks of the window layers live in a request's first page).
    return FullAttentionSpec(block_size=QBLOCK, num_kv_heads=4, head_size=QCONTENT, head_size_v=QCONTENT, dtype=torch.uint8,
                             page_size_padded=QPAGE)


def _exact_spec(cfg: VllmConfig) -> KVCacheSpec:
    return SlidingWindowSpec(block_size=cfg.cache_config.block_size, num_kv_heads=4, head_size=128, head_size_v=128,
                             dtype=torch.bfloat16, sliding_window=EXACT_WINDOW)


def _slide_spec(cfg: VllmConfig) -> KVCacheSpec:
    return SlidingWindowSpec(block_size=cfg.cache_config.block_size, num_kv_heads=4, head_size=128, head_size_v=128,
                             dtype=torch.bfloat16, sliding_window=kv800.WIN)


_TABLES: dict = {}


def _tables(device, max_pos):
    """(cos32, sin32) [P, 32] for the kernels, (cos, sin) [P, 128] for torch RoPE, Hadamard: as HF Lfm3 full layers
    (theta 1e6, 32 rotating frequencies over head_dim 128, fp32 product, bf16 cast)."""
    key = (str(device), max_pos)
    if key not in _TABLES:
        inv = 1.0 / (1e6 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=device) / 128))
        fr = torch.arange(max_pos, device=device, dtype=torch.float32)[:, None] * inv[None]
        c32, s32 = fr.cos().to(torch.bfloat16).contiguous(), fr.sin().to(torch.bfloat16).contiguous()
        one, zero = torch.ones_like(c32), torch.zeros_like(s32)
        c64, s64 = torch.cat((c32, one), 1), torch.cat((s32, zero), 1)
        _TABLES[key] = (c32, s32, torch.cat((c64, c64), 1), torch.cat((s64, s64), 1), kv800.hadamard(128, device),
                        torch.ones_like(c32), torch.zeros_like(s32))
    return _TABLES[key]


def _rope(k, cos, sin):
    """HF apply_rotary_pos_emb for keys (T, 4, 128) bf16 with cos/sin (T, 128) bf16 -> bf16."""
    kf = k.float(); c, s = cos.float()[:, None], sin.float()[:, None]
    return (kf * c + torch.cat((-kf[..., 64:], kf[..., :64]), -1) * s).to(torch.bfloat16)



_FLASH = os.environ.get("LFM3KV_FLASH_PREFILL", "1") == "1"
_CHECK = {"on": os.environ.get("LFM3KV_PREFILL_CHECK", "1") == "1", "done": set()}


def _prefill_torch(q, K, V, kp, n0, n, slide, win):
    """Reference prefill: q (Tq, 16, 128) at positions n0..n-1, K/V (S, 4, 128) bf16 at positions kp."""
    Kf, Vf = K.float().repeat_interleave(4, 1).transpose(0, 1), V.float().repeat_interleave(4, 1).transpose(0, 1)
    qi = q.float().transpose(0, 1)
    qp = torch.arange(n0, n, device=q.device)
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for a in range(0, n - n0, 512):
        qa = qp[a:a + 512, None]
        m = kp[None] <= qa
        if slide:
            m = m & (kp[None] > qa - kv800.WIN)
        elif win:
            m = m & ((kp[None] > qa - kv800.WIN) | (kp[None] < kv800.NSINK))
        sc = (qi[:, a:a + 512] @ Kf.transpose(1, 2)) * (128 ** -0.5)
        out[a:a + 512] = (sc.masked_fill(~m[None], float("-inf")).softmax(-1) @ Vf).transpose(0, 1)
    return out


def _prefill_flash(q, K, V, kp, n0, n, slide, win):
    """Fast prefill: torch SDPA in bf16 (oneDNN fused attention on XPU) over 2048-query chunks with one bool mask
    shared by all heads (bottom-right causal, plus window / window+sinks). Same semantics as _prefill_torch."""
    import torch.nn.functional as F
    dt = torch.bfloat16                                 # fp32 here was 2.6x slower prefill and did not change recall
    CH = 2048
    Kh = K.to(dt).repeat_interleave(4, 1).transpose(0, 1)[None]                                                # (1,16,S,128)
    Vh = V.to(dt).repeat_interleave(4, 1).transpose(0, 1)[None]
    qi = q.to(dt).transpose(0, 1)[None]                                                                         # (1,16,Tq,128)
    qp = torch.arange(n0, n, device=q.device)
    out = torch.empty_like(q)
    for a in range(0, n - n0, CH):
        qa = qp[a:a + CH, None]
        m = kp[None] <= qa
        if slide:
            m = m & (kp[None] > qa - kv800.WIN)
        elif win:
            m = m & ((kp[None] > qa - kv800.WIN) | (kp[None] < kv800.NSINK))
        o = F.scaled_dot_product_attention(qi[:, :, a:a + CH], Kh, Vh, attn_mask=m[None, None], scale=128 ** -0.5)
        out[a:a + CH] = o[0].transpose(0, 1).to(out.dtype)
    return out


class KV800Attention(nn.Module):
    """Attention core of one LFM3 full-attention layer with the 800h cache. forward(q, k, v): q (T, 16, 128) post-RoPE,
    k (T, 4, 128) PRE-RoPE (k_norm output), v (T, 4, 128) -> (T, 16, 128)."""

    def __init__(self, layer_idx: int, prefix: str, max_pos: int, slide: bool = False):
        """slide=True: an LFM3 native sliding layer (window 256, no sinks, K stored POST-RoPE, kernel RoPE = identity)."""
        super().__init__()
        self.l, self.max_pos, self.slide = layer_idx, max_pos, slide
        self.win = slide or layer_idx in WINDOW_LAYERS
        self.root = prefix.split(".layers.")[0]
        self.x_name = f"{prefix}.kv800_exact"
        self.xc = _CacheLayer(self.x_name, _slide_spec if slide else _exact_spec)
        if not self.win:
            self.q_name = f"{prefix}.kv800_quant"
            self.qc = _CacheLayer(self.q_name, _quant_spec)
        self._ctx = get_current_vllm_config().compilation_config.static_forward_context
        self._attn_prefix = prefix

    def _host(self, layer):
        name = self._attn_prefix.replace(f".layers.{self.l}.", f".layers.{layer}.") + ".kv800_quant"
        return name, self._ctx[name].kv_cache

    # ---- helpers on vLLM views
    @staticmethod
    def _exact_rows(ev, block_table_rows, pos):
        """element offsets of the K rows (…, 4, 128) of positions `pos` ((…,) long) given per-position page ids."""
        s0, s1, s2 = ev.stride(0), ev.stride(1), ev.stride(2); ebs = ev.shape[2]
        base = block_table_rows.long() * s0 + (pos % ebs) * s2
        return base[..., None, None] + (torch.arange(4, device=pos.device) * s1)[:, None] + torch.arange(128, device=pos.device)

    def forward(self, q, k, v):
        fc = get_forward_context()
        amd = fc.attn_metadata
        out = torch.zeros_like(q)
        if amd is None:
            return out
        if isinstance(amd, list):
            amd = amd[0]
        mx: KV800Meta = amd[self.x_name]
        T, R, nd = mx.num_actual_tokens, mx.num_reqs, mx.num_decodes
        dev = q.device
        c32, s32, c128, s128, H, one32, zero32 = _tables(dev, self.max_pos)
        if self.slide:
            c32, s32 = one32, zero32
        ev = self.xc.kv_cache; ef = kv800.flat(ev); ebs = ev.shape[2]
        s0, s1, s2 = ev.stride(0), ev.stride(1), ev.stride(2)
        C = mx.cache
        # ---- 1. exact K/V of every scheduled token into the window pages (index computed once per step)
        key = ("wr", s0, s1, s2, ebs)
        if key not in C:                                  # one host sync per step and group (nonzero), shared by layers
            slot = mx.slot_mapping[:T].long(); ok = slot >= 0
            okidx = ok.nonzero().squeeze(1)
            idx = self._exact_rows(ev, slot // ebs, slot % ebs).index_select(0, okidx)
            C[key] = (ok, okidx, idx, idx + 128)
        ok, okidx, idx_k, idx_v = C[key]
        ef[idx_k] = k[:T].index_select(0, okidx).to(torch.bfloat16); ef[idx_v] = v[:T].index_select(0, okidx).to(torch.bfloat16)
        if "rq" not in C:
            qsl_, sl_ = mx.query_start_loc_cpu, mx.seq_lens_cpu
            qlen_ = qsl_[1:] - qsl_[:-1]; Tq_ = int(qsl_[R])
            rot = torch.zeros(T, dtype=torch.long, device=dev)
            rot[:Tq_] = torch.repeat_interleave(torch.arange(R, device=dev), qlen_.to(dev), output_size=Tq_)
            C["rq"] = (qsl_, sl_, sl_ - qlen_, rot)
        qsl, sl, prev, req_of_tok = C["rq"]
        pos = mx.positions[:T]
        if self.slide:
            pass
        elif self.win:
            # ---- sinks: positions < 4 into the hosts' page-0 aux (bf16 views)
            if ("sinkmask",) not in C:                   # any token below position 4 this step: decided on CPU data
                anysm = bool((prev < kv800.NSINK).any())
                sm = (pos < kv800.NSINK) & ok if anysm else None
                C[("sinkmask",)] = (sm, anysm)
            sm, anysm = C[("sinkmask",)]
            if anysm:
                for hl, src in zip(SINK_HOST[self.l], (k, v)):
                    hname, hv = self._host(hl)
                    hb = hv.view(torch.bfloat16); hf = kv800.flat(hb)
                    hm: KV800Meta = amd[hname]
                    page0 = hm.block_table[req_of_tok[sm], 0].long()
                    aux = page0 * hb.stride(0) + (hv.shape[2] // 256) * (kv800.SUB // 2)
                    o = aux[:, None, None] + ((pos[sm][:, None] * 4 + torch.arange(4, device=dev)) * 128)[..., None] \
                        + torch.arange(128, device=dev)
                    hf[o] = src[:T][sm].to(torch.bfloat16)
        else:
            # ---- 2. flush sub-blocks that became eligible this step (n >= 256 j + 384); plan computed once per step
            mq: KV800Meta = amd[self.q_name]
            qv = self.qc.kv_cache; qf = kv800.flat(qv); bsq = qv.shape[2]; qs0 = qv.stride(0)
            fkey = ("flush", s0, s1, s2, ebs, qs0, bsq)
            if fkey not in mq.cache:
                nq_prev = torch.clamp(prev - kv800.LAG, min=0) // kv800.BLK
                nq_new = torch.clamp(sl - kv800.LAG, min=0) // kv800.BLK
                pairs = [(i, j) for i in range(R) for j in range(int(nq_prev[i]), int(nq_new[i]))]
                plan = []
                for c0 in range(0, len(pairs), 32):
                    ch = pairs[c0:c0 + 32]
                    ri = torch.tensor([p_[0] for p_ in ch], device=dev); ji = torch.tensor([p_[1] for p_ in ch], device=dev)
                    pp = ji[:, None] * kv800.BLK + torch.arange(kv800.BLK, device=dev)       # (M, 256)
                    kidx = self._exact_rows(ev, mx.block_table[ri[:, None], pp // ebs], pp)
                    tok0 = ji * kv800.BLK
                    boff = mq.block_table[ri, tok0 // bsq].long() * qs0 + ((tok0 % bsq) // kv800.BLK) * kv800.SUB
                    plan.append((kidx, kidx + 128, boff[:, None] + torch.arange(kv800.SUB, device=dev)))
                mq.cache[fkey] = plan
            for kidx, vidx, qidx in mq.cache[fkey]:
                qf[qidx] = kv800.quantize_blocks(ef[kidx], ef[vidx], H)
        # ---- 3. decode rows: kernels (tables computed once per step)
        if nd:
            dkey = ("dec", s0, nd)
            if dkey not in C:
                N_ = mx.seq_lens[:nd].to(torch.int32)
                C[dkey] = (N_, (mx.block_table[:nd].long() * s0).contiguous())
            N, eoff = C[dkey]
            if self.slide:
                NQ = torch.zeros_like(N); nqmax = 0; boff = eoff; qpool = ev
                skb = svb = skoff = svoff = None
            elif self.win:
                NQ = torch.zeros_like(N); nqmax = 0; boff = eoff; qpool = ev
                skn, skv = self._host(SINK_HOST[self.l][0]); svn, svv = self._host(SINK_HOST[self.l][1])
                skb, svb = skv.view(torch.bfloat16), svv.view(torch.bfloat16)
                skoff = amd[skn].block_table[:nd, 0].long() * skb.stride(0) + (skv.shape[2] // 256) * (kv800.SUB // 2)
                svoff = amd[svn].block_table[:nd, 0].long() * svb.stride(0) + (svv.shape[2] // 256) * (kv800.SUB // 2)
            else:
                qkey = ("dec", qs0, bsq, nd)
                if qkey not in mq.cache:
                    NQ_ = (torch.clamp(N - kv800.LAG, min=0) // kv800.BLK).to(torch.int32)
                    nqmax_ = int(torch.clamp(sl[:nd] - kv800.LAG, min=0).max()) // kv800.BLK
                    if nqmax_:
                        sub = torch.arange(nqmax_, device=dev) * kv800.BLK
                        boff_ = (mq.block_table[:nd, sub // bsq].long() * qs0 + ((sub % bsq) // kv800.BLK) * kv800.SUB).contiguous()
                    else:
                        boff_ = torch.zeros(nd, 1, dtype=torch.int64, device=dev)
                    mq.cache[qkey] = (NQ_, nqmax_, boff_)
                NQ, nqmax, boff = mq.cache[qkey]
                qpool = qv
                skb = svb = skoff = svoff = None
            P = kv800.Paged(qpool=qpool, boff=boff, epool=ev, eoff=eoff, ebs=ebs, e_hs=s1, e_ts=s2, e_v=128, sk=skb, sv=svb,
                            skoff=skoff, svoff=svoff, N=N, NQ=NQ, nqmax=nqmax, win=kv800.WIN if self.win else 0)
            out[:nd] = kv800.decode(P, q[:nd].contiguous(), c32, s32, H)
        # ---- 4. prefill / extend rows: torch
        for i in range(nd, R):
            s, e = int(qsl[i]), int(qsl[i + 1]); n, n0 = int(sl[i]), int(prev[i])
            parts_k, parts_v, parts_p = [], [], []
            if self.win:
                lo = max(0, n0 - (kv800.WIN - 1))
                ns = 0 if self.slide else min(kv800.NSINK, lo)
                if ns:
                    for hl, lst in zip(SINK_HOST[self.l], (parts_k, parts_v)):
                        hname, hv = self._host(hl); hb = hv.view(torch.bfloat16); hf = kv800.flat(hb)
                        aux = int(amd[hname].block_table[i, 0]) * hb.stride(0) + (hv.shape[2] // 256) * (kv800.SUB // 2)
                        lst.append(hf[aux:aux + ns * 512].view(ns, 4, 128).float())
                    parts_p.append(torch.arange(ns, device=dev))
            else:
                nqp = int(torch.clamp(prev[i] - kv800.LAG, min=0)) // kv800.BLK
                lo = nqp * kv800.BLK
                if nqp:
                    tok0 = torch.arange(nqp, device=dev) * kv800.BLK
                    boff = mq.block_table[i, tok0 // bsq].long() * qs0 + ((tok0 % bsq) // kv800.BLK) * kv800.SUB
                    kq, vq = kv800.dequantize_blocks(qf[boff[:, None] + torch.arange(kv800.SUB, device=dev)], H)
                    parts_k.append(kq.reshape(-1, 4, 128).float()); parts_v.append(vq.reshape(-1, 4, 128))
                    parts_p.append(torch.arange(lo, device=dev))
            if n0 > lo:
                pp = torch.arange(lo, n0, device=dev)
                kidx = self._exact_rows(ev, mx.block_table[i, pp // ebs], pp)
                parts_k.append(ef[kidx].float()); parts_v.append(ef[kidx + 128].float()); parts_p.append(pp)
            parts_k.append(k[s:e].float()); parts_v.append(v[s:e].float()); parts_p.append(torch.arange(n0, n, device=dev))
            kp_ = torch.cat(parts_p)
            Kb = torch.cat(parts_k).to(torch.bfloat16)
            Kb = Kb if self.slide else _rope(Kb, c128[kp_], s128[kp_])
            Vb = torch.cat(parts_v).to(torch.bfloat16)
            o = None
            if _FLASH:
                try:
                    o = _prefill_flash(q[s:e], Kb, Vb, kp_, n0, n, self.slide, self.win)
                except Exception as ex:                          # unsupported config: torch path
                    _log.warning("flash prefill failed (%s: %s), torch path", type(ex).__name__, str(ex)[:120])
                    o = None
            kind = "slide" if self.slide else ("window" if self.win else "quant")
            if o is None or (_CHECK["on"] and kind not in _CHECK["done"] and n0 > 300):
                ref = _prefill_torch(q[s:e], Kb, Vb, kp_, n0, n, self.slide, self.win)
                if o is not None:
                    _CHECK["done"].add(kind)
                    err = ((o.float() - ref.float()).norm() / ref.float().norm()).item()
                    _log.info("prefill flash vs torch (%s layer %d, %d q at %d, %d k): rel %.2e", kind, self.l, e - s, n0,
                              Kb.shape[0], err)
                    if not err < 2e-2:
                        _log.warning("flash prefill disagrees, using torch prefill")
                        globals()["_FLASH"] = False; o = ref
                else:
                    o = ref
            out[s:e] = o.to(out.dtype)
        return out
