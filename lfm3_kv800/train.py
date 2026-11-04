"""Training-side hook: put the 800h KV cache in the loop of an SFT / RL forward of LiquidAI LFM3-1.7B.

    from lfm3_kv800.train import patch_for_training
    hook = patch_for_training(model)            # HF Lfm3ForCausalLM (trust_remote_code), PeftModel, DDP, ... all fine
    loss = model(input_ids=x, attention_mask=m, labels=y).loss    # attention of layers 2/6/10/14/18/22 = 800h cache
    hook.set_mode("stock")                      # back to the unmodified attention (same weights)
    with hook.prefill(prompt_width):            # RL: columns < prompt_width were prefilled by the sampler
        logps = ...
    hook.remove()                               # restores the model exactly

Pure PyTorch, any device (CUDA, XPU, CPU). Nothing is cached between forwards except small masks.

What the 800h forward computes (the cache the lfm3-kv800 vLLM plugin samples with):
  * full-attention layers 6, 10, 14, 22: keys/values older than the exact window are replaced by their 800h
    quantize-dequantize: K 2-bit asymmetric per channel over aligned 256-token groups PRE-RoPE (RoPE applied after
    dequantization), V 1-bit per token over 256-dim groups after a per-head 128x128 Hadamard rotation, fp8 e4m3
    lo / scale per group. Same math as kv800.quantize_blocks -> dequantize_blocks, on the bf16 values the cache
    stores. Gradients pass straight through the rounding (STE). Each query sees, in one SDPA over [exact K/V | 800h
    K/V] with a [T, 2T] boolean mask, the exact or the quantized copy of each older key, never both.
  * full-attention layers 2, 18: 4 sink tokens + the last 256 tokens (including the query itself).
  * every other layer: unchanged.

Which keys are exact (rule="cache", the default, = what the vLLM cache does at decode): for the query at real index i
of its sequence (cache length n = i + 1), the 256-token group g is quantized once n >= 256 g + 384, i.e. keys at real
index < 256 * ((n - 128) // 256) are seen quantized and everything after is exact (between 128 and 383 exact tokens).
A group is therefore only ever read quantized once it is complete; the trailing incomplete group of a sequence is never
quantized, so any T works (K is zero-padded to a multiple of 256 inside the quantizer, the padding is never visible).
rule="last128" reproduces lfm3kv/qat_probe.py (exactly the newest 128 tokens exact, everything older quantized; tokens
of a group that is not complete in the sequence stay exact). It is the reference of the RESULTS.md recovery probe,
not what the sampler does.

Prefill (RL): vLLM computes the prompt in a prefill pass where the prompt's own keys are exact (quantized history only
exists from previous chunks). `hook.prefill(b)` (b = int or (B,) tensor of column indices) marks the query columns < b
as prefill queries: they see all keys exact on layers 6/10/14/22 (windows on 2/18 still apply, as in vLLM). Completion
tokens are scored as decode steps. This matches the sampler exactly when a prompt is prefilled in one chunk or is
shorter than 384 tokens; with chunked prefill of a long prompt (a TRL colocate engine uses max_num_batched_tokens=4096
shared by the batch) the prompt positions of later chunks differ slightly from the sampler; keep importance-sampling
correction on.

Batches: the caller's 2D attention_mask is respected (left and/or right padding): groups, the 128-token lag, windows and
sinks are counted on each row's real tokens (cumulative sum of the mask), padding keys are never visible, and pad
query rows only see themselves (no all-masked rows, so no NaNs). Packed documents (position_ids restarting at 0, no
attention_mask, the LFM3 packing convention) are split into documents the same way. 4D / dict masks and incremental
decoding with past_key_values are not supported in 800h mode (generate with vLLM, or set_mode("stock")).

The patch also rebuilds the rotary inverse frequencies in fp32 if they were loaded in a lower precision (vLLM computes
them in fp32); remove() puts the original module back."""
import contextlib, sys, weakref
from collections import OrderedDict

import torch
import torch.nn.functional as F

QUANT_LAYERS, WINDOW_LAYERS = (6, 10, 14, 22), (2, 18)
BLK, LAG, NSINK, WIN = 256, 128, 4, 256              # quant group, exact lag, sinks, window (= vLLM plugin constants)
F8 = torch.float8_e4m3fn
MODES, RULES = ("800h", "stock"), ("cache", "last128")

__all__ = ["patch_for_training", "KV800TrainHook", "fake_quant_k", "fake_quant_v", "hadamard",
           "apply_transformers_compat", "QUANT_LAYERS", "WINDOW_LAYERS"]


# ---------------------------------------------------------------- 800h quantize-dequantize (kv800.quantize_blocks math)
def hadamard(n=128, device=None):
    """Orthonormal Sylvester Hadamard (n x n, fp32), identical to kv800.hadamard."""
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat((torch.cat((H, H), 1), torch.cat((H, -H), 1)), 0)
    return (H / n ** 0.5).to(device)


_H = {}


def _had(device):
    key = str(device)
    if key not in _H:
        _H[key] = hadamard(128, device)
    return _H[key]


def fake_quant_k(x):
    """x (..., G, 256, C) fp32 pre-RoPE keys, groups of 256 tokens on dim -2 -> dequantized 800h K (fp32), per channel:
    2-bit codes, lo and scale = (hi - lo) / 3 rounded to fp8 e4m3."""
    lo, hi = x.amin(-2, keepdim=True), x.amax(-2, keepdim=True)
    lo8 = lo.to(F8).float()
    sc8 = ((hi - lo) / 3).clamp_min(1e-8).to(F8).float()
    c = ((x - lo8) / sc8.clamp_min(1e-8)).round().clamp(0, 3)
    return c * sc8 + lo8


def fake_quant_v(v, H):
    """v (..., 4, 128) fp32 values -> dequantized 800h V (fp32): per-head Hadamard, 1 bit per dim over 2 groups of
    256 dims (2 heads each) per token, fp8 lo and scale = hi - lo, rotated back."""
    y = (v @ H).reshape(*v.shape[:-2], 2, 256)
    lo, hi = y.amin(-1, keepdim=True), y.amax(-1, keepdim=True)
    lo8 = lo.to(F8).float()
    sc8 = (hi - lo).clamp_min(1e-8).to(F8).float()
    b = ((y - lo8) / sc8.clamp_min(1e-8)).round().clamp(0, 1)
    return (b * sc8 + lo8).reshape(v.shape) @ H


# ---------------------------------------------------------------- per-forward layout: real indices, masks, K compaction
_FAST_MASKS: "OrderedDict" = OrderedDict()           # (T, rule, kind, device) -> mask; small LRU, RL has many lengths


def _lru(key, build, cap=8):
    if key in _FAST_MASKS:
        _FAST_MASKS.move_to_end(key)
        return _FAST_MASKS[key]
    m = _FAST_MASKS[key] = build()
    while len(_FAST_MASKS) > cap:
        _FAST_MASKS.popitem(last=False)
    return m


class _Layout:
    """Everything one forward's patched layers share: built once from the caller's mask / positions / prefill
    boundary and attached to the forward's rotary cos tensor, so gradient-checkpoint recomputation sees the same."""

    def __init__(self, B, T, device, rule, attention_mask=None, position_ids=None, prefill=None):
        self.B, self.T, self.dev, self.rule = B, T, device, rule
        self.masks = {}
        valid = doc = None
        if attention_mask is not None:
            am = attention_mask.to(device)
            if am.shape != (B, T):
                raise ValueError(f"800h: attention_mask must be 2D (batch, seq) = {(B, T)}, got {tuple(am.shape)}")
            if not bool(am.bool().all()):
                valid = am.bool()
        if position_ids is not None and T > 1:
            pos = position_ids.to(device).expand(B, T)
            restart = torch.ones(B, T, dtype=torch.bool, device=device)
            restart[:, 1:] = pos[:, 1:] != pos[:, :-1] + 1
            if bool(restart[:, 1:].any()):
                doc = restart.long().cumsum(1) - 1
        pf = None
        if prefill is not None:
            b = torch.as_tensor(prefill, device=device).long().reshape(-1)
            if b.numel() not in (1, B):
                raise ValueError(f"800h: prefill boundary must be an int or a ({B},) tensor, got {b.numel()} values")
            pf = torch.arange(T, device=device)[None] < b[:, None].expand(B, 1)
            if not bool(pf.any()):
                pf = None
        self.fast = valid is None and doc is None and pf is None
        if self.fast:                                  # one sequence layout for the whole batch, masks cached by T
            self.Tc = -(-T // BLK) * BLK
            return
        valid = valid if valid is not None else torch.ones(B, T, dtype=torch.bool, device=device)
        doc = doc if doc is not None else torch.zeros(B, T, dtype=torch.long, device=device)
        vl = valid.long()
        prior = vl.cumsum(1) - vl                      # valid tokens before each column
        start = torch.ones(B, T, dtype=torch.bool, device=device)
        start[:, 1:] = doc[:, 1:] != doc[:, :-1]
        base = torch.where(start, prior, torch.zeros_like(prior)).cummax(1).values
        r = prior - base                               # real index of each token in its document
        L = torch.zeros(B, T, dtype=torch.long, device=device).scatter_add_(1, doc, vl)   # real tokens per document
        padded = (L + BLK - 1) // BLK * BLK
        dbase = padded.cumsum(1) - padded
        comp = dbase.gather(1, doc) + r                # compact slot: documents laid out 256-aligned
        self.Tc = max(BLK, int(padded.sum(1).max()))
        self.comp_scatter = torch.where(valid, comp, torch.full_like(comp, self.Tc))      # pads -> dump slot
        self.comp_gather = torch.where(valid, comp, torch.zeros_like(comp)).clamp_max(self.Tc - 1)
        self.valid, self.doc, self.r, self.pf = valid, doc, r, pf
        self.complete = L.gather(1, doc) >= (r // BLK + 1) * BLK

    # ---- K fake quant with real-index group alignment
    def fq_k(self, kp):
        """kp (B, T, 4, 128) pre-RoPE keys -> dequantized 800h keys (same dtype), groups of 256 real tokens."""
        B, T = kp.shape[:2]
        x = kp.detach().to(torch.bfloat16).float().reshape(B, T, -1)
        C = x.shape[-1]
        if self.fast:
            if self.Tc != T:
                x = F.pad(x, (0, 0, 0, self.Tc - T))
            y = fake_quant_k(x.view(B, self.Tc // BLK, BLK, C)).view(B, self.Tc, C)[:, :T]
        else:
            xc = x.new_zeros(B, self.Tc + 1, C)
            xc.scatter_(1, self.comp_scatter[..., None].expand(B, T, C), x)
            y = fake_quant_k(xc[:, :self.Tc].reshape(B, self.Tc // BLK, BLK, C)).reshape(B, self.Tc, C)
            y = y.gather(1, self.comp_gather[..., None].expand(B, T, C))
        return y.to(torch.bfloat16).to(kp.dtype).reshape(kp.shape)

    # ---- masks
    def mask(self, kind):
        if kind not in self.masks:
            if self.fast:
                key = (self.T, self.rule, kind, str(self.dev))
                self.masks[kind] = _lru(key, lambda: self._fast_mask(kind))
            else:
                self.masks[kind] = self._mask(kind)
        return self.masks[kind]

    def _fast_mask(self, kind):
        T, dev = self.T, self.dev
        i = torch.arange(T, device=dev)[:, None]
        j = torch.arange(T, device=dev)[None]
        d = i - j
        vis = d >= 0
        if kind == "win":
            return (vis & ((d < WIN) | (j < NSINK)))[None, None]
        if self.rule == "cache":
            q = vis & (j < (i + 1 - LAG).clamp_min(0) // BLK * BLK)
        else:
            q = vis & (d >= LAG) & ((j // BLK + 1) * BLK <= T)
        return torch.cat((vis & ~q, q), 1)[None, None]

    def _mask(self, kind):
        T, dev = self.T, self.dev
        tri = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
        vis = tri[None] & self.valid[:, None, :] & (self.doc[:, :, None] == self.doc[:, None, :])
        rq, rk = self.r[:, :, None], self.r[:, None, :]                              # comparisons only: no (B,T,T) ints
        self_only = torch.eye(T, dtype=torch.bool, device=dev)[None] & ~self.valid[:, :, None]   # pad query rows
        if kind == "win":
            return ((vis & ((rk > rq - WIN) | (rk < NSINK))) | self_only)[:, None]
        if self.rule == "cache":
            bound = (rq + 1 - LAG).clamp_min(0) // BLK * BLK
            if self.pf is not None:
                bound = torch.where(self.pf[:, :, None], torch.zeros_like(bound), bound)
            q = vis & (rk < bound)
        else:
            q = vis & (rk <= rq - LAG) & self.complete[:, None, :]
            if self.pf is not None:
                q = q & ~self.pf[:, :, None]
        return torch.cat(((vis & ~q) | self_only, q), 2)[:, None]


# ---------------------------------------------------------------- the hook
def _find_backbone(model):
    """The Lfm3Model inside `model` (CausalLM, PeftModel, DDP, ...): the module holding `layers` + `rotary_emb`."""
    for mod in model.modules():
        cfg = getattr(mod, "config", None)
        if hasattr(mod, "layers") and hasattr(mod, "rotary_emb") and hasattr(cfg, "layer_types"):
            return mod
    raise ValueError("no LFM3 backbone (module with .layers, .rotary_emb and config.layer_types) found in the model")


def _ste(x, fq):
    """Forward value exactly fq(x); gradient straight through (identity)."""
    with torch.no_grad():
        y = fq(x)
    return y + (x - x.detach())


class KV800TrainHook:
    """Handle returned by patch_for_training. mode: "800h" | "stock"; rule: "cache" | "last128" (see module doc)."""

    def __init__(self, model, mode="800h", rule="cache", fp32_rope=True):
        if rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
        lm = _find_backbone(model)
        prev_hook = getattr(lm, "_kv800_hook", None)
        if prev_hook is not None and prev_hook() is not None:
            raise RuntimeError("this model is already patched; call .remove() on the existing hook first")
        lt = lm.config.layer_types
        for l in QUANT_LAYERS + WINDOW_LAYERS:
            if l >= len(lt) or lt[l] != "full_attention":
                raise ValueError(f"layer {l} is not a full_attention layer ({lt[l] if l < len(lt) else 'missing'}): "
                                 "not the LFM3-1.7B layout the 800h cache is defined for")
        self.rule, self._lm, self._prefill, self._state = rule, lm, None, None
        self.mode = None
        self.set_mode(mode)
        self._rope_orig = None
        rot = lm.rotary_emb
        if fp32_rope and any(n.endswith("inv_freq") and b.dtype != torch.float32 for n, b in rot.named_buffers()):
            dev = next(b.device for n, b in rot.named_buffers() if n.endswith("inv_freq"))
            self._rope_orig = rot
            lm.rotary_emb = type(rot)(lm.config, device=dev)
        self._patched = []
        for l in QUANT_LAYERS + WINDOW_LAYERS:
            a = lm.layers[l].self_attn
            prev = a.__dict__.get("forward", None)
            orig = a.forward                                       # effective forward (bound, or a prior wrapper)
            a.forward = self._make_forward(a, l, orig)
            self._patched.append((a, prev))
        self._pre = lm.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        lm._kv800_hook = weakref.ref(self)

    # ---- public API
    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        return self

    @contextlib.contextmanager
    def using(self, mode):
        """Temporarily switch mode: `with hook.using("stock"): ...`."""
        old = self.mode
        self.set_mode(mode)
        try:
            yield self
        finally:
            self.mode = old

    @contextlib.contextmanager
    def prefill(self, boundary):
        """Query columns < boundary (int, or (B,) tensor per row) were computed by the sampler's prefill: they see their
        keys exact on the quantized layers. Completion columns are scored as decode steps."""
        old = self._prefill
        self._prefill = boundary
        try:
            yield self
        finally:
            self._prefill = old

    def set_prefill(self, boundary):
        """Non-context-manager form of prefill(); None clears it."""
        self._prefill = boundary
        return self

    def remove(self):
        """Undo the patch: original attention forwards, rotary module and no pre-hook."""
        if self._lm is None:
            return
        for a, prev in self._patched:
            if prev is None:
                a.__dict__.pop("forward", None)
            else:
                a.forward = prev
        self._patched = []
        self._pre.remove()
        if self._rope_orig is not None:
            self._lm.rotary_emb = self._rope_orig
            self._rope_orig = None
        self._lm._kv800_hook = None
        self._lm = None

    # ---- internals
    def _pre_hook(self, module, args, kwargs):
        names = ("input_ids", "attention_mask", "position_ids", "past_key_values")
        got = dict(zip(names, args))
        got.update({k: kwargs[k] for k in names if k in kwargs})
        if self.mode == "800h":
            am = got.get("attention_mask")
            if got.get("past_key_values") is not None:
                raise RuntimeError("800h training mode runs full-sequence forwards only (no past_key_values / generate); "
                                   "sample with the vLLM plugin, or use hook.set_mode('stock') to generate in HF")
            if am is not None and not (isinstance(am, torch.Tensor) and am.ndim == 2):
                raise RuntimeError("800h training mode needs a 2D (batch, seq) attention_mask or None")
        self._state = (got.get("attention_mask"), got.get("position_ids"), self._prefill)
        return None

    def _layout(self, hidden_states, cos):
        lay = getattr(cos, "_kv800_layout", None)
        if lay is not None and lay[0] is self:
            return lay[1]
        B, T = hidden_states.shape[:2]
        am, pos, pf = self._state if self._state is not None else (None, None, self._prefill)
        if am is not None and tuple(am.shape) != (B, T):
            am = None                                              # stale state (layers called directly): no padding
        if pos is not None and pos.shape[-1] != T:
            pos = None
        out = _Layout(B, T, hidden_states.device, self.rule, am, pos, pf)
        try:
            cos._kv800_layout = (self, out)
        except Exception:
            pass
        return out

    def _make_forward(self, a, l, orig):
        M = sys.modules[type(a).__module__]
        rope = M.apply_rotary_pos_emb
        hook = self

        def forward(hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kw):
            if hook.mode == "stock":
                return orig(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask,
                            past_key_values=past_key_values, **kw)
            B, T, _ = hidden_states.shape
            hsh = (B, T, -1, a.head_dim)
            cos, sin = position_embeddings
            lay = hook._layout(hidden_states, cos)
            q = a.q_norm(a.q_proj(hidden_states).view(hsh)).transpose(1, 2)
            kp = a.k_norm(a.k_proj(hidden_states).view(hsh))                       # (B, T, 4, 128) pre-RoPE
            v = a.v_proj(hidden_states).view(hsh)
            qr, k = rope(q, kp.transpose(1, 2), cos, sin)
            v4 = v.transpose(1, 2)
            if l in QUANT_LAYERS:
                H = _had(hidden_states.device)
                kq = _ste(kp, lay.fq_k)
                vq = _ste(v, lambda x: fake_quant_v(x.to(torch.bfloat16).float(), H).to(torch.bfloat16).to(x.dtype))
                _, kqr = rope(q, kq.transpose(1, 2), cos, sin)
                K, V = torch.cat((k, kqr), 2), torch.cat((v4, vq.transpose(1, 2)), 2)
                m = lay.mask("quant")
            else:
                K, V = k, v4
                m = lay.mask("win")
            g = a.num_key_value_groups
            K, V = K.repeat_interleave(g, 1), V.repeat_interleave(g, 1)
            o = F.scaled_dot_product_attention(qr, K.to(qr.dtype), V.to(qr.dtype), attn_mask=m, scale=a.scaling)
            gate = torch.sigmoid(a.gate_proj(hidden_states)).unsqueeze(-1)
            return a.o_proj((o.transpose(1, 2) * gate).reshape(B, T, -1)), None

        return forward


def patch_for_training(model, mode="800h", rule="cache", fp32_rope=True):
    """Patch an LFM3-1.7B HF model (or any wrapper of it: PEFT, DDP, ...) so its forward sees the 800h KV cache.
    Returns a KV800TrainHook: .set_mode("800h" | "stock"), .using(mode), .prefill(boundary), .remove().
    Patch before or after PEFT wrapping; LoRA on q/k/v/o/gate of the patched layers is picked up either way."""
    return KV800TrainHook(model, mode=mode, rule=rule, fp32_rope=fp32_rope)


def apply_transformers_compat():
    """Config-parsing shims some transformers 5.x releases need for the LFM3 remote code (layer type "conv" and the
    per-layer-type rope_parameters dict). No-op where the private symbols do not exist. Call before from_pretrained."""
    try:
        import transformers.configuration_utils as cu
        from transformers.modeling_rope_utils import RotaryEmbeddingConfigMixin
    except ImportError:
        return
    remap = getattr(cu, "_LEGACY_LAYER_TYPE_REMAP", None)
    if isinstance(remap, dict):
        remap.pop("conv", None)
    allowed = getattr(cu, "ALLOWED_ATTN_LAYER_TYPES", None)
    if isinstance(allowed, tuple) and "conv" not in allowed:
        cu.ALLOWED_ATTN_LAYER_TYPES = allowed + ("conv",)
    if getattr(RotaryEmbeddingConfigMixin, "_kv800_compat", False) or \
            not hasattr(RotaryEmbeddingConfigMixin, "convert_rope_params_to_dict"):
        return
    orig = RotaryEmbeddingConfigMixin.convert_rope_params_to_dict

    def convert(self, **kw):
        rp = getattr(self, "rope_parameters", None)
        if isinstance(rp, dict) and any(isinstance(v, dict) for v in rp.values()):
            self._rope_type_labels = getattr(self, "_rope_type_labels", getattr(self, "_attention_layer_types", None))
            self.standardize_rope_params()
            return kw
        return orig(self, **kw)

    RotaryEmbeddingConfigMixin.convert_rope_params_to_dict = convert
    RotaryEmbeddingConfigMixin._kv800_compat = True
