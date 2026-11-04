# SPDX-License-Identifier: Apache-2.0
"""LFM3 (``Lfm3ForCausalLM``) for vLLM.

Ported from vLLM's ``lfm2.py`` and the LiquidAI/LFM3 remote code (``modeling_lfm3.py``).
Differences from LFM2:

* three layer types: ``conv`` (gated short conv, width 7), ``full_attention`` and
  ``sliding_attention`` (window 256);
* per-layer-type RoPE in fp32; the full layers use a zero-tail partial rotation
  (32 of 64 frequencies nonzero over head_dim 128), the sliding layers full RoPE;
* QK RMSNorm per head and a per-head sigmoid output gate;
* sandwich norms around the operator and the feed-forward block.

``Lfm3Attention`` is self-contained (projections, norms, RoPE, gate, the vLLM
``Attention`` layer and its KV cache) so that the full-attention layers can be
swapped for another implementation without touching the rest of the model. With ``LFM3_KV800=1``
the six full-attention layers and the native sliding layers run on the 800h cache (``lfm3_kv800.vllm_attn``).
Registered by ``lfm3_kv800.plugin.register`` (the ``vllm.general_plugins`` entry point).
"""

import os
from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.short_conv import ShortConv
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

ATTENTION_LAYER_TYPES = ("full_attention", "sliding_attention")

_DEFAULT_ROPE = {
    "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0, "partial_rotary_factor": 0.5},
    "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
}


def lfm3_layer_type(config, layer_idx: int) -> str:
    """``full_attention``, ``sliding_attention`` or ``conv``.

    Anything that is not an attention type is a short-conv layer (some transformers
    versions rename ``conv`` to ``linear_attention`` when parsing the config).
    """
    layer_type = config.layer_types[layer_idx]
    return layer_type if layer_type in ATTENTION_LAYER_TYPES else "conv"


def lfm3_head_dim(config) -> int:
    return getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads


def lfm3_rope_parameters(config, layer_type: str) -> dict:
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    params = rope_parameters.get(layer_type)
    if not isinstance(params, dict):
        params = _DEFAULT_ROPE[layer_type]
    rope_type = params.get("rope_type", "default")
    if rope_type != "default":
        raise NotImplementedError(f"LFM3 port supports rope_type 'default' only, got {rope_type!r} for {layer_type}")
    return params


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Lfm3RotaryEmbedding(nn.Module):
    """Neox-style RoPE over the whole head dim, computed like the HF reference.

    Partial rotation is LFM3's zero tail: the first ``int(factor * head_dim) // 2``
    frequencies follow ``1 / theta**(2i / head_dim)`` (progression over the full head
    dim) and the rest are 0, so those dims are left unrotated. ``rotate_half`` pairs
    dim d with d + head_dim / 2. This is not vLLM's ``rotary_dim`` partial RoPE.

    Numerics follow ``modeling_lfm3.py``: angles in fp32, cos/sin rounded to the
    activation dtype, then q/k rotated in fp32 and cast back. cos/sin are computed
    from ``positions`` on the fly (no per-layer max-position cache).
    """

    def __init__(self, head_dim: int, rope_theta: float, partial_rotary_factor: float, cos_sin_dtype: torch.dtype):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.cos_sin_dtype = cos_sin_dtype
        rope_angles = int(partial_rotary_factor * head_dim) // 2
        # Same expression as the reference, evaluated on CPU in fp32.
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32, device="cpu") / head_dim))
        inv_freq = torch.cat((inv_freq, torch.zeros(head_dim // 2 - rope_angles, dtype=torch.float32, device="cpu")))
        # Allocated on the model's device (vLLM builds the model under a device context); exact copy of the CPU values.
        buf = torch.empty(head_dim // 2, dtype=torch.float32)
        buf.copy_(inv_freq)
        self.register_buffer("inv_freq", buf, persistent=False)

    def cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.flatten().to(torch.float32)[:, None] * self.inv_freq.to(torch.float32)[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)
        cos = emb.cos().to(self.cos_sin_dtype).to(torch.float32)
        sin = emb.sin().to(self.cos_sin_dtype).to(torch.float32)
        return cos, sin

    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """q: (T, Hq, D), k: (T, Hkv, D) -> rotated, same dtypes and shapes."""
        cos, sin = self.cos_sin(positions)
        cos, sin = cos[:, None, :], sin[:, None, :]
        qf, kf = q.to(torch.float32), k.to(torch.float32)
        q_out = qf * cos + _rotate_half(qf) * sin
        k_out = kf * cos + _rotate_half(kf) * sin
        return q_out.to(q.dtype), k_out.to(k.dtype)

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, rope_theta={self.rope_theta}, "
            f"partial_rotary_factor={self.partial_rotary_factor}, zero_tail=True"
        )


class Lfm3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.w13 = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w13",
        )
        self.w2 = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w2",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.w13(x)
        x, _ = self.w2(self.act_fn(gate_up))
        return x


class Lfm3Attention(nn.Module):
    """LFM3 attention operator (full or sliding window), self-contained.

    hidden (T, hidden) -> qkv_proj -> per-head q_norm / k_norm -> fp32 RoPE (zero-tail
    partial on full layers) -> vLLM ``Attention`` (KV cache, sliding window on sliding
    layers) -> per-head ``sigmoid(gate_proj(hidden))`` gate -> o_proj.

    Replace this class (or ``self.attn``) to change how the full-attention layers store
    and read their KV cache. The q/k/v handed to ``self.attn`` are post-norm, post-RoPE,
    flattened to (T, heads * head_dim); v is not normed.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        layer_type: str,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        cos_sin_dtype: torch.dtype = torch.bfloat16,
        prefix: str = "",
    ) -> None:
        super().__init__()
        assert layer_type in ATTENTION_LAYER_TYPES, layer_type
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = lfm3_head_dim(config)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        # One gate logit per query head; sharded with the heads under TP.
        self.gate_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.total_num_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate_proj",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)

        rope = lfm3_rope_parameters(config, layer_type)
        self.rotary_emb = Lfm3RotaryEmbedding(
            head_dim=self.head_dim,
            rope_theta=float(rope["rope_theta"]),
            partial_rotary_factor=float(rope.get("partial_rotary_factor", 1.0)),
            cos_sin_dtype=cos_sin_dtype,
        )

        # HF: key visible iff kv_idx > q_idx - sliding_window (W keys including the query).
        # vLLM: per_layer_sliding_window=W -> FlashAttention window (W - 1, 0), the same W keys.
        self.sliding_window = config.sliding_window if layer_type == "sliding_attention" else None
        # LFM3_KV800=1: full-attention layers keep their KV in the 800h cache (lfm3_kv800/vllm_attn.py) instead of a
        # bf16 vLLM Attention cache: 2-bit K / 1-bit V on 6, 10, 14, 22; 4 sinks + window 256 on 2, 18.
        self.kv800 = None
        if os.environ.get("LFM3_KV800", "0") == "1":
            # native sliding layers ride the same exact-window kernel (K post-RoPE, identity RoPE, no sinks): the image's
            # paged-decode kernels are not built for this model's sliding configuration
            from vllm.config import get_current_vllm_config
            from lfm3_kv800.vllm_attn import KV800Attention
            max_pos = get_current_vllm_config().model_config.max_model_len + 16
            self.kv800 = KV800Attention(layer_idx, prefix, max_pos, slide=layer_type == "sliding_attention")
            return
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=self.sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        n_tokens = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.contiguous().view(n_tokens, self.num_heads, self.head_dim))
        k = self.k_norm(k.contiguous().view(n_tokens, self.num_kv_heads, self.head_dim))
        k_pre = k
        q, k = self.rotary_emb(positions, q, k)
        if self.kv800 is not None:
            kin = k if self.kv800.slide else k_pre
            attn_output = self.kv800(q, kin, v.view(n_tokens, self.num_kv_heads, self.head_dim))
            gate, _ = self.gate_proj(hidden_states)
            attn_output = attn_output * torch.sigmoid(gate).unsqueeze(-1)
            output, _ = self.o_proj(attn_output.reshape(n_tokens, self.q_size))
            return output
        attn_output = self.attn(
            q.reshape(n_tokens, self.q_size),
            k.reshape(n_tokens, self.kv_size),
            v.contiguous(),
        )
        gate, _ = self.gate_proj(hidden_states)
        attn_output = attn_output.view(n_tokens, self.num_heads, self.head_dim) * torch.sigmoid(gate).unsqueeze(-1)
        output, _ = self.o_proj(attn_output.view(n_tokens, self.q_size))
        return output


class Lfm3ShortConv(ShortConv):
    """vLLM ``ShortConv`` (same weights, state layout and metadata) with a torch conv.

    vLLM's Triton ``causal_conv1d_fn`` / ``causal_conv1d_update`` unroll the kernel
    width only up to 5 / 6; LFM3 uses width 7. This class keeps everything else of
    ``ShortConv`` and replaces the two conv calls:

    * prefill: every request's tokens are laid out behind its (L-1)-entry conv state
      (zeros without an initial state) and one depthwise ``F.conv1d`` runs over the
      packed buffer; the last L-1 inputs of each request become its new state;
    * decode: ``conv1d(cat(state, x))`` per request, state shifted by one.

    Only ``mamba_cache_mode == "none"`` (no prefix caching) and no speculative decode.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_spec:
            raise NotImplementedError("Lfm3ShortConv does not support speculative decoding")
        mode = getattr(self.cache_config, "mamba_cache_mode", "none") if self.cache_config is not None else "none"
        if mode != "none":
            raise NotImplementedError(f"Lfm3ShortConv supports mamba_cache_mode 'none' only, got {mode!r}")

    def forward_native(self, hidden_states: torch.Tensor, output: torch.Tensor):
        return self.forward_cuda(hidden_states, output)

    def forward_cuda(self, hidden_states: torch.Tensor, output: torch.Tensor):
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        attn_metadata = None
        if attn_metadata_raw is not None:
            assert isinstance(attn_metadata_raw, dict)
            attn_metadata = attn_metadata_raw.get(self.prefix)

        BCx, _ = self.in_proj(hidden_states)
        B, C, x = BCx.chunk(3, dim=-1)

        if attn_metadata is None:
            # Profile / dummy run: shapes only.
            output[: hidden_states.shape[0]], _ = self.out_proj(C * (B * x))
            return

        assert getattr(attn_metadata, "num_accepted_tokens", None) is None, "spec decode not supported"
        conv_state = self.kv_cache[0] if is_conv_state_dim_first() else self.kv_cache[0].transpose(-1, -2)
        # conv_state: (num_slots, dim, L - 1) view into the cache
        weight = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2))  # (dim, L)

        num_prefills = attn_metadata.num_prefills
        num_decodes = attn_metadata.num_decode_tokens  # one token per decode request
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_actual_tokens = num_decodes + num_prefill_tokens

        Bx = B[:num_actual_tokens] * x[:num_actual_tokens]
        Bx_d, Bx_p = torch.split(Bx, [num_decodes, num_prefill_tokens], dim=0)  # decodes come first
        outs = []
        if num_decodes > 0:
            idx_d = attn_metadata.state_indices_tensor_d
            if idx_d.dim() > 1:
                idx_d = idx_d[:, 0]
            init_d = getattr(attn_metadata, "has_initial_states_d", None)
            outs.append(self._conv_decode(Bx_d, conv_state, weight, idx_d[:num_decodes],
                                          None if init_d is None else init_d[:num_decodes]))
        if num_prefills > 0:
            outs.append(
                self._conv_prefill(
                    Bx_p,
                    conv_state,
                    weight,
                    attn_metadata.query_start_loc_p,
                    attn_metadata.state_indices_tensor_p,
                    attn_metadata.has_initial_states_p,
                    num_prefills,
                )
            )
        conv_out = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
        output[:num_actual_tokens], _ = self.out_proj(C[:num_actual_tokens] * conv_out)

    def _conv_prefill(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_state: torch.Tensor | None,
        num_prefills: int,
    ) -> torch.Tensor:
        """x: (T, dim) prefill tokens of ``num_prefills`` requests -> (T, dim)."""
        T, dim = x.shape
        S = weight.shape[1] - 1
        R = num_prefills
        device = x.device
        qsl = query_start_loc[: R + 1].to(torch.long)
        slots = state_indices[:R].to(torch.long)
        ar_r = torch.arange(R, device=device)
        ar_s = torch.arange(S, device=device)
        tok = torch.arange(T, device=device)
        req = torch.searchsorted(qsl[1:].contiguous(), tok, right=True)  # request of each token

        # Request r occupies ext rows [qsl[r] + S*r, qsl[r+1] + S*(r+1)): S state rows, then its tokens.
        ext = x.new_zeros(T + S * R, dim)
        ext[tok + S * (req + 1)] = x
        init = conv_state[slots].transpose(1, 2).to(x.dtype)  # (R, S, dim)
        # A fresh request's state slot holds whatever its recycled block held before (KV pages of other groups share
        # the memory; 800h quant pages hold arbitrary bytes, i.e. bf16 NaN/Inf). Select, never multiply by 0.
        if has_initial_state is not None:
            init = torch.where(has_initial_state[:R].to(torch.bool)[:, None, None], init, torch.zeros_like(init))
        else:
            init = torch.zeros_like(init)
        state_rows = (qsl[:R] + S * ar_r)[:, None] + ar_s[None, :]  # (R, S)
        ext[state_rows.reshape(-1)] = init.reshape(R * S, dim)

        # y[j] = bias + sum_k w[k] * ext[j + k]; token t of request r reads y[t + S*r]. Shifted multiply-adds instead
        # of F.conv1d: the XPU depthwise conv1d costs ~0.6 ms of host time per call.
        M = ext.shape[0] - S
        wt = weight.t().to(ext.dtype)                                            # (L, dim)
        y = ext[0:M] * wt[0]
        for kk in range(1, S + 1):
            y = y + ext[kk:kk + M] * wt[kk]
        if self.conv.bias is not None:
            y = y + self.conv.bias.to(y.dtype)
        out = y[tok + S * req]

        # New state: the last S rows of each request block (includes old state rows if the chunk is short).
        last_rows = (qsl[1 : R + 1] + S * (ar_r + 1))[:, None] - S + ar_s[None, :]  # (R, S)
        conv_state[slots] = ext[last_rows.reshape(-1)].view(R, S, dim).transpose(1, 2).to(conv_state.dtype)
        return out.to(x.dtype)

    def _conv_decode(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: (N, dim), one new token per request -> (N, dim). Rows without an initial state (a 1-token prompt) start
        from zeros, not from the recycled slot's bytes."""
        dim = x.shape[1]
        idx = state_indices.to(torch.long)
        valid = idx >= 0  # PAD_SLOT_ID rows (graph padding) neither read nor write a real slot
        safe_idx = torch.where(valid, idx, torch.zeros_like(idx))
        state = conv_state[safe_idx]  # (N, dim, S)
        if has_initial_state is not None:
            state = torch.where(has_initial_state.to(torch.bool)[:, None, None], state, torch.zeros_like(state))
        window = torch.cat((state, x.to(state.dtype).unsqueeze(-1)), dim=-1)  # (N, dim, L)
        out = (window * weight.to(window.dtype)).sum(-1)                         # depthwise conv, one output position
        if self.conv.bias is not None:
            out = out + self.conv.bias.to(out.dtype)
        conv_state[safe_idx] = torch.where(valid[:, None, None], window[..., 1:], state)
        return out.to(x.dtype)


class _Lfm3DecoderLayerBase(nn.Module):
    """Residual stream shared by both layer kinds (vLLM fused add + norm form).

    HF: h = h + post_operator_norm(op(operator_norm(h)))
        h = h + post_ffn_norm(ffn(ffn_norm(h)))
    Here ``residual`` carries h and the returned tensor is the not-yet-added branch.
    """

    def _init_common(self, config, quant_config, prefix: str) -> None:
        self.feed_forward = Lfm3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.feed_forward",
        )
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.use_sandwich_norm = bool(getattr(config, "use_sandwich_norm", True))
        if self.use_sandwich_norm:
            self.post_operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.post_ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def operator(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.operator_norm(hidden_states)
        else:
            hidden_states, residual = self.operator_norm(hidden_states, residual)
        hidden_states = self.operator(positions, hidden_states)
        if self.use_sandwich_norm:
            hidden_states = self.post_operator_norm(hidden_states)
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        if self.use_sandwich_norm:
            hidden_states = self.post_ffn_norm(hidden_states)
        return hidden_states, residual


class Lfm3AttentionDecoderLayer(_Lfm3DecoderLayerBase):
    def __init__(
        self,
        config,
        layer_idx: int,
        layer_type: str,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        cos_sin_dtype = model_config.dtype if model_config is not None else torch.get_default_dtype()
        self.self_attn = Lfm3Attention(
            config=config,
            layer_idx=layer_idx,
            layer_type=layer_type,
            cache_config=cache_config,
            quant_config=quant_config,
            cos_sin_dtype=cos_sin_dtype,
            prefix=f"{prefix}.self_attn",
        )
        self._init_common(config, quant_config, prefix)

    def operator(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.self_attn(positions=positions, hidden_states=hidden_states)


class Lfm3ShortConvDecoderLayer(_Lfm3DecoderLayerBase):
    def __init__(
        self,
        config,
        layer_idx: int,
        layer_type: str = "conv",
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.short_conv = Lfm3ShortConv(
            config=config,
            dim=config.hidden_size,
            layer_idx=layer_idx,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.short_conv",
        )
        self._init_common(config, quant_config, prefix)

    def operator(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(hidden_states)
        self.short_conv(hidden_states, output)
        return output


@support_torch_compile
class Lfm3Model(nn.Module):
    # HF ``.conv.`` -> vLLM ``.short_conv.`` (as in lfm2.py), stacked q/k/v and w1/w3.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={".conv.": ".short_conv."},
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".w1": (".w13", 0),
            ".w3": (".w13", 1),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size, config.hidden_size, org_num_embeddings=config.vocab_size
        )

        def get_layer(prefix: str):
            layer_idx = extract_layer_index(prefix)
            layer_type = lfm3_layer_type(config, layer_idx)
            layer_class = Lfm3ShortConvDecoderLayer if layer_type == "conv" else Lfm3AttentionDecoderLayer
            return layer_class(
                config,
                layer_idx,
                layer_type,
                model_config,
                cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        if get_pp_group().is_last_rank:
            self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        else:
            self.embedding_norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions=positions, hidden_states=hidden_states, residual=residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        hidden_states, _ = self.embedding_norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Lfm3ForCausalLM(nn.Module, HasInnerState, SupportsPP, IsHybrid):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "w13": ["w1", "w3"],
        "in_proj": ["in_proj"],
    }
    hf_to_vllm_mapper = Lfm3Model.hf_to_vllm_mapper

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: "VllmConfig") -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: "VllmConfig") -> tuple[tuple[int, int]]:
        """Short-conv state per conv layer: (hidden_size / tp, conv_L_cache - 1) (orientation per vLLM layout)."""
        hf_config = vllm_config.model_config.hf_config
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
            intermediate_size=hf_config.hidden_size,
            conv_kernel=hf_config.conv_L_cache,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.short_conv_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.model = Lfm3Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if getattr(config, "tie_word_embeddings", True):
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        tied = getattr(self.config, "tie_word_embeddings", True)
        loader = AutoWeightsLoader(self, ignore_unexpected_prefixes=["lm_head."] if tied else None)
        return loader.load_weights(weights)
