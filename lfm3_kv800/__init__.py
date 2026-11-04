"""800h KV cache for LiquidAI LFM3 (2-bit K / 1-bit V paged cache, decode kernels, vLLM plugin).

Kept free of torch / triton / vllm imports: the vLLM plugin entry point imports this package in every process.
Modules: kv800 (cache layout, quantizer, decode dispatch), torch_decode (pure-torch decode), kv800_sycl (prebuilt
SYCL op for Intel Xe2), vllm_attn / vllm_model (vLLM 0.32 integration), plugin (``vllm.general_plugins`` entry point).
"""
import logging

__version__ = "0.1.0"


def get_logger() -> logging.Logger:
    """Package logger that prints INFO lines in vLLM worker processes too (vLLM only configures its own logger)."""
    lg = logging.getLogger("lfm3_kv800")
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(levelname)s %(asctime)s [lfm3_kv800] %(message)s", "%m-%d %H:%M:%S"))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
        lg.propagate = False
    return lg
