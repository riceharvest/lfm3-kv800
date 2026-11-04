# lfm3-kv800

A 15x smaller KV cache for [LiquidAI/LFM3-1.7B-Base](https://huggingface.co/LiquidAI/LFM3-1.7B-Base) with no training: about 800 bytes/token instead of 12,288. It comes with decode kernels, a vLLM plugin and a training hook. On one 32 GB Intel Arc Pro B70, 800h fits 398 concurrent 32K-token sequences where stock vLLM fits 27, and decodes 11.9x stock's total throughput (5,124 vs 431 tok/s).

Full results, method and limitations are on the model card: **https://huggingface.co/darioooooo0o/LFM3-1.7B-Base-KV800** (it also hosts the optional recovery LoRA).

| at 32K, one B70 | stock vLLM | 800h |
|---|---|---|
| sequences that fit | 27 | 398 (measured, all resident) |
| decode throughput | 431 tok/s (c=27) | 861 tok/s at c=27, 5,124 tok/s at c=398 |
| needles 1K-32K (96) | 96/96 | 88/96 |
| rollouts (32) | 32/32 | 28/32 |
| cross-entropy cost | - | about +0.3 nats/token without training |

## What's here

| path | what |
|---|---|
| `lfm3_kv800/kv800.py` | paged 800h cache layout, quantizer, Triton kernels, backend chain (sycl → triton → torch) |
| `lfm3_kv800/csrc/kv800_sycl.sycl`, `_lib/` | SYCL decode kernels (XMX/DPAS + SIMT) and the prebuilt op for torch 2.14+xpu (bmg-g31) |
| `lfm3_kv800/torch_decode.py` | pure-torch decode, any device |
| `lfm3_kv800/vllm_attn.py`, `vllm_model.py`, `plugin.py` | vLLM 0.32 integration (`vllm.general_plugins` entry point) |
| `lfm3_kv800/train.py` | `patch_for_training(model)`: the same cache inside your SFT/RL forward pass |
| `examples/` | SFT and TRL GRPO examples (GRPO untested) |
| `eval/` | concurrency, needles, rollouts, throughput sweep; results in `eval/RESULTS_RELEASE.md` |
| `tests/` | kernel parity (12 cases, every backend) |
| `nvidia/` | one-click NVIDIA (Kaggle/Colab) check |
| `docker/`, `scripts/` | B70 image Dockerfile, SYCL op build script |

## Quick start

```bash
pip install --no-deps --no-build-isolation .           # in the vLLM 0.32 B70 image; elsewhere: pip install --no-deps .
LFM3_KV800=1 vllm serve LiquidAI/LFM3-1.7B-Base --trust-remote-code --dtype bfloat16 \
  --max-model-len 34816 --max-num-seqs 256 --max-num-batched-tokens 2048
```

```python
from lfm3_kv800.train import patch_for_training
h = patch_for_training(model, mode="800h")   # train with the cache vLLM samples with
```

More details are in `README_INSTALL.md`.

## License

The code is Apache-2.0 (see `LICENSE`). LFM3 weights and anything derived from them, including the LoRA on the model card, fall under the LFM Open License v1.0.

## Credits

Made by [@imdariotoo](https://x.com/imdariotoo). I post my local-model releases on X, so follow there for the next ones.
