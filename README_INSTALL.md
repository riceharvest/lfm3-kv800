# lfm3-kv800: install and run

## Install

Intel B70 (Xe2, bmg-g31) in the vLLM 0.32 B70 image. The SYCL decode op ships prebuilt inside the package, so no compiler is needed:

```bash
docker run -it --device /dev/dri -v /dev/dri:/dev/dri:ro --shm-size 8g \
  -v /path/to/lfm3-kv800:/src:ro -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash ghcr.io/riceharvest/b70-vllm:v0.32.0-b70.7
cp -r /src /tmp/lfm3-kv800 && pip install --no-deps --no-build-isolation /tmp/lfm3-kv800
```

Or build an image that already has it installed (run from the package root):

```bash
docker build -f docker/Dockerfile -t lfm3-kv800:b70 .
```

On any other environment (CUDA, CPU, another torch):

```bash
pip install /path/to/lfm3-kv800            # add --no-deps to keep your torch / vllm
```

## Serve with vLLM

```bash
LFM3_KV800=1 vllm serve LiquidAI/LFM3-1.7B-Base --trust-remote-code --dtype bfloat16 \
  --enforce-eager --no-enable-prefix-caching -cc '{"mode":0}' \
  --max-model-len 34816 --max-num-seqs 64 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.75
```

When `LFM3_KV800=1` is set, the plugin applies `--enforce-eager`, prefix caching off and compile mode 0 even if you leave those flags out. The same goes for `LLM(...)` in Python: installing the package registers the plugin (entry point `vllm.general_plugins` = `lfm3_kv800.plugin:register`), and the default multiprocess engine works.

Which LFM3 model runs:

| Setting | Model used |
|---|---|
| `LFM3_KV800=1` | The vendored `lfm3_kv800.vllm_model.Lfm3ForCausalLM` with the 800h cache, even if your vLLM has its own LFM3 (upstream PR #64959). |
| `LFM3_KV800` unset, vLLM has no `Lfm3ForCausalLM` (true for 0.32) | The vendored model, with stock bf16 attention. |
| `LFM3_KV800` unset, vLLM has its own `Lfm3ForCausalLM` | vLLM's model. |

## Decode backends

`lfm3_kv800.kv800.decode` picks a backend per device and logs the choice once:

| Device | Order tried |
|---|---|
| xpu | `sycl` (prebuilt op) → `triton` → `torch` |
| cuda | `triton` → `torch` |
| cpu and others | `torch` |

The prebuilt op only loads on the exact torch it was built for (`lfm3_kv800/_lib/BUILD_INFO.json`, torch 2.14.0+xpu). With another torch, the op is JIT-built if `icpx` (oneAPI 2026.x) is on `PATH`; otherwise decode uses Triton.

Environment variables:

| Variable | Effect |
|---|---|
| `LFM3KV_BACKEND=sycl\|triton\|torch` | Forces one backend. Errors if that backend is unavailable. |
| `LFM3KV_XMX=0` | Uses the SIMT SYCL kernel instead of XMX. |
| `LFM3KV_BUILD_DIR` | Sets the JIT build cache directory. |

## Test

```bash
python tests/test_parity.py sycl triton torch      # strict: every named backend must run
python tests/test_parity.py --device cpu torch
pytest tests/test_parity.py                        # skips backends this machine lacks
```

## Rebuild the prebuilt op

Run this after changing `lfm3_kv800/csrc/kv800_sycl.sycl` or moving to another image or torch. It needs oneAPI 2026.x on the host, mounted read-only into the build container; no GPU is needed:

```bash
ONEAPI=/opt/intel/oneapi CVER=2026.0 scripts/build_sycl.sh
```
