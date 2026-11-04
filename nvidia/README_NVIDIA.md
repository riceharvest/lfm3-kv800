# lfm3-kv800 on NVIDIA: one-click check (free Kaggle / Colab GPU)

`check_nvidia.ipynb` and `check_nvidia.py` run the same code. They check the 800h KV cache package (`lfm3_kv800`, LiquidAI LFM3-1.7B) on an NVIDIA GPU:

1. They report the environment.
2. They run kernel parity against a torch reference.
3. They benchmark decode speed against torch SDPA.
4. Optionally, they run an end-to-end needle test with the HF model against stock.

Everything runs on free tiers. Nothing here has been run on NVIDIA hardware yet. See "What was verified offline" below.

## Run it

**Kaggle (recommended: 2x T4, 30 GPU-hours/week)**
1. New notebook > File > Import notebook > `check_nvidia.ipynb`.
2. Settings > Accelerator > **GPU T4 x2**. Do not pick P100; see the GPU notes below.
3. Settings > **Internet on**. This needs a phone-verified Kaggle account.
4. In the first code cell, set `PACKAGE`. Optionally set `RUN_E2E = True`.
5. Run all.

**Colab (free T4)**
1. Upload `check_nvidia.ipynb`.
2. Runtime > Change runtime type > **T4 GPU**.
3. Set `PACKAGE` and run all.

**Script** (any machine with a CUDA GPU and the package installed):
```bash
pip install --no-deps "<package spec>"          # keep the preinstalled CUDA torch / triton
python check_nvidia.py                          # env + parity + speed  (~5-15 min on T4, mostly Triton compiles)
pip install "transformers==5.20.2" && python check_nvidia.py --e2e   # + HF end to end (~3.4 GB download, +15-40 min)
```
Useful flags:
- `--backends triton,torch`
- `--n 32768`
- `--batches 64,16,1`
- `--lengths 1024,4096,16384`
- `--needles 8`
- `--e2e-dtype bf16|fp16`
- `--skip-speed`
- `--no-sweep`
- `--device cpu` runs the torch backend only, as a dry run.

Exit codes:
- `0`: everything passes as shipped.
- `1`: a parity failure.
- `2`: the package can't be imported.
- `3`: correct only with the in-script shim (see "Known risks").

**Re-running cells.** The environment cell resets the report, so a re-run does not duplicate rows. Two exceptions need a restart of the runtime followed by Run all:

- **Changing `RUN_E2E`.** A newly pip-installed transformers needs a fresh process. `load_model` detects an installed vs imported version mismatch and tells you to restart.
- **Getting a fresh "as shipped" Triton verdict after the fp8 shim has run.** The shim patches the imported package in place. A re-run in the same session tests `triton+f8shim` and repeats the earlier as-shipped verdict, with a NOTE.

### The install placeholder

Nothing is published yet. The first notebook cell has `PACKAGE = "REPLACE_ME"` and refuses to run until you replace it. It accepts any pip spec, for example:
- `git+https://github.com/<user>/<repo>.git#subdirectory=release/lfm3-kv800`
- a wheel URL on the HF Hub

It installs with `--no-deps` so pip can't swap out Colab's or Kaggle's CUDA torch. Set `NO_DEPS = False` if the import then fails on a missing dependency.

## What each cell does

| Cell | Does |
|---|---|
| 0. install | pip installs `PACKAGE`. With `RUN_E2E`, it also installs `transformers==5.20.2`. Then it imports `lfm3_kv800` and prints its version and path. |
| check code (9 cells) | Definitions only, identical to `check_nvidia.py`: the package adapter, the reference math, timing helpers, Triton probes, parity, speed, e2e and summary. |
| Settings | Edits `CFG`: speed context `n`, batch sizes, e2e lengths, needles per length, e2e dtype. |
| 1. Environment | Reports the GPU name, compute capability, VRAM and free memory, torch/CUDA/cuDNN/triton/transformers versions, the torch arch list (warns when this torch has no kernels for the GPU), native bf16 (sm >= 80), and whether torch can cast fp8. It then runs **four tiny Triton probes**: a bf16 load and round, an `fp8e4nv` bitcast (needs sm_89), the integer fp8 decoder used as the shim, and a launch with `warp_size=16`. |
| 2. Parity | Runs the 12 cases of `run/kv800_check.py` (quantized layer n = 50, 383, 384, 700, 1500, 5000; window layer n = 3, 10, 200, 259, 260, 1500; B=2; the last 5 tokens appended one at a time) for the **triton** and **torch** backends. The reference is a copy of `lfm3kv.qkv.reference` with `surgery_gpu._aq`, so nothing from `lfm3kv` is imported. A case passes at relative error < 1e-2, the same threshold as the original. On the n=5000 case, the torch profiler lists the CUDA kernels that actually ran. This catches a "triton" run that silently fell back to torch. Two more lines with an fp16 query are informational. Finally it reports what the automatic backend chain picks here. |
| 3. Speed | Times one decode call (CUDA events, 3 warmup + 20 timed) at B=64/16/1 with n=32K, for the quantized layer (127 flushed 256-token blocks + exact tail) and the window layer (4 sinks + 256). Cache contents are random but valid, filled as in `kv800_check.py`. It reports µs/call, GB/s of cache read and the % of the GPU's nominal bandwidth. It also runs a `num_warps` 1/2/4 sweep of the Triton quantized kernel at B=16 if the package keeps the module global `QW`. The baseline is torch SDPA decode (1 query, 16 q heads over 4 kv heads, `enable_gqa`) on a 16-bit cache of the same length, in **bf16 and fp16**. |
| 4. End to end | Optional. Loads `LiquidAI/LFM3-1.7B-Base` at pinned revision `7bca5038` with `trust_remote_code`. It applies the guarded transformers config patches from `lfm3kv/model.py` and rebuilds RoPE in fp32. It checks the config was parsed right: layer types, 32 of 64 nonzero full-attention RoPE frequencies, 64 of 64 sliding. Then it patches the 800h cache into the 6 full-attention layers, porting `KVLayers` from `lfm3kv/qkv_run.py`: layers 6/10/14/22 get the 2-bit K / 1-bit V blocks, layers 2/18 get 4 sinks + window 256. Prefill stays exact. Every decode step reads through `lfm3_kv800.kv800.decode`. **Lossless check first:** with a 196-token prompt and 4 decode steps, no block is quantized yet and the window covers everything. 800h must then match stock to within rounding; a wiring bug shows up as rel ~ 1. Then come **8 needles at each of 1K, 4K and 16K** (24 in total, depths spread evenly, fixed seed, haystack = this Python's argparse/difflib/inspect/typing source). Stock and 800h run on identical items. It reports recall per length, mean log-prob of the code, and harness prefill tok/s and decode ms/token. Those are batch-1 HF harness numbers, not kernel numbers. |
| Summary | One table, plus `nvidia_check_results.json` with every row, error and note. |

## GPU notes

| GPU (free where) | sm | What to expect |
|---|---|---|
| **T4** (Colab free, Kaggle 2x) | 7.5 | No native bf16: torch emulates bf16 math, and the kernels only load and store bf16 (cheap integer conversion), so they are unaffected. **Triton has no `float8e4nv` below sm_89**, so the shipped quantized-layer Triton kernel (`_qdec`) fails to compile. The window/exact kernel (`_xdec`) compiles. The check reports `triton: FAIL as shipped` and re-runs as `triton+f8shim` (below). SDPA: bf16 falls back to slower kernels, while fp16 has tensor cores; both are reported, labelled. The e2e model runs bf16 (emulated, slow). `--e2e-dtype fp16` is faster but LFM3 was not validated in fp16. |
| **P100** (Kaggle) | 6.0 | Triton does not support sm < 7.0, so only the torch backend is checked. Recent torch wheels may also lack sm_60 kernels. The environment cell warns about that. **Use T4 x2 instead.** |
| L4 (Colab paid, not free) | 8.9 | Native bf16 and fp8: the shipped kernels should compile as is. |

**Memory (16 GB T4):**
- The 800h cache at B=64 x 32K is about 0.42 GB of quantized blocks plus 84 MB of exact pages.
- The bf16/fp16 SDPA baseline at B=64 x 32K is about 4.3 GB of K+V.
- Both fit, so nothing is scaled down by default. Every case is still sized against `torch.cuda.mem_get_info()`. If something doesn't fit (another notebook holding memory, a bigger `--n`), n is halved until it does and a NOTE says so.
- The torch backend may OOM at B=64 x 32K, depending on how it dequantizes. That shows as `OOM` in its row and does not stop the run.
- The e2e test needs about 3.4 GB of weights plus the exact DynamicCache (about 0.7 GB at 16K) and a deepcopy for the answer/teacher-forcing split. It fits a T4.

**Free-tier time limits** (as of late 2026; check the current terms):
- Kaggle: 30 GPU-hours/week, at most 12 h per session.
- Colab free: sessions are capped at about 12 h, disconnect after about 90 min idle, and a GPU is not guaranteed.

The whole check with e2e is well under 1 h on a T4.

## Assumptions about the package API

The check is written against `lfm3_kv800.kv800`, mirroring `lfm3kv/kv800.py`. All of it goes through the `Pkg` class in `check_nvidia.py`. If a name changes, edit that one class.

- `PagedStore(batch, max_len, device, window=0, ...)` with `.H`, `.append(k, v)` (pre-RoPE K and V, `(B, T, 4, 128)`), `.paged()`, and the attributes `qpool` (viewable as `(-1, 51200)` uint8), `epool`, `sk`, `sv`, `N`, `NQ`, `n`, `nq`, `win`.
- `decode(P, q, cos_t, sin_t, H[, spb, target])`. `H` is passed only if the signature has it. `q` is `(B, 16, 128)` bf16 post-RoPE. `cos_t`/`sin_t` are `(max_pos, 32)` bf16.
- Backend selection, in this order:
  1. `set_backend(name)` on `kv800` or the package root;
  2. otherwise the module attribute `BACKEND`;
  3. also the env vars `LFM3KV_BACKEND` and `LFM3_KV800_BACKEND`.

  Names: `triton`, `torch`, `auto`. An optional `get_backend()`/`active_backend()` is reported.
- Constants `BLK, LAG, NSINK, WIN, SUB, *_O` are read from `kv800` and fall back to the reference values. A mismatch prints a NOTE.
- For the shims and the warps sweep: the Triton kernels are module-level `_qdec`/`_xdec` in some `lfm3_kv800.*` module, with an fp8 helper named `_f8` and the launch width in a module global `QW`. If these names differ, the shims and the sweep report "not applicable" and the plain result stands.
- There is no `clone()` on `PagedStore`. The e2e snapshot clones every tensor attribute except `H` itself.

## Known risks

- **T4: fp8 in the Triton quantized kernel.** This was confirmed offline. Triton 3.6 compiled `lfm3kv/kv800.py`'s `_qdec` for sm_75, and it fails with `type fp8e4nv not supported in this architecture`. For sm_89 it compiles. The check's `triton+f8shim` swaps the package's `_f8` for an exact integer e4m3 decoder (`_f8_sw` in `check_nvidia.py`) and resets Triton's kernel hashes. It runs in a fresh `TRITON_CACHE_DIR` per run, so a shimmed binary is never reused for an "as shipped" run. **Fix for the package:** use the integer decode below on sm < 89. It is bit-exact on all 254 non-NaN bytes, and costs a few integer ops per scale load, once per block.
  ```python
  @triton.jit
  def _f8(p):  # fp8 e4m3fn byte -> fp32, exact, no NaN
      b = tl.load(p).to(tl.int32); e = (b >> 3) & 15; m = b & 7; s = (b >> 7) & 1
      nrm = ((s << 31) | ((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
      sub = m.to(tl.float32) * 0.001953125
      return tl.where(e == 0, tl.where(s == 1, -sub, sub), nrm)
  ```
- **Triton version and `warp_size=16`.** Triton 3.6-3.8 have a `warp_size` field in `CUDAOptions`. They accept the XPU launch kwarg and ignore it, since CUDA warps are 32 lanes. An older Triton raises `Keyword argument warp_size was specified but unrecognised`. The check detects that, re-runs as `triton+kwshim`, and the package should drop the kwarg on CUDA. `num_warps=1` was tuned for one Intel 16-lane sub-group, so NVIDIA speed at the shipped launch is not a tuned number. The sweep shows how much `num_warps` 2/4 change it. Read the speed rows as "as shipped", not as the NVIDIA ceiling.
- **The kernels are bf16 by design.** Exact pages are bf16, `_xtile` rounds keys to bf16 like HF, and `combine()` returns bf16. There is no fp16 kernel path to test. The fp16-query parity lines only check that an fp16 `q` is accepted and upcast. On T4 the fp16 vs bf16 difference shows up in the SDPA baseline and the e2e model, not in the kernels.
- **transformers.** The LFM3 remote code needs transformers 5.x. Some 5.x releases (5.17 seen locally) rename the `conv` layer type and flatten nested `rope_parameters`. The check applies `lfm3kv/model.py`'s patches in guarded form, then refuses to continue unless the layer types and RoPE frequencies come out right. The notebook pins `transformers==5.20.2` for e2e: that is the version checked locally (CPU) with these patches.
- **The model's own Triton conv kernel.** LFM3's remote code runs a Triton depthwise-conv kernel on CUDA. If it fails on the GPU, the e2e cell switches the model to its torch conv path and prints a NOTE. This affects stock and 800h equally.
- **Backend fallback.** If the package's auto chain catches a Triton failure and quietly runs torch, the profiler check flags `parity passed but no _qdec/_xdec CUDA kernel ran`. If the profiler itself is unavailable, that field reads `None`.
- **The model may be gated** on the Hub. If loading fails with a 401 or 403, add an HF token: Kaggle Secrets or Colab `userdata`, then `huggingface_hub.login()`.

## What was verified offline

There is no NVIDIA GPU on the machine this was written on.

- **Syntax:** `py_compile` of the script, and `json.tool` of the notebook.
- **CPU dry run.** A stand-in `lfm3_kv800` package was used: `kv800.py`'s torch quantizer and `PagedStore`, plus a dequantize-and-attend torch decode.
  - All 12 parity cases pass against the copied reference, at rel ≈ 3-5e-3.
  - The speed and SDPA code paths run.
  - The shim retry flow works with a simulated sm_75 fp8 failure.
  - Every notebook cell executes in order.
- **End-to-end plumbing.** It ran on CPU with a tiny random-weight LFM3 built from the real config and remote code, under transformers 5.20.2: model load, patch checks, the `KVLayers` port, the lossless check (rel 1.9e-2, argmax agrees), needles, and the snapshot/restore.
- **Offline Triton compiles** (Triton 3.6 NVIDIA backend, no GPU), using the original kernels from `lfm3kv/kv800.py`:
  - `_qdec` fails for sm_75 and compiles for sm_89.
  - `_xdec` compiles for both.
  - `_qdec` with the `_f8` shim compiles for sm_75.
  - All four probes compile; the `fp8e4nv` one fails on sm_75, as expected.
- **Not verified:** anything that needs a real GPU (numerics on device, speed, the profiler kernel names, the real model's recall), and the real `lfm3_kv800` package, which did not exist yet when this was written.

The notebook is generated from `check_nvidia.py`. Its code cells are the `# %%` chunks with the `__main__` block removed, plus install, settings and step cells. If you change the script, change the matching notebook cell too.
