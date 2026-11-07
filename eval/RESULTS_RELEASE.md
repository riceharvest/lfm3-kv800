# LFM3-1.7B: 800h KV cache vs stock bf16 KV in vLLM on one Intel Arc B70 (32 GB)

**0.1.1 update.**

0.1.0 had a bug in the vLLM integration: past token 384, layer 10 quantized keys and values from the wrong cache pages, because a per-step flush plan was shared across KV groups (one-line fix in `lfm3_kv800/vllm_attn.py`). Every multi-token generation past that point was affected, in all decode backends. The training hook did not have the bug, so trainer and engine disagreed: on long rollouts from my own finetune, 13 % of tokens fell outside the [0.5, 5] importance ratio band, now 0.5 %.

Measured again after the fix: needles (chunk 2048) 88/96 to 91/96, needles (chunk 8192) 79/96 to 88/96, rollouts (chunk 2048) 28/32 to 30/32, rollouts (chunk 8192) 28/32 to 31/32. Sections 2 and 3 below show the 0.1.0 numbers. Not measured again: capacity, throughput, and the cross-entropy and recovery tables, which did not go through this code path.

Status: final, after the fix for the cross-request state bug (see "History" at the end).

- **Capacity (section 1):** comes from vLLM's KV cache config plus a scheduler-instrumented run. The bug does not
  affect it.
- **Needles and rollouts (sections 2 and 3):** measured after the fix through the shipped package. The package was
  pip-installed into the container and vLLM loaded it through its `vllm.general_plugins` entry point (`--plugin`). Every
  run checks, and records in its JSON, that the model class is `lfm3_kv800.vllm_model` from site-packages and that no
  in-tree module (`vllm_lfm3`, `lfm3kv`) was imported.
- **Stock needles:** the in-tree code ran them before the fix. Stock does not touch the 800h path.

Setup: image `ghcr.io/riceharvest/b70-vllm:v0.32.0-b70.7` (vLLM 0.32.0), in-process engine
(`VLLM_ENABLE_V1_MULTIPROCESSING=0`), `LFM3_KV800=1` (800h, SYCL backend) or `0` (stock). `PLUGIN=1 ./run_all.sh ...`
selects the package path; without it, runs use the in-tree `vllm_lfm3.register()` (capacity runs, stock needles). Every run uses `enforce_eager`, prefix caching off, and bf16 weights. Docker runs with `--memory=24g` and holds
the B70 lock for the whole container. Scripts: `concurrency.py`, `needles.py`, `rollouts.py`, `run_all.sh`. Raw per-item
JSON and logs are in `eval/out/`.

## 1. Concurrent 32K-token requests that fit (gpu_memory_utilization 0.85, prefill chunk 8192)

Both modes use the same engine arguments. A request is 32,768 prompt tokens + 64 new tokens = 32,832 tokens.

| | stock bf16 KV | 800h KV | ratio |
|---|---|---|---|
| KV pool vLLM allocated | 21.58 GiB (35,353 blocks x 640 KiB) | 21.58 GiB (22,098 blocks x 1 MiB) | same memory |
| vLLM's stated max concurrency (`get_max_concurrency_for_kv_cache_config`, the "Maximum concurrency" figure) | **27.4** | **39.5** | 1.44x |
| steady-state capacity: (num_blocks - one admission) / blocks held per decoding request, +1 | **33** (1,034 blocks = 678 MB per request) | **449** (48 blocks = 50 MB per request) | 13.6x |
| measured: N x 32K all resident at once, 0 preemptions | **27 / 27 PASS** (Triton attention), KV usage 79.5 % | **398 / 398 PASS** (0.9 x 443 steady), KV usage 90.9 % | >= 14.5x stock's 27.4 stated, >= 12x stock's 33 steady |
| measured at the stated N | - | 39 / 39 PASS, 0 preemptions, KV usage 10.0 % | |

How the two "max" numbers differ: vLLM's stated figure reserves, for every request, each sliding/exact-window group's
window **plus** `max_in_flight_tokens` (2 x 8192 = 16,384 tokens). That is the admission cap the scheduler applies
while a request is still being prefilled. Once a request is decoding it holds only its window blocks, and vLLM frees
the rest as the window slides. In the 800h layout those four window groups make up 530 of the 560 admission blocks, so
vLLM's stated number badly undercounts 800h. The scheduler only applies the admission cap to the request being
admitted, so the steady-state number is the one a server reaches. The measured run shows it.

The confirmation run (`concurrency.py --confirm steady --stagger`, its own engine with `max_num_seqs 768`, 21,817
blocks, steady 443) instruments the scheduler. It records every preemption and, per step, the number of running
requests whose full 32,768-token prompt is in the KV cache. vLLM decodes a request as soon as its prefill ends, so a
request generating only 64 tokens would finish long before the last of 398 is prefilled. Each request therefore
generates its 64 tokens plus enough extra tokens to outlast the prefills queued behind it: 182 to 1,851 new tokens,
which keeps every context at 32,768 tokens or more.

Result: all 398 requests were resident at the same time with their full 32K prompt, for 183 scheduler steps (15 s).
There were 0 preemptions, every request returned all of its tokens, and KV usage peaked at 91.0 %. The run took 1,091
s and averaged about 12.7K prompt tokens/s.

The run at the stated N, 39 x (32,768 + 284 new tokens), passed the same way: 39/39 resident for 132 steps.

Stock at its stated maximum ran in the throughput sweep below, using vLLM's built-in Triton attention
(`attention_backend="TRITON_ATTN"`): 27 x 32K, all resident at once, 0 preemptions, KV usage 79.5 %. That engine
uses 16-token blocks and reports the same capacity: 21.48 GiB pool, stated 27.35x, steady 33. The image's own kernels
are not built for head_dim 128, and its PyTorch reference fallback cannot hold 8192-token chunks next to a 0.85 KV
pool.

### KV bytes per layer type (from the specs vLLM built)

| layer type | layers | stock | 800h |
|---|---|---|---|
| full attention (K+V, 4 kv heads x 128) | 2, 6, 10, 14, 18, 22 | bf16, 2,048 B/token/layer -> **12,288 B/token** | 6, 10, 14, 22: 800h quant, 200 B/token/layer content (204.8 allocated: 1280-token page of 256,000 B padded to 262,144; the padding holds the sinks) -> **800 B/token (819.2 allocated)**. 2, 18: window 640 + 4 sinks, no long-range KV |
| exact window (bf16, newest tokens) | all 6 full-attention layers | - | 640 tokens x 2,048 B x 6 = 7.86 MB / request (fixed) |
| native sliding window 256 (bf16) | 4, 8, 12, 16, 20 | 256 x 2,048 x 5 = 2.62 MB / request | same, 2.62 MB / request |
| short-conv state | 13 conv layers | 24,576 B content x 13 = 0.32 MB (vLLM pads each page to 131,072 B: 1.70 MB) | 0.32 MB content (padded to 262,144: 3.41 MB) |
| **per 32,832-token request, content** | | **~406 MB** | **~37 MB** (26.3 MB grows with length + 10.8 MB fixed) |
| **per request as vLLM allocates it (steady state)** | | **678 MB** (1,034 blocks x 640 KiB; hybrid-group padding) | **50 MB** (48 blocks x 1 MiB) |

The KV that grows with length goes from 12,288 to 800 B/token, 15.4x less (15.0x counting the sink padding).

## 1b. Throughput vs concurrency at 32K context (gpu_memory_utilization 0.85, prefill chunk 8192)

Every point is one burst of c requests of 32,768 prompt tokens, all submitted at once. Generation lengths are
staggered so that all c requests are resident and decoding together for 136-183 scheduler steps (`concurrency.py
--sweep`, the same staggering as section 1). All points ran through the shipped package (`--plugin`).

Column definitions:
- **decode tok/s:** aggregate = c x (steps - 1) / time, over the steps where all c requests are decoding. The
  context during that window is 32.9K-34.6K tokens.
- **TPOT:** that window's time per step.
- **prefill tok/s:** c x 32,768 / (time until the last request's prompt is fully prefilled).
- **TTFT:** measured from the burst start, so for a burst it is mostly queueing.
- **KV usage:** during the window.

| | c | decode tok/s (aggregate) | TPOT ms | prefill tok/s | TTFT median / max s | KV usage | preemptions | decode vs stock at c=27 |
|---|---|---|---|---|---|---|---|---|
| stock bf16 (Triton attention) | **27** (its max at 32K) | **431** | 62.6 | 11,252 | 42.2 / 78.6 | 79.5 % | 0 | 1.0x |
| 800h | 27 | 861 | 31.4 | 10,732 | 47.6 / 82.4 | 7.3 % | 0 | 2.0x |
| 800h | 64 | 1,719 | 37.2 | 12,329 | 89.5 / 170.1 | 16.4 % | 0 | 4.0x |
| 800h | 128 | 3,336 | 38.4 | 12,456 | 170.7 / 336.7 | 29.3 % | 0 | 7.7x |
| 800h | **398** | **5,124** | 77.7 | 12,363 | 524.7 / 1,054.9 | 90.7 % | 0 | **11.9x** |

- **Stock caps at 27-33 sequences of 32K on this card:** 27 is vLLM's stated maximum and 33 is the steady-state limit.
  800h holds 398 measured, 444 steady-state.
- **Same concurrency (c=27):** 800h decodes 2.0x faster than stock, because its decode reads 15x fewer KV bytes.
- **At each mode's own limit:** 800h at c=398 decodes 11.9x stock's aggregate at c=27, with TPOT 78 ms vs 63 ms.
- **Prefill is the same for both, about 11-12K tok/s.** The 800h prefill uses bf16 SDPA. A burst of 398 x 32K
  therefore takes 17.6 min to prefill, and that is what the TTFT column shows.
- **Not covered:** c=256 was dropped to save time. Stock at 33 was not measured, because vLLM's admission gate
  (stated 27.35) is what limits a burst.

## 2. Needles: 96 = seeds 3..14 x 1K/4K/16K/32K x depth 25/75 %

The construction is identical to `kv800_e2e.py:needle_items`: same RNG, same haystack (stdlib argparse, difflib,
inspect and typing source), same question. Decoding is greedy with 8 new tokens, and a hit means the 4-digit code
appears in the output. All 96 run in one batched `generate` call. Seeds 3-5 were seen during kernel development; seeds
6-14 are held out. The intervals are Wilson 95%.

| run | recall | 1K | 4K | 16K | 32K | dev 3-5 | held-out 6-14 |
|---|---|---|---|---|---|---|---|
| stock bf16 (prefill chunk 1024) | **96/96 = 100.0 % (96.2-100.0)** | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 72/72 |
| **800h, prefill chunk 2048** | **88/96 = 91.7 % (84.4-95.7)** | 24/24 | 23/24 | 19/24 | 22/24 | 22/24 | 66/72 |
| 800h, prefill chunk 8192 (kv800_e2e setting) | 79/96 = 82.3 % (73.5-88.6) | 24/24 | 22/24 | 15/24 | 18/24 | 19/24 | 60/72 |

- **Paired comparison (same items):** stock got every needle, so each 800h miss is a discordant pair. 800h is lower than
  stock with McNemar exact p = 0.008 at chunk 2048 (8 misses) and p < 1e-4 at chunk 8192 (17 misses).
- **No garbage outputs:** no output degenerated ("oyoyoy") in any run after the fix, all in one batch and without
  `--zero-kv`.
- **The misses are truncated codes:** the model writes the first digit token(s) and stops or guesses, for example
  " 739." for 7397 or " 1." for 2854. They sit mostly at 16K and 32K.
- **Stock chunk size:** stock ran at chunk 1024 to keep the image's PyTorch reference-attention fallback within memory.
  Stock attention is exact, so the chunk size does not change what it computes. The full 96 ran in 739 s.

## 3. Rollouts: 32 = seeds 0..7 x 8K/16K x depth 25/75 %

The construction matches `kv800_e2e.py:rollout_items`. Step 1 feeds the needle context and generates 1024 greedy
tokens. Step 2 feeds context + those tokens + the question and generates 8 tokens. Prefix caching is off, so step 2
re-prefills the whole sequence in chunks, and the question chunk reads the needle back from the cache under test.

| run | recall | 8K | 16K | dev 0-1 | held-out 2-7 |
|---|---|---|---|---|---|
| stock bf16 (chunk 1024, 8 per call) | **32/32 = 100.0 % (89.3-100.0)** | 16/16 | 16/16 | 8/8 | 24/24 |
| **800h, chunk 2048, 32 per call** | **28/32 = 87.5 % (71.9-95.0)** | 15/16 | 13/16 | 7/8 | 21/24 |
| 800h, chunk 8192, 32 per call | 28/32 = 87.5 % (71.9-95.0) | 16/16 | 12/16 | 6/8 | 22/24 |

- **Paired comparison:** 800h vs stock, McNemar exact p = 0.125 (4 discordant pairs at each chunk size).
- **The misses:** all at 16K, plus one off-by-one-digit at 8K (" 7950." for 7955). No output degenerated.
- **Stock throughput:** stock rollouts ran 8 per call (stock runs the reference attention at decode) and took 757 s.

## Notes

- Chunk size matters for 800h quality and not for stock. Stock attention is exact. The 800h prefill reads all earlier
  chunks back from the quantized cache, and a prompt that fits in one chunk is attended exactly while it is
  prefilled. So 1K and 4K needles at chunk 8192 mostly test the decode steps, while chunk 2048 is the harder setting.
  Stock runs at chunk 1024 because of the reference-attention memory limit.
- Throughput sweep: `PLUGIN=1 ./run_all.sh sweep_stock sweep` (32 min of GPU time).
- Commands: `PLUGIN=1 ./run_all.sh needles_kv800 needles_kv800_8k rollouts_kv800 rollouts_kv800_8k rollouts_stock`
  (post-fix; about 25 min of GPU time, 13 of it stock rollouts).

## History: pre-fix 800h numbers (superseded)

**Root cause.** The short-conv prefill cleared a new request's conv state by multiplying it by 0 (`state * 0`). The KV
pool is shared, and a recycled page that last held 800h data can contain any bytes, some of which are NaN when read as
bf16. NaN * 0 is NaN, so a request that got such a page started from a NaN conv state and produced "oyoyoy...".

The fix uses `torch.where`. It is in `lfm3_kv800/vllm_model.py`, which also covers decode via `has_initial_states_d`,
and in the in-tree prefill in `vllm_lfm3/lfm3.py`. The bug only hit requests that ran after others in the same engine.
That is why earlier small runs swung (needles 20-22/24, rollouts 6-7/8).

Pre-fix measurements, in-tree code. They are kept here only to document the bug:

| pre-fix run | needles 96 | rollouts 32 | "oyoyoy" outputs |
|---|---|---|---|
| 800h chunk 2048, one batch | 71/96 (74.0 %) | 14/32 | 19 needles / 15 rollouts |
| 800h chunk 8192, one batch | 64/96 (66.7 %) | 7/32 | 18 / 25 |
| 800h chunk 2048, 8 per call | 72/96 | - | 10: the first request of every call except the first |
| 800h chunk 2048, 8 per call, every KV tensor zeroed before each call (`--zero-kv`) | 82/96 | - | 0 |

The post-fix run in one batch (88/96) is better than the zeroed workaround (82/96). Zeroing only cleans the pool
between calls. Inside a call, a page freed by a finished request can still come back as conv state holding uint8 800h
bytes that read as NaN.
