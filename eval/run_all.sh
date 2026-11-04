#!/usr/bin/env bash
# Release eval for the 800h KV cache on LFM3-1.7B in vLLM 0.32 (image ghcr.io/riceharvest/b70-vllm:v0.32.0-b70.7, B70).
# usage: run_all.sh [step ...]   (default: every step except the optional "stretch")
#   steps: conc_stock conc_kv800 needles_kv800 needles_kv800_8k rollouts_kv800 rollouts_kv800_8k needles_stock
#          rollouts_stock stretch sweep_stock sweep (diagnostics: diag_g8 *_zkv)
# env: PLUGIN=1 -> the shipped package (the directory above eval/) is pip-installed inside the container
#      (pip install --no-deps --no-build-isolation of a /tmp copy), PYTHONPATH is left empty, and the scripts get
#      --plugin: vLLM loads the model through the package's vllm.general_plugins entry point. The scripts abort if the
#      model is not served by lfm3_kv800 or if the in-tree modules (vllm_lfm3, lfm3kv) were imported.
#      PLUGIN=0 (default): in-tree code via PYTHONPATH and vllm_lfm3.register().
# Each container run holds the B70 lock (flock) for its whole lifetime; logs and JSON land in eval/out/.
set -u
EVAL=$(cd "$(dirname "$0")" && pwd)
OUT=$EVAL/out; mkdir -p "$OUT"
# the shared B70 lock file (override with B70_LOCK); default: the b70.lock under /tmp/<agent>-<uid>/
LOCK=${B70_LOCK:-$(ls -d /tmp/*-"$(id -u)"/b70.lock 2>/dev/null | head -1)}
[ -n "$LOCK" ] && [ -e "$LOCK" ] || { echo "B70 lock file not found; set B70_LOCK"; exit 2; }
IMG=ghcr.io/riceharvest/b70-vllm:v0.32.0-b70.7
PKG=$(dirname "$EVAL")
if [ "${PLUGIN:-0}" = 1 ]; then
  PLUG="--plugin"; PYPATH=""
  PRE="rm -rf /tmp/pkg && mkdir /tmp/pkg && cp -r $PKG/pyproject.toml $PKG/README_INSTALL.md $PKG/lfm3_kv800 /tmp/pkg/ && pip install -q --no-deps --no-build-isolation /tmp/pkg && pip show lfm3-kv800 | head -2 &&"
else
  PLUG=""; PYPATH=/mnt/ssd/lfm3-kv:/mnt/ssd/tiny-moe-box/tinykv; PRE=""
fi

run() {   # run <log name> <script> <args...>
  local name=$1 script=$2; shift 2
  echo "[$(date +%H:%M:%S)] start $name: $script $*"
  local t0=$SECONDS
  flock "$LOCK" docker run --rm --name "lfm3-eval-$name" --device /dev/dri -v /dev/dri:/dev/dri:ro --shm-size 8g --memory=24g \
    -e HF_HUB_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e LFM3KV_PROF=0 -e LFM3KV_XMX=${LFM3KV_XMX:-1} \
    -e LFM3KV_BACKEND=${LFM3KV_BACKEND:-sycl} -e TORCH_XPU_ARCH_LIST=bmg-g31 \
    -e PYTHONPATH=$PYPATH \
    -e PATH=/home/dario/oneapi2026/opt/intel/oneapi/compiler/2026.0/bin:/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    -v /mnt/ssd/huggingface:/mnt/ssd/huggingface:ro -v /mnt/ssd/lfm3-kv:/mnt/ssd/lfm3-kv \
    -v /mnt/ssd/tiny-moe-box/tinykv:/mnt/ssd/tiny-moe-box/tinykv:ro \
    -v /home/dario/oneapi2026:/home/dario/oneapi2026:ro -v /usr/lib64/python3.14:/usr/lib64/python3.14:ro \
    --entrypoint bash "$IMG" -lc "cd /tmp && $PRE python $EVAL/$script $* $PLUG; rc=\$?; chown -R 1000:1000 /mnt/ssd/lfm3-kv/build $OUT $EVAL/__pycache__ 2>/dev/null; exit \$rc" \
    > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] done  $name rc=$rc in $(( (SECONDS - t0) / 60 )) min"
  grep -h "^RESULT" "$OUT/$name.log"
}

STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(conc_stock conc_kv800 needles_kv800 needles_kv800_8k rollouts_kv800 rollouts_kv800_8k
                                needles_stock rollouts_stock)
for s in "${STEPS[@]}"; do
  case $s in
    # capacity at gpu_memory_utilization 0.85, 32768 + 64-token requests, prefill chunk 8192 (both modes identical)
    conc_stock)        run $s concurrency.py --mode stock ;;
    conc_kv800)        run $s concurrency.py --mode kv800 --confirm stated ;;
    # optional: confirmation at 95% of the steady-state capacity (long: ~N x 32K prefill)
    # throughput vs concurrency at 32K, 800h only (stock decode cannot run on this image; stock caps at 33 x 32K)
    sweep_stock)       run $s concurrency.py --mode stock --attn-backend TRITON_ATTN --sweep stated --window 0 --tag sweep ;;
    sweep)             run $s concurrency.py --mode kv800 --sweep 27,64,128,398 --window 0 --max-num-seqs 768 --tag sweep ;;
    stretch)           run $s concurrency.py --mode kv800 --confirm steady --stagger --max-num-seqs 768 --tag steady ;;
    # quality: prefill chunk 2048 (the harder setting for 800h: more of every prompt is read back from the cache),
    # plus 800h at chunk 8192 (kv800_e2e.py's setting). Stock attention is exact, so its chunk size (1024, chosen to keep
    # the image's PyTorch reference attention fallback within memory) does not change what it computes.
    needles_kv800)     run $s needles.py --mode kv800 --mnbt 2048 ;;
    needles_kv800_8k)  run $s needles.py --mode kv800 --mnbt 8192 ;;
    rollouts_kv800)    run $s rollouts.py --mode kv800 --mnbt 2048 ;;
    rollouts_kv800_8k) run $s rollouts.py --mode kv800 --mnbt 8192 ;;
    needles_stock)     run $s needles.py --mode stock --mnbt 1024 --util 0.6 --budget-min 40 ;;
    rollouts_stock)    run $s rollouts.py --mode stock --mnbt 1024 --util 0.6 --group 8 --budget-min 25 ;;
    # stale-block check: the in-tree 800h path gives garbage ("oyoyoy...") for requests that get KV blocks recycled from
    # earlier requests. *_zkv: one seed per generate call (no recycling inside a call) and every KV tensor zeroed before
    # each call, i.e. every request starts on clean blocks as in a fresh engine. diag_g8: same grouping without zeroing.
    diag_g8)           run $s needles.py --mode kv800 --mnbt 2048 --group 8 ;;
    needles_kv800_zkv)     run $s needles.py --mode kv800 --mnbt 2048 --group 8 --zero-kv ;;
    needles_kv800_8k_zkv)  run $s needles.py --mode kv800 --mnbt 8192 --group 8 --zero-kv ;;
    rollouts_kv800_zkv)    run $s rollouts.py --mode kv800 --mnbt 2048 --group 8 --zero-kv ;;
    rollouts_kv800_8k_zkv) run $s rollouts.py --mode kv800 --mnbt 8192 --group 8 --zero-kv ;;
    *) echo "unknown step $s"; exit 2 ;;
  esac
done
