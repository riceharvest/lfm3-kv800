#!/usr/bin/env bash
# Build the prebuilt SYCL op lfm3_kv800/_lib/kv800_sycl.so (AOT for bmg-g31 + SPIR-V) against the torch of the target
# image, inside a container from that image. Needs no GPU. Writes _lib/BUILD_INFO.json (torch, compiler, arch, hashes).
#
#   scripts/build_sycl.sh                      # defaults below
#   IMAGE=... ONEAPI=... scripts/build_sycl.sh
#
# ONEAPI: an oneAPI install root holding compiler/<ver>/bin/icpx (mounted read-only, only for this build).
set -euo pipefail
IMAGE=${IMAGE:-ghcr.io/riceharvest/b70-vllm:v0.32.0-b70.7}
ONEAPI=${ONEAPI:-/home/dario/oneapi2026/opt/intel/oneapi}
CVER=${CVER:-2026.0}
ARCH=${ARCH:-bmg-g31}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CBIN=$ONEAPI/compiler/$CVER/bin
[ -x "$CBIN/icpx" ] || { echo "icpx not found at $CBIN" >&2; exit 1; }

docker run --rm --memory=24g --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e MAX_JOBS=4 -e TORCH_XPU_ARCH_LIST="$ARCH" -e CBIN="$CBIN" -e ARCH="$ARCH" -e IMAGE="$IMAGE" \
  -e PATH="$CBIN:/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  -v "$ONEAPI:$ONEAPI:ro" -v "$ROOT:/src" \
  --entrypoint bash "$IMAGE" -lc '
set -euo pipefail
python - <<"EOF"
import hashlib, json, os, shutil, subprocess, torch
from torch.utils import cpp_extension as ce
src = "/src/lfm3_kv800/csrc/kv800_sycl.sycl"
bd = "/tmp/kv800-build"; os.makedirs(bd, exist_ok=True)
ce.load(name="kv800_sycl", sources=[src], build_directory=bd, is_python_module=False, verbose=True,
        extra_cflags=["-O3"], extra_sycl_cflags=["-O3"])
out = "/src/lfm3_kv800/_lib/kv800_sycl.so"
os.makedirs(os.path.dirname(out), exist_ok=True)
shutil.copyfile(os.path.join(bd, "kv800_sycl.so"), out)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
info = {"torch": torch.__version__, "arch": os.environ["ARCH"], "targets": "spir64_gen,spir64",
        "compiler": subprocess.run(["icpx", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
        "image": os.environ["IMAGE"], "source_sha256": sha(src), "so_sha256": sha(out), "so_bytes": os.path.getsize(out)}
json.dump(info, open("/src/lfm3_kv800/_lib/BUILD_INFO.json", "w"), indent=1)
print(json.dumps(info, indent=1))
EOF
'
ls -la "$ROOT/lfm3_kv800/_lib"
