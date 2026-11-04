# vLLM 0.32 for Intel B70 + lfm3-kv800 (prebuilt SYCL op inside the package, no compiler needed).
# Build from the package root, after scripts/build_sycl.sh has produced lfm3_kv800/_lib/kv800_sycl.so:
#   docker build -f docker/Dockerfile -t lfm3-kv800:b70 .
FROM ghcr.io/riceharvest/b70-vllm:v0.32.0-b70.7
COPY pyproject.toml README_INSTALL.md /opt/lfm3-kv800/
COPY lfm3_kv800 /opt/lfm3-kv800/lfm3_kv800
RUN test -f /opt/lfm3-kv800/lfm3_kv800/_lib/kv800_sycl.so \
 && pip install --no-deps --no-build-isolation --no-cache-dir /opt/lfm3-kv800 \
 && rm -rf /opt/lfm3-kv800/build
ENV LFM3_KV800=1
