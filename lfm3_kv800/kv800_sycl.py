"""SYCL backend of kv800.decode for Intel Xe2 (B70 / bmg-g31): torch.ops.kv800.{qdecode, xdecode, combine}.

Load order (first that works):
  1. the prebuilt op shipped in the package (_lib/kv800_sycl.so, AOT for bmg-g31 + SPIR-V), only when this torch is
     exactly the one in _lib/BUILD_INFO.json (the op is tied to the libtorch C++ ABI);
  2. a JIT build of csrc/kv800_sycl.sycl with torch.utils.cpp_extension, only when icpx (oneAPI 2026.x) is on PATH
     (build cache: $LFM3KV_BUILD_DIR or ~/.cache/lfm3_kv800, TORCH_XPU_ARCH_LIST defaults to bmg-g31);
  3. otherwise ImportError, and kv800.decode moves on to Triton.
Env LFM3KV_XMX=0 selects the SIMT quantized kernel instead of the XMX (DPAS) one."""
import json, math, os, shutil
import torch

from . import kv800

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "_lib", "kv800_sycl.so")
INFO = os.path.join(HERE, "_lib", "BUILD_INFO.json")
SRC = os.path.join(HERE, "csrc", "kv800_sycl.sycl")
XMX = int(os.environ.get("LFM3KV_XMX", "1"))
TARGET = 2048 if XMX else 4096
SOURCE = None          # how the op was obtained (for the backend log line)
_OPS = None
_ERR = None


def _registered():
    try:
        torch.ops.kv800.qdecode
        return True
    except (AttributeError, RuntimeError):
        return False


def _prebuilt():
    """None on success, else the reason the prebuilt op was not used."""
    if not os.path.isfile(LIB):
        return "no prebuilt op in the package"
    try:
        with open(INFO) as f:
            want = json.load(f).get("torch")
    except (OSError, ValueError) as e:
        return f"unreadable {INFO}: {e}"
    if want != torch.__version__:
        return f"prebuilt op is for torch {want}, this is torch {torch.__version__}"
    try:
        torch.ops.load_library(LIB)
    except (OSError, RuntimeError) as e:
        return f"loading {LIB} failed: {e}"
    return None


def _jit():
    if shutil.which("icpx") is None:
        return "icpx not on PATH (no JIT build)"
    from torch.utils import cpp_extension as ce
    root = os.environ.get("LFM3KV_BUILD_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "lfm3_kv800")
    bd = os.path.join(root, "kv800-torch" + torch.__version__.replace("+", "-"))
    os.makedirs(bd, exist_ok=True)
    os.environ.setdefault("TORCH_XPU_ARCH_LIST", "bmg-g31")
    try:
        ce.load(name="kv800_sycl", sources=[SRC], build_directory=bd, is_python_module=False, verbose=False,
                extra_cflags=["-O3"], extra_sycl_cflags=["-O3"])
    except Exception as e:
        return f"JIT build failed: {type(e).__name__}: {str(e)[:300]}"
    return None


def ops():
    """torch.ops.kv800, loading the op on first use; raises ImportError when no SYCL op can be had."""
    global _OPS, _ERR, SOURCE
    if _OPS is not None:
        return _OPS
    if _ERR is not None:
        raise ImportError(_ERR)
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        _ERR = "torch has no usable XPU device"
        raise ImportError(_ERR)
    if _registered():
        SOURCE = "kv800 ops already registered in this process"
    else:
        why = _prebuilt()
        if why is None:
            SOURCE = f"prebuilt op {LIB}"
        else:
            why_jit = _jit()
            if why_jit is not None:
                _ERR = f"{why}; {why_jit}"
                raise ImportError(_ERR)
            SOURCE = f"JIT-built op ({why})"
    _OPS = torch.ops.kv800
    return _OPS


def decode(P, q, cos_t, sin_t, spb=None, target=None):
    o = ops()
    B = q.shape[0]; q = q.contiguous()
    nqmax = 0 if P.win else P.nqmax
    target = target or TARGET
    if spb is None: spb = max(1, math.ceil(nqmax * B * 4 / target)) if nqmax else 1
    ns = math.ceil(nqmax / spb) if nqmax else 0
    MO = torch.empty(B, ns + 1, 16, device=q.device, dtype=torch.float32)
    LO = torch.empty_like(MO); AO = torch.empty(B, ns + 1, 16, 128, device=q.device, dtype=torch.float32)
    scale = 128 ** -0.5 * kv800.LOG2E
    if ns:
        o.qdecode(q, P.qpool, P.boff, P.NQ, cos_t, sin_t, MO, LO, AO, ns, spb, scale, XMX)
    sk = P.sk if P.sk is not None else P.epool; sv = P.sv if P.sv is not None else P.epool
    dummy = P.N.to(torch.int64) if (P.skoff is None) else None
    skoff = P.skoff if P.skoff is not None else dummy; svoff = P.svoff if P.svoff is not None else dummy
    nsink = kv800.NSINK if (P.win and P.sk is not None) else 0
    o.xdecode(q, P.epool, P.eoff, P.ebs, P.e_hs, P.e_ts, P.e_v, sk, sv, skoff, svoff, P.N, P.NQ, cos_t, sin_t, MO, LO, AO,
              ns, scale, P.win, nsink)
    return o.combine(MO, LO, AO, ns)
