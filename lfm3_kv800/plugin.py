"""vLLM plugin (entry point group ``vllm.general_plugins``, name ``lfm3_kv800``): registers LFM3 with vLLM.

vLLM calls register() in every process it starts (API server, engine core, workers), so ``vllm serve`` and ``LLM()``
work with the default multiprocess engine.
  * LFM3_KV800=1: always registers the vendored ``Lfm3ForCausalLM`` (lfm3_kv800.vllm_model), even if vLLM ships its
    own (a vLLM-provided LFM3 would not use the 800h cache), and adds a config hook that forces enforce_eager,
    compilation mode 0 (NONE) and prefix caching off, which the 800h path needs.
  * otherwise: registers the vendored model only if vLLM has no ``Lfm3ForCausalLM`` (stock bf16 attention).
"""
import os

from . import get_logger

ARCH = "Lfm3ForCausalLM"
MODEL = "lfm3_kv800.vllm_model:Lfm3ForCausalLM"
_done = False


def kv800_enabled() -> bool:
    return os.environ.get("LFM3_KV800", "0") == "1"


def register() -> None:
    global _done
    if _done:
        return
    _done = True
    log = get_logger()
    from vllm import ModelRegistry

    upstream = ARCH in ModelRegistry.get_supported_archs()
    if kv800_enabled():
        ModelRegistry.register_model(ARCH, MODEL)
        _add_config_hook()
        log.info("LFM3_KV800=1: %s -> %s with the 800h KV cache%s", ARCH, MODEL,
                 " (overrides the LFM3 model that ships with vLLM)" if upstream else "")
    elif not upstream:
        ModelRegistry.register_model(ARCH, MODEL)
        log.info("%s -> %s (stock attention; set LFM3_KV800=1 for the 800h KV cache)", ARCH, MODEL)
    else:
        log.info("%s: using the model that ships with vLLM (set LFM3_KV800=1 for the 800h KV cache)", ARCH)


def _add_config_hook() -> None:
    try:
        from vllm.model_executor.models.config import MODELS_CONFIG_MAP, VerifyAndUpdateConfig
    except ImportError:                              # other vLLM layout: pass the flags on the command line
        return
    prev = MODELS_CONFIG_MAP.get(ARCH)
    if prev is not None and getattr(prev, "_lfm3_kv800", False):
        return

    class Lfm3KV800Config(VerifyAndUpdateConfig):
        _lfm3_kv800 = True

        @staticmethod
        def verify_and_update_model_config(model_config) -> None:
            if prev is not None:
                prev.verify_and_update_model_config(model_config)

        @staticmethod
        def verify_and_update_config(vllm_config) -> None:
            if prev is not None:
                prev.verify_and_update_config(vllm_config)
            changed = []
            mc, cc, comp = vllm_config.model_config, vllm_config.cache_config, vllm_config.compilation_config
            if not mc.enforce_eager:
                mc.enforce_eager = True
                changed.append("enforce_eager=True")
            if cc.enable_prefix_caching is not False:
                cc.enable_prefix_caching = False
                changed.append("enable_prefix_caching=False")
            try:
                from vllm.config.compilation import CompilationMode
                if comp.mode != CompilationMode.NONE:
                    comp.mode = CompilationMode.NONE
                    changed.append("compilation mode NONE")
            except ImportError:
                pass
            if changed:
                get_logger().info("800h KV cache needs: %s (set)", ", ".join(changed))

    MODELS_CONFIG_MAP[ARCH] = Lfm3KV800Config
