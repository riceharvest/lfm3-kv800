"""UNTESTED: TRL GRPO (LoRA) of LFM3-1.7B with the 800h KV cache on both sides of the RL loop.

UNTESTED. TRL is not installed where this was written, so this file has never been run. It was written against the
TRL v1.17.1 source (GRPOTrainer, GRPOConfig, trl/generation/vllm_generation.py). Treat it as a starting point.

  * Sampling: vLLM in colocate mode (inside the trainer process), with the lfm3-kv800 vLLM plugin
    (entry point `lfm3_kv800.plugin:register`, enabled by LFM3_KV800=1). The completions are sampled from the
    800h cache.
  * Training: the policy forward is patched with lfm3_kv800.train.patch_for_training. The log-probs TRL computes
    (old / current / ref policy) see the same cache as the sampler, so the RL stays on-policy for the model that
    gets deployed. Prompt columns are marked as prefill, which is how vLLM computes them.

    LFM3_KV800=1 python examples/grpo_trl_kv800.py PROMPTS.jsonl [--model LiquidAI/LFM3-1.7B-Base] [--out run/grpo_kv800]

PROMPTS.jsonl holds one {"prompt": str, "answer": str} per line, read from local disk. The reward is a toy: 1 if the
answer string appears in the completion, else 0. Replace reward_contains with your own reward.

Known gaps (also untested):
  * TRL's colocate engine uses max_num_batched_tokens=4096, shared across the batch. A long prompt (> 383 tokens)
    can therefore be prefilled in several chunks, and its later chunks then see quantized history that the training
    forward computes as exact. Completion tokens are unaffected. vllm_importance_sampling_correction (on by default)
    absorbs the small residual.
  * The kv800 vLLM backend needs eager mode and no prefix caching. TRL exposes no engine kwargs, so this script wraps
    vllm.LLM.__init__ to set them. Drop the wrapper if the plugin already forces them.
  * TRL patches the model with a fused LM head (fused_lm_head=True) and runs the backbone directly. The hook sits on
    the LFM3 backbone, so it still applies. Whether TRL's fused head handles this remote-code model is untested.
  * Weight sync to vLLM merges the LoRA weights and calls load_weights on the plugin's LFM3 model. The parameter names
    must match the plugin's loader.
"""
import argparse, os

os.environ.setdefault("LFM3_KV800", "1")                 # before vLLM is imported: the plugin registers the cache

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from lfm3_kv800.train import apply_transformers_compat, patch_for_training


def force_kv800_engine_args():
    """Colocated vLLM is built inside GRPOTrainer as LLM(...) with no engine-kwargs hook: inject what the kv800
    backend needs (eager, no prefix caching)."""
    import vllm
    orig = vllm.LLM.__init__
    if getattr(orig, "_kv800", False):
        return

    def init(self, *args, **kw):
        kw["enforce_eager"] = True
        kw["enable_prefix_caching"] = False
        return orig(self, *args, **kw)

    init._kv800 = True
    vllm.LLM.__init__ = init


class KV800GRPOTrainer(GRPOTrainer):
    """GRPOTrainer whose policy forward runs on the 800h cache. Prompt columns are scored as prefill."""

    def __init__(self, *args, kv800_rule="cache", **kwargs):
        super().__init__(*args, **kwargs)
        # The model is PEFT-wrapped (and maybe accelerate-wrapped) by now. The hook finds the LFM3 backbone inside.
        self.kv800 = patch_for_training(self.model, mode="800h", rule=kv800_rule)
        if self.ref_model is not None:                     # beta > 0 without PEFT: the reference sees the same cache
            self.kv800_ref = patch_for_training(self.ref_model, mode="800h", rule=kv800_rule)

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, *args, **kwargs):
        # input_ids = [left-padded prompt | completion]. The first input_ids.shape[1] - logits_to_keep columns are the
        # prompt that vLLM prefilled. The completion columns are its decode steps.
        with self.kv800.prefill(input_ids.shape[1] - logits_to_keep):
            ref = getattr(self, "kv800_ref", None)
            if ref is None:
                return super()._get_per_token_logps_and_entropies(model, input_ids, attention_mask, logits_to_keep,
                                                                  *args, **kwargs)
            with ref.prefill(input_ids.shape[1] - logits_to_keep):
                return super()._get_per_token_logps_and_entropies(model, input_ids, attention_mask, logits_to_keep,
                                                                  *args, **kwargs)


def reward_contains(completions, answer, **kwargs):
    return [1.0 if a and a in c else 0.0 for c, a in zip(completions, answer)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts")
    ap.add_argument("--model", default="LiquidAI/LFM3-1.7B-Base")
    ap.add_argument("--out", default="run/grpo_kv800")
    ap.add_argument("--rule", default="cache", choices=["cache", "last128"])
    a = ap.parse_args()

    apply_transformers_compat()
    force_kv800_engine_args()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, trust_remote_code=True,
                                                 attn_implementation="sdpa")
    lt = model.config.layer_types
    targets = [f"layers.{l}.self_attn.{p}" for l in range(len(lt)) if lt[l] != "conv"
               for p in ("q_proj", "k_proj", "v_proj", "o_proj")]

    ds = load_dataset("json", data_files=a.prompts, split="train")      # local file
    args = GRPOConfig(
        output_dir=a.out,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        vllm_importance_sampling_correction=True,        # absorbs the chunked-prefill residual (see header)
        num_generations=8,
        max_completion_length=512,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_train_epochs=1,
        bf16=True,
        logging_steps=5,
        save_steps=200,
        report_to="none",
    )
    trainer = KV800GRPOTrainer(
        model=model,
        reward_funcs=reward_contains,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0, target_modules=targets, task_type="CAUSAL_LM"),
        kv800_rule=a.rule,
    )
    trainer.train()
    trainer.save_model(a.out)


if __name__ == "__main__":
    main()
