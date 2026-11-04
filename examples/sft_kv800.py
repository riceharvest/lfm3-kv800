"""Minimal LoRA SFT of LFM3-1.7B with the 800h KV cache in the loop (plain PyTorch + PEFT).

    python examples/sft_kv800.py DATA [--model LiquidAI/LFM3-1.7B-Base] [--tokens 1000000] [--out run/sft_kv800]

DATA is a local file:
  * .txt (or anything not .jsonl): one long text, cut into --seq-len token rows (loss on every token);
  * .jsonl with {"text": ...} per line (same), or {"prompt": ..., "completion": ...} per line: loss on the completion
    only, and the prompt is marked as prefill (hook.prefill) like the vLLM sampler computes it.
Defaults follow the measured recovery probe: LoRA r32 on q/k/v/o of every attention layer, lr 5e-5, 2048-token rows
(1M tokens closed 40% of the 800h CE gap in 3.5 min on one B70). Saves the adapter to --out; serve the merged weights
with the lfm3-kv800 vLLM plugin."""
import argparse, json, math, os, random, time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from lfm3_kv800.train import apply_transformers_compat, patch_for_training


def rows_from(path, tok, seq_len, n_tokens, seed):
    """-> list of (ids, prompt_len). prompt_len 0 = plain text row."""
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    rows = []
    if path.endswith(".jsonl"):
        texts = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if "prompt" in d and "completion" in d:
                    p = tok(d["prompt"], add_special_tokens=False)["input_ids"]
                    c = tok(d["completion"], add_special_tokens=False)["input_ids"] + [eos]
                    if len(p) < seq_len:
                        rows.append(((p + c)[:seq_len], len(p)))
                else:
                    texts.append(d["text"])
        stream = []
        for t in texts:
            stream += tok(t, add_special_tokens=False)["input_ids"] + [eos]
    else:
        stream = tok(open(path).read(), add_special_tokens=False)["input_ids"]
    rows += [(stream[i:i + seq_len], 0) for i in range(0, len(stream) - seq_len + 1, seq_len)]
    random.Random(seed).shuffle(rows)
    out, n = [], 0
    for r in rows:
        if n >= n_tokens:
            break
        out.append(r); n += len(r[0])
    return out


def batches(rows, bs, pad_id, device):
    """Right-padded batches: input_ids, attention_mask, labels (-100 on padding and prompts), prompt lengths."""
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        T = max(len(ids) for ids, _ in chunk)
        x = torch.full((len(chunk), T), pad_id, dtype=torch.long)
        am = torch.zeros_like(x)
        for j, (ids, _) in enumerate(chunk):
            x[j, :len(ids)] = torch.tensor(ids); am[j, :len(ids)] = 1
        labels = x.masked_fill(am == 0, -100)
        plen = torch.tensor([p for _, p in chunk])
        for j, p in enumerate(plen.tolist()):
            labels[j, :p] = -100
        yield x.to(device), am.to(device), labels.to(device), plen.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--model", default="LiquidAI/LFM3-1.7B-Base")
    ap.add_argument("--out", default="run/sft_kv800")
    ap.add_argument("--tokens", type=int, default=1_000_000)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--rule", default="cache", choices=["cache", "last128"])
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else
                       "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")

    apply_transformers_compat()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if dev != "cpu" else torch.float32,
                                                 trust_remote_code=True, attn_implementation="sdpa").to(dev)
    lt = model.config.layer_types
    targets = [f"layers.{l}.self_attn.{p}" for l in range(len(lt)) if lt[l] != "conv"
               for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.rank, lora_dropout=0.0, target_modules=targets))
    model.print_trainable_parameters()
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    hook = patch_for_training(model, mode="800h", rule=a.rule)      # after PEFT is fine (before works too)

    rows = rows_from(a.data, tok, a.seq_len, a.tokens, seed=0)
    pad = model.config.pad_token_id if model.config.pad_token_id is not None else (tok.pad_token_id or 0)
    steps = math.ceil(len(rows) / a.batch)
    print(f"{len(rows)} rows, {sum(len(r) for r, _ in rows)} tokens, {steps} steps on {dev}, rule {a.rule}")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    model.train(); t0 = time.time(); run = 0.0
    for i, (x, am, labels, plen) in enumerate(batches(rows, a.batch, pad, dev)):
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (i + 1) / 20)                    # 20-step warmup
        with hook.prefill(plen if bool((plen > 0).any()) else None):
            loss = model(input_ids=x, attention_mask=am, labels=labels, use_cache=False).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        run += loss.item()
        if (i + 1) % 20 == 0 or i + 1 == steps:
            n = (i % 20) + 1
            print(f"step {i + 1}/{steps} loss {run / n:.3f} {(i + 1) * a.batch * a.seq_len / (time.time() - t0):.0f} tok/s",
                  flush=True)
            run = 0.0
    hook.remove()
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    print(f"adapter saved to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
