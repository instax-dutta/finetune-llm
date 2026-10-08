#!/usr/bin/env python3
"""Neutral held-out evaluation for LoRA adapters.

Loads the base model once with a plain transformers+peft path (no Soup, no
Unsloth anywhere in the import chain), then scores every adapter with the same
loader and the same rows. Also reports adapter health so a zero-norm adapter
cannot pass as a trained one.

Input rows (val.jsonl), one JSON object per line, any of:
  {"prompt_ids": [...], "completion_ids": [...]}   # only completion ids supervised
  {"prompt": "...", "completion": "..."}           # tokenized, completion supervised
  {"messages": [{"role": ..., "content": ...}]}     # full text supervised (no mask)
  {"text": "..."}                                   # full text supervised

Usage:
  eval_adapter.py --model <hf-id> --val-data data/val.jsonl \
      --adapter baseline=out/run_a --adapter candidate=out/run_b --out eval.json
"""
import argparse
import json
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from verify_run import read_safetensors
except Exception:
    read_safetensors = None


def adapter_health(path):
    f = None
    if os.path.isdir(path):
        f = os.path.join(path, "adapter_model.safetensors")
    elif path.endswith(".safetensors"):
        f = path
    if not f or not os.path.exists(f) or read_safetensors is None:
        return None
    tensors = read_safetensors(f)
    if tensors is None:
        return None
    total = sum(a.size for a in tensors.values())
    nonzero = sum(int((abs(a).max() if a.size else 0) > 0) * a.size for a in tensors.values())
    frob2 = sum(float((a.astype("float64") ** 2).sum()) for a in tensors.values() if a.size)
    return {"tensors": len(tensors), "params": total,
            "nonzero_frac": round(nonzero / total, 4) if total else 0.0,
            "frobenius": round(math.sqrt(frob2), 2)}


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_examples(rows, tok):
    examples = []
    unmasked = 0
    for r in rows:
        if "prompt_ids" in r and "completion_ids" in r:
            p, c = list(r["prompt_ids"]), list(r["completion_ids"])
        elif "prompt" in r and "completion" in r:
            p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            c = tok(r["completion"], add_special_tokens=False)["input_ids"]
        elif "messages" in r:
            ids = tok.apply_chat_template(r["messages"], tokenize=True, add_generation_prompt=False)
            p, c = [], list(ids)
            unmasked += 1
        elif "text" in r:
            p, c = [], list(tok(r["text"], add_special_tokens=False)["input_ids"])
            unmasked += 1
        else:
            continue
        if not c:
            continue
        examples.append({"input_ids": p + c, "labels": [-100] * len(p) + c})
    return examples, unmasked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--adapter", action="append", required=True, metavar="name=path")
    ap.add_argument("--out", help="write results JSON here")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-quantize", action="store_true", help="load the base unquantized (slow; for CPU/MPS tests)")
    ap.add_argument("--include-base", action="store_true", help="also score the raw base model")
    args = ap.parse_args()

    adapters = []
    for spec in args.adapter:
        if "=" not in spec:
            print(f"--adapter wants name=path, got: {spec}", file=sys.stderr)
            return 2
        name, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"adapter not found: {path}", file=sys.stderr)
            return 2
        adapters.append((name, path))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or ("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    load_kwargs = {}
    if device.startswith("cuda") and not args.no_quantize:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    else:
        load_kwargs["dtype"] = torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.to(device)
    model.eval()

    rows = load_rows(args.val_data)
    examples, unmasked = build_examples(rows, tok)
    if not examples:
        print("no usable val rows", file=sys.stderr)
        return 2

    def collate(batch):
        mx = max(len(x["input_ids"]) for x in batch)
        ids = torch.full((len(batch), mx), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(batch), mx), dtype=torch.long)
        lab = torch.full((len(batch), mx), -100, dtype=torch.long)
        for i, x in enumerate(batch):
            n = len(x["input_ids"])
            ids[i, :n] = torch.tensor(x["input_ids"])
            att[i, :n] = 1
            lab[i, :n] = torch.tensor(x["labels"])
        return {"input_ids": ids.to(device), "attention_mask": att.to(device), "labels": lab.to(device)}

    def score():
        total, tokens = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(examples), args.batch_size):
                batch = collate(examples[i:i + args.batch_size])
                out = model(**batch)
                k = int((batch["labels"] != -100).sum())
                total += float(out.loss) * k
                tokens += k
        return (total / tokens if tokens else float("nan")), tokens

    results = {"model": args.model, "val_rows": len(examples),
               "supervised_tokens": None, "device": device,
               "note": "full-text rows had no prompt/completion split; completion tokens were not masked" if unmasked else None,
               "adapters": {}}

    from peft import PeftModel

    if args.include_base:
        loss, tokens = score()
        results["adapters"]["base"] = {"held_out_loss": round(loss, 5), "tokens": tokens, "health": None}

    for name, path in adapters:
        m = PeftModel.from_pretrained(model, path, adapter_name=name)
        m.eval()
        loss, tokens = score()
        results["adapters"][name] = {
            "held_out_loss": round(loss, 5),
            "tokens": tokens,
            "health": adapter_health(path),
        }
        m.delete_adapter(name)

    if results["adapters"]:
        results["supervised_tokens"] = next(iter(results["adapters"].values()))["tokens"]

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
