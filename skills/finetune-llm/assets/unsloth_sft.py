#!/usr/bin/env python3
"""Unsloth-native QLoRA SFT template.

Use when Unsloth is the chosen engine but Soup is not in the loop (or the
Soup install conflict forces a standalone script). Quality is identical to
Soup's unsloth backend; this script adds exact step control and timing.

Hardware-aware defaults:
  - bf16 only on sm_80+ with real hardware support, fp16 elsewhere (T4/V100)
  - dataset preprocessing pinned to 1 worker on <= 4-core boxes
  - fused LM head / cross-entropy comes from Unsloth itself

Example:
  python unsloth_sft.py --model Qwen/Qwen2.5-7B-Instruct --data data/text.jsonl \
    --out out/run1 --steps 200 --seq 1024 --bs 1 --accum 8
"""
import argparse
import json
import os
import statistics
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import unsloth  # noqa: F401  MUST precede trl/transformers/peft

import torch
from datasets import load_dataset
from transformers.trainer_callback import TrainerCallback


def pick_fp16():
    if not torch.cuda.is_available():
        return False
    try:
        real_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        real_bf16 = torch.cuda.is_bf16_supported()
    cc_major = torch.cuda.get_device_properties(0).major
    return not (real_bf16 and cc_major >= 8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="jsonl with a 'text' field (or prompt/completion fields)")
    ap.add_argument("--out", required=True, help="adapter output dir")
    ap.add_argument("--steps", type=int, default=None, help="exact optimizer steps (mutually exclusive with --epochs)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import FastLanguageModel

    torch.manual_seed(args.seed)
    raw = load_dataset("json", data_files={"train": args.data}, split="train")

    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.seq, dtype=None, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable_params={trainable}", flush=True)

    cores = os.cpu_count() or 1
    if cores <= 4:
        try:
            import unsloth_zoo.dataset_num_proc as _dnp
            if hasattr(_dnp, "DATASET_NUM_PROC"):
                _dnp.DATASET_NUM_PROC = 1
        except Exception:
            pass

    fp16 = pick_fp16()

    class Timings(TrainerCallback):
        def __init__(self):
            self.steps, self.losses, self._t = [], [], None

        def on_step_begin(self, a, state, control, **kw):
            if self._t is not None:
                torch.cuda.synchronize()
                self.steps.append(time.perf_counter() - self._t)
            torch.cuda.synchronize()
            self._t = time.perf_counter()

        def on_log(self, a, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                self.losses.append(round(float(logs["loss"]), 5))

    total_steps = args.steps or int(len(raw) * args.epochs / (args.bs * args.accum))
    warmup = max(1, int(round(0.03 * total_steps)))
    timings = Timings()
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=raw,
        dataset_text_field="text",
        max_length=args.seq,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.bs,
            gradient_accumulation_steps=args.accum,
            max_steps=args.steps if args.steps else -1,
            num_train_epochs=args.epochs if not args.steps else 1.0,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=warmup,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            seed=args.seed,
            data_seed=args.seed,
            bf16=not fp16,
            fp16=fp16,
            optim="adamw_torch",
            gradient_checkpointing=False,
            remove_unused_columns=False,
            logging_first_step=True,
            dataloader_num_workers=0 if cores <= 4 else 2,
        ),
        callbacks=[timings],
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    result = {
        "steps_measured": len(timings.steps),
        "wall_seconds": round(wall, 1),
        "median_step_s": round(statistics.median(timings.steps), 3) if timings.steps else None,
        "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
        "trainable_params": trainable,
        "final_loss": timings.losses[-1] if timings.losses else None,
        "losses": timings.losses,
        "fp16": fp16,
    }
    with open(os.path.join(args.out, "run_summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "losses"}, indent=2))


if __name__ == "__main__":
    main()
