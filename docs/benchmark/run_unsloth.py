"""Arm C: Unsloth-native SFT. No Soup in the loop.

Same model, same rows, same hyper-parameters, same step count and seed as the
Soup arms -- so any delta is attributable to the framework.

Uses the pre-tokenized prompt/completion form (see prep_pc.py), which takes
unsloth's `_tokenize_pc` branch. Response tokens only are supervised, matching
the Soup arms' `train_on_responses_only: true`.

Instrumentation: per-step wall time via a TrainerCallback plus torch.cuda
allocator peaks. Median step time is reported, not mean, so a single
compile/warmup step cannot dominate.
"""
import argparse
import json
import os
import statistics
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import unsloth  # noqa: F401  MUST precede trl/transformers/peft


import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers.trainer_callback import TrainerCallback  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--arm", default="unsloth_native")
    args = ap.parse_args()

    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import FastLanguageModel

    torch.manual_seed(args.seed)

    raw = load_dataset("json", data_files={"train": args.data}, split="train")
    print(f"[{args.arm}] rows={len(raw)}", flush=True)

    t0 = time.perf_counter()
    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.seq,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    load_s = time.perf_counter() - t0
    # Count with the already-loaded tokenizer. Counting earlier meant one
    # from_pretrained per row, which stalls on this VM's flaky HF link.
    n_tok = sum(len(tok(r["text"], add_special_tokens=False)["input_ids"]) for r in raw)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[{args.arm}] trainable={trainable} total={total}", flush=True)

    torch.cuda.reset_peak_memory_stats()

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

    warmup = max(1, int(round(0.03 * args.steps)))
    timings = Timings()
    import unsloth_zoo.dataset_num_proc as _dnp
    if hasattr(_dnp, "DATASET_NUM_PROC"):
        _dnp.DATASET_NUM_PROC = 1

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=raw,
        dataset_text_field="text",
        max_length=args.seq,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.bs,
            gradient_accumulation_steps=args.accum,
            max_steps=args.steps,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=warmup,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            seed=args.seed,
            data_seed=args.seed,
            bf16=False,
            fp16=True,
            optim="adamw_torch",
            gradient_checkpointing=False,
            remove_unused_columns=False,
            logging_first_step=True,
            # This box has 4 cores. unsloth's default num_proc=min(8, cpu-1)
            # oversubscribes and the dataset .map stalls in uninterruptible
            # worker spawn. Pin to 1; it only affects preprocessing, and the
            # soup arms tokenize on the main thread too.
            dataloader_num_workers=0,
        ),
        callbacks=[timings],
    )

    torch.cuda.synchronize()
    t_start = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t_start
    if timings._t is not None:
        torch.cuda.synchronize()
        timings.steps.append(time.perf_counter() - timings._t)

    result = {
        "arm": args.arm,
        "steps_requested": args.steps,
        "steps_measured": len(timings.steps),
        "wall_seconds": round(wall, 3),
        "load_seconds": round(load_s, 3),
        "median_step_s": round(statistics.median(timings.steps), 4) if timings.steps else None,
        "mean_step_s": round(statistics.mean(timings.steps), 4) if timings.steps else None,
        "min_step_s": round(min(timings.steps), 4) if timings.steps else None,
        "max_step_s": round(max(timings.steps), 4) if timings.steps else None,
        "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
        "trainable_params": trainable,
        "total_params": total,
        "text_tokens": n_tok,
        "final_loss": timings.losses[-1] if timings.losses else None,
        "losses": timings.losses,
        "config": vars(args),
        "versions": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "losses"}, indent=2), flush=True)


if __name__ == "__main__":
    main()