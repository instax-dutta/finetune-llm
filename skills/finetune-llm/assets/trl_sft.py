#!/usr/bin/env python3
"""Plain-TRL QLoRA SFT template - the engine-neutral baseline.

Use when Unsloth is not available. Budget ~2x the Unsloth peak VRAM: stock TRL
materializes fp32 logits unless a fused cross-entropy is enabled.

Notes that bite in transformers 5.x / trl 0.29:
  - warmup_ratio no longer exists; pass warmup_steps
  - assistant-only masking needs a {% generation %} chat template; without it the
    mask silently no-ops, so this template supervises the full sequence
  - unquantized 8B + activations does not fit a 16 GB card; keep load_in_4bit
"""
import argparse
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="jsonl with a 'text' field")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    torch.manual_seed(args.seed)
    raw = load_dataset("json", data_files={"train": args.data}, split="train")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    fp16 = True
    if torch.cuda.is_available():
        try:
            fp16 = not (torch.cuda.is_bf16_supported(including_emulation=False)
                        and torch.cuda.get_device_properties(0).major >= 8)
        except TypeError:
            fp16 = True

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16 if fp16 else torch.bfloat16,
            bnb_4bit_use_double_quant=True),
        device_map="auto",
    )
    model.config.use_cache = False
    model = __import__("peft").prepare_model_for_kbit_training(model)
    model = __import__("peft").get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))

    total_steps = args.steps or int(len(raw) * args.epochs / (args.bs * args.accum))
    cfg = SFTConfig(
        output_dir=args.out,
        dataset_text_field="text",
        max_length=args.seq,
        packing=False,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        max_steps=args.steps if args.steps else -1,
        num_train_epochs=args.epochs if not args.steps else 1.0,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(round(0.03 * total_steps))),
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        bf16=not fp16,
        fp16=fp16,
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    )
    trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=raw, args=cfg)
    trainer.train()
    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"saved adapter to {args.out}")


if __name__ == "__main__":
    main()
