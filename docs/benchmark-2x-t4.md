# Soup vs Unsloth: a controlled head-to-head on 2x Tesla T4

> Evidence document for the `finetune-llm` skill. Every budget, red flag, and
> failure fix in the skill traces back to this experiment. Originally produced
> as a standalone benchmark report. The harness and raw results are in
> [benchmark/](benchmark/).

**Date:** 2026-10-02
**Hardware:** 2x Tesla T4 (sm_75, Turing, 15.6 GB each), 31 GB RAM, 4 vCPU, driver 580.178.04
**Model:** `NousResearch/Meta-Llama-3.1-8B-Instruct` (rev `d10aef79`), NF4 4-bit QLoRA, fp16
**Data:** `yahma/alpaca-cleaned`, seed 3407, 400 rows × chat template = 73,657 tokens
**Stack:** torch 2.10.0+cu128, transformers 5.17.0, trl 0.29.1, peft 0.21.2, bitsandbytes 0.50.2, unsloth 2026.9.14, soup-cli 0.75.0

---

## Verdict

| Question | Winner | Margin |
|---|---|---|
| GPU compute cost (VRAM) | Unsloth, clearly | 7.1 GB vs 14.9 GB, about 2.09x |
| Finetune quality | Tie | held-out loss 1.0373 vs 1.0405 (0.3% apart, both adapters valid) |
| Ease of use | Soup, clearly | 1 config file, zero code; Unsloth needed a custom script and hit 4 separate API/library failures |

Neither is better at fine-tuning quality. They produce statistically
indistinguishable adapters. The real difference is that Unsloth uses half the
VRAM, and Soup is dramatically easier to drive.

And the headline finding: Soup's speed and memory come almost entirely from
Unsloth. `soup` with `backend: unsloth` used 7,109 MiB; Unsloth used 7,111 MiB.
That is a 2 MiB difference. Whatever "2-5x faster" means, it is Unsloth's
number, not Soup's.

---

## Results

All four arms: same model, same 400 rows, same LoRA (r=16, alpha=32, 7 target
modules, 41,943,040 trainable params in every arm), same optimizer, LR,
schedule, seed, batch 1 × accum 8 × 50 steps.

| Arm | Result | Peak VRAM | Final loss |
|---|---|---|---|
| `unsloth_native` | completed | 7,111 MiB | 0.8290 |
| `soup` `backend: unsloth` | completed | 7,109 MiB | 0.8044 |
| `soup` `backend: transformers` (unsloth installed) | completed | 14,867 MiB | 0.8058 |
| `soup` `backend: transformers` (unsloth blocked) | OOM | 14,817 MiB (died) | - |

### Quality: a tie, verified independently

Held-out loss (200 rows, 26,897 supervised tokens) computed by both adapters
through one neutral loader, with no Soup and no Unsloth in the eval path, so
neither trainer gets an advantage:

| Adapter | Held-out loss |
|---|---|
| unsloth | 1.03731 |
| soup transformers | 1.04050 |

Adapter health, both trained, neither silently no-op'd:

| Adapter | Tensors | Non-zero | Params | Frobenius norm |
|---|---|---|---|---|
| unsloth | 448 | 448 | 41,943,040 | 588.81 |
| soup transformers | 448 | 448 | 41,943,040 | 588.23 |

Identical parameter counts, all tensors non-zero, norms within 0.1%. Training
loss curves overlay almost exactly (1.479 to 0.806 vs 1.470 to 0.804). The two
frameworks are interchangeable on quality.

### Compute cost: Unsloth wins by 2.09x

The gap is entirely about where the memory goes. Unsloth fuses the LM head and
cross-entropy so the `[1, 1024, 128256]` logits are never materialized. Stock
TRL materializes them in fp32, about 1 GB per copy. Soup's transformers path
has no equivalent optimization enabled by default.

Measured FLOPs confirm both arms did the same work, so the memory gap is not
one arm doing less:

| Arm | Total FLOPs |
|---|---|
| soup transformers | 3.3353e15 |
| soup `backend: unsloth` | 3.2628e15 |

Within 2.2%: the same computation, a different memory plan.

---

## Two defects found in the process

### 1. `backend: transformers` is silently unsloth-backed

This is the most consequential finding. `src/soup_cli/commands/train.py:1517`
calls `is_unsloth_available()` even when the config says `backend:
transformers`, purely to print a "try unsloth" tip. That function does
`import unsloth` (`utils/unsloth.py:9`), and Unsloth patches the entire
HF/TRL stack at import time.

Consequence: you cannot get a stock-HF Soup run on a machine where Unsloth is
installed. Verified directly. With the import blocked,
`is_unsloth_available()` returns `False`; without the block, `True`.

The proof is in the OOM traceback. Soup's transformers arm reaches
`transformers/trainer.py:2088` → `accelerate/utils/operations.py:811`, a clean
stock HF path, because we blocked the import. Earlier, unblocked runs showed
`unsloth/trainer.py` → `UnslothSFTTrainer.py` frames while the config said
`backend: transformers`.

This makes the README's comparison table misleading: the "transformers" column
on an Unsloth-equipped machine is not the transformers backend.

### 2. `train_on_responses_only: true` silently did nothing

Soup logged:

```
[transformers] return_assistant_tokens_mask==True but chat template
does not contain `{% generation %}` keyword.
```

It is a warning, not an error. The run proceeds and trains on prompt tokens
too. We set this flag on both arms and got assistant-only masking on neither.
The quality comparison stayed fair, but the flag did not do what it said.

### Also worth knowing

- Dependency conflict on install. `pip install "soup-cli[train,fast]"`
  produced a broken environment: it resolved unsloth 2025.9.5 against
  transformers 5.17.0, and `import unsloth` failed with
  `NameError: name 'auto_docstring' is not defined`. Forcing unsloth 2026.9.14
  downgrades transformers to at most 5.5.0 and trl to at most 0.24.0, below
  Soup's own declared floors (`transformers>=5.16.1`, `trl>=0.29.0`). We ran on
  transformers 5.17.0 + trl 0.29.1 where Unsloth imports and works, outside its
  declared range.
- `TrainingArguments` lost `warmup_ratio`. Gone in transformers 5.x from both
  `TrainingArguments` and trl's `SFTConfig`. Soup still exposes
  `warmup_ratio` and converts it to `warmup_steps` internally.
- Soup has no `max_steps` and requires integer `epochs`. Step count is only
  controllable via dataset size. We capped the dataset to 400 rows to hit 50
  steps.
- Unsloth's `formatting_func` is internally inconsistent in 2026.9.14. Its
  validator requires a list (`UnslothSFTTrainer.py:1331`) but its `.map`
  concatenates the result as a `str`, so a correct list raises
  `TypeError: can only concatenate str (not "list") to str`.
- No custom kernels in Soup. We grepped `src/` for Triton/CUDA kernels and
  found none. Its documented speed features (Liger, cut-CE, FlashAttention) are
  all off by default and self-measured small in its own docstrings: Liger
  "5.1% throughput / 12.9% memory" on H100, FlashAttention "1.015x, no memory
  saving."

---

## Throughput

Measured per-step for the Unsloth arm only. Soup's timing artifacts do not
record per-step wall time, and its `bench train` command explicitly refuses
to measure an unsloth-backend run (`bench/train_run.py:37`), so a comparable
per-step comparison across all arms was not obtainable without patching Soup.

| Arm | Median step | Note |
|---|---|---|
| `unsloth_native` | 8.65 s | precise, TrainerCallback |
| `soup` arms | about 9-10 s (upper bound) | whole-process span ÷ 50, includes about 36 s model load |

The upper bound includes model load and tokenization, so it overstates the
training time. We are not claiming Soup is slower per step. We could not
measure it cleanly, and the VRAM result already settles the compute-cost
question. Recorded as a gap rather than guessed.

---

## Honest limitations

1. T4 is not the hardware either project targets. sm_75 has no bf16 units
   and no tensor cores for bf16; both frameworks ran fp16. Unsloth's published
   2-5x is measured on Ampere+. Our numbers do not transfer to A100/H100, and
   neither do Soup's published H100 figures to T4.
2. One seed, 50 steps, 400 rows. Enough to separate 2x memory from noise.
   Not enough to separate a 0.3% quality difference. Treat quality as a tie,
   not a ranking.
3. `soup_hf_plain` was not truly stock HF either (see defect 1). Only
   `soup_hf_pure` (import blocked) is a genuine stock-HF arm, and it OOM'd.
4. No `backend: unsloth` adapter held-out eval. We evaluated the unsloth and
   soup-transformers adapters. The soup-unsloth adapter had the same training
   loss (0.8044) but did not go through the neutral evaluator.
5. Per-step timing is single-arm. See above.

---

## Recommendation

Use Unsloth as the engine and Soup as the interface. If you can only pick one,
pick based on whether you write code.

- Pick Unsloth if VRAM is your binding constraint (2x is the difference
  between fitting 8B on a 15 GB card and not fitting it), or if you already have
  a training script. It is the engine; everything fast about Soup's numbers is
  Unsloth's.
- Pick Soup if you want a config file instead of a script, one-command
  reproducibility, config validation that catches typos (it correctly flagged 3
  of our bad guesses with suggestions), built-in eval/merge/export/GGUF, and a
  Web UI. Those are real and genuinely better.
- Best of both today: `soup train --config soup.yaml` with
  `backend: unsloth` in it. That measured at Unsloth's 7.1 GB. Soup's ergonomics,
  Unsloth's engine. The caveat is the install conflict above.

Do not choose on quality. There is no measurable difference.

---

## Reproducing

```
# data (identical for all arms)
python prep_data.py        # 4000 train / 200 val, seed 3407
python prep_pc.py          # chat-template render + token ids
python prep_shared.py      # chatml.jsonl (soup) + text.jsonl (unsloth)
python cap_rows.py         # 400 rows => 50 steps at bs1 x accum8

# arms, one per GPU
PYTHONPATH=./no_unsloth soup train --config soup_cfg_hf_pure.yaml --yes
CUDA_VISIBLE_DEVICES=1 python run_unsloth.py --model <model> \
  --data data/text_50.jsonl --out out/unsloth_native.json --steps 50

# neutral eval of both adapters
python eval_adapter.py
```

`no_unsloth/sitecustomize.py` blocks `import unsloth` for the stock-HF arm.
Without it you are not measuring the transformers backend.
