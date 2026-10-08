---
name: finetune-llm
description: "Use when fine-tuning or post-training a language model (LoRA, QLoRA, SFT, DPO, full fine-tune) or automating a fine-tuning pipeline - choosing an engine (Soup, Unsloth, TRL) for the available GPU, writing or debugging training configs (batch size, quantization, LoRA rank, sequence length), hitting CUDA out-of-memory, runs that finish suspiciously fast or produce zero-norm adapters, deciding whether a GPU can fit a model, training on a T4/V100/Colab/Kaggle box, or comparing training frameworks on measured VRAM, speed, and quality."
metadata:
  author: instax-dutta
  version: "1.0.0"
---

# Hardware-Aware LLM Fine-Tuning

Fine-tune LLMs on whatever hardware is present, without OOM, silent no-ops, or
unverifiable "it trained" claims. Every number in this skill is measured; the
source is the controlled 4-arm benchmark in `docs/benchmark-2x-t4.md` (Soup vs
Unsloth, 2x Tesla T4).

**Core principle:** probe hardware -> pick engine -> pin data -> smoke test ->
train instrumented -> verify with a neutral evaluator -> record provenance.
No phase is skipped because "it's just a small run".

All paths below are relative to this skill's directory.

## Workflow

### 0. Probe the hardware before writing anything

```bash
python scripts/probe_hardware.py --model <hf-model-id> --out hardware.json
```

Reports GPUs, VRAM, compute capability, **true** bf16 support, cores, RAM, and
fits the model against the measured engine budgets below. Never trust
`torch.cuda.is_bf16_supported()` alone: on T4 (sm_75) it returns `True` via
emulation. The probe calls it with `including_emulation=False`.

### 1. Pick the engine from the VRAM budget

Measured peaks, 8B model, NF4 4-bit QLoRA, r=16, seq 1024, batch 1 (T4):

| Engine | Peak VRAM | Why |
|---|---|---|
| `backend: unsloth` (Soup or native Unsloth) | **7.1 GB** | Fused LM head + cross-entropy; fp16 on sm<80 |
| Stock `transformers` / TRL | **14.9 GB** | Materializes fp32 logits (~1 GB per copy) |
| `backend: transformers` while Unsloth is installed | silently Unsloth-backed | Soup imports Unsloth to print a tip; the import patches the whole stack |

Planning formula calibrated on that run: peak ~= 0.9 GB x params(billions) for
the Unsloth engine, ~= 1.85 GB x params(B) for stock TRL. Require planned peak
<= 90% of VRAM. Details and caveats: `references/hardware-and-engines.md`.

Decision:
- **NVIDIA sm_80+** (A100, L40S, 4090, H100): bf16, Unsloth engine if available.
- **NVIDIA sm_75/sm_70** (T4, V100): fp16 only (no bf16 hardware); Unsloth
  engine when VRAM-bound, it is the difference between fitting 8B on 15 GB and not.
- **Apple Silicon**: `backend: mlx` (Soup) or mlx-lm; no CUDA math applies.
- **No GPU**: rent one. CPU `stream_layers` exists in Soup but is for tiny
  models; do not promise training throughput without a GPU.

### 2. Pin the data

One canonical file, hashed, with token counts recorded. Identical bytes and
order across every arm of a comparison. Pre-render/pre-tokenize when formats
differ between frameworks; never let each trainer re-render.

```bash
python scripts/prep_sft_data.py --in raw.jsonl --out-dir data \
  --model <hf-model-id> --max-len 1024 --val-split 0.05 --seed 3407
```

Emits `train.jsonl`, `val.jsonl`, and `manifest.json` (sha256, rows, token
counts). Commit the manifest with the results.

### 3. Write the config from the probe, not from memory

Start from `assets/soup_qlora.yaml` (Soup), `assets/unsloth_sft.py`
(Unsloth-native), or `assets/trl_sft.py` (plain TRL). Component values that
must come from the probe: dtype (fp16 vs bf16), quantization, batch x
accumulation, sequence length. Step math: `steps = rows x epochs / (batch x accumulation)`.
Soup has **no `max_steps`** and integer epochs only - control steps by capping
rows, or use Unsloth/TRL when an exact step count matters.

Full knob table and failure-prone fields: `references/config-cookbook.md`.

### 4. Smoke test - mandatory gate

Run ~20-50 steps (or 5 minutes) with the real config before any long run. Cap
rows to hit the step budget (Soup has no `max_steps`):

```bash
head -64 data/train.jsonl > data/smoke.jsonl        # 8 steps at batch 1 x accum 8
# point a copy of the config at data/smoke.jsonl, then:
soup train --config soup_smoke.yaml --yes 2>&1 | tee logs/smoke.log
python scripts/verify_run.py --run-dir out/smoke --log logs/smoke.log
```

Pass requires: loss falling from step 1, peak VRAM within plan, adapter weights
nonzero. `verify_run.py` exits 2 on failure. It is cheap and catches OOM,
silent no-op training, data bugs, and masking flags that did nothing.

### 5. Train with instrumentation

```bash
./scripts/sample_gpu.sh 0 logs/run_gpu.csv &
PYTORCH_ALLOC_CONF=expandable_segments:True \
  soup train --config soup.yaml --yes 2>&1 | tee logs/run.log
```

One heavy training job per GPU. Do not co-locate two arms on one GPU unless the
planned peaks plus headroom fit - throughput collapses when they do not.

### 6. Verify with a neutral evaluator - mandatory gate

```bash
python scripts/eval_adapter.py --model <hf-model-id> --val-data data/val.jsonl \
  --adapter myrun=out/myrun --out eval.json
```

The evaluator imports neither Soup nor Unsloth, so no trainer gets an
advantage. It computes supervised-token held-out loss plus adapter health
(nonzero tensors, parameter count, Frobenius norm). Claims of quality
differences require this number; training loss is not evidence.

### 7. Record provenance

Write into the repo when the run finishes: `hardware.json`, `pip freeze`
output, `manifest.json`, final training loss, held-out loss, peak VRAM, wall
time. Numbers on an ephemeral box are one rebuild from lost.

## Quick reference

| VRAM (8B, seq 1024) | Engine | Setup |
|---|---|---|
| 8 GB | Unsloth, NF4, batch 1 | gradient checkpointing on; tight fit |
| 12-16 GB | Unsloth, NF4, batch 1 x accum 8 | the benchmark config (7.1 GB peak); stock TRL needs ~15 GB |
| 24 GB | Unsloth or TRL, NF4 | batch 2 x accum 4 |
| 48 GB+ | bf16 LoRA unquantized | batch 4 x accum 2, no QLoRA |
| Apple Silicon | mlx | probe-dependent |

LoRA defaults that worked: `r=16, alpha=32, dropout=0`, targets
`[q,k,v,o,gate,up,down]_proj`, LR `2e-4` cosine, warmup 3%, AdamW, seed fixed.

## Red flags - stop and investigate

- Log says `train_on_responses_only` / `return_assistant_tokens_mask` but chat
  template has no `{% generation %}` -> the flag silently did nothing.
- `backend: transformers` with Unsloth installed -> the run is not stock HF.
- `is_bf16_supported()` true on sm_70/sm_75 -> emulation; force fp16.
- Training loss flat, or adapter tensors all zero -> data/labels/masking bug,
  not a hyperparameter problem.
- `NameError: auto_docstring`, `TypeError: can only concatenate str` ->
  known dependency/API conflicts; see `references/failure-modes.md`.
- A run "finished" in seconds -> 0 steps; check step math and dataset size.

## References

- `references/hardware-and-engines.md` - measured budgets, engine matrix, install pins
- `references/config-cookbook.md` - Soup / Unsloth / TRL configs and knobs
- `references/failure-modes.md` - each defect: symptom, detection, fix
- `docs/benchmark-2x-t4.md` - the controlled benchmark behind every number
