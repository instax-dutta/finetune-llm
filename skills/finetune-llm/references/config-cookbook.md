# Config Cookbook

Exact fields and values that passed config validation and a real 8B QLoRA run on
Soup 0.75.x (transformers 5.17.0 / trl 0.29.1). When a field is version
sensitive, the symptom of getting it wrong is noted.

## Soup baseline (validated)

```yaml
base: NousResearch/Meta-Llama-3.1-8B-Instruct
task: sft
backend: unsloth                # transformers | unsloth | mlx

data:
  train: ./data/train.jsonl
  format: auto                  # auto | alpaca | chatml | sharegpt | pre_tokenized | ...
  max_length: 1024
  val_split: 0.05               # 0.1 default; 0 = train on everything
  train_on_responses_only: false

training:
  epochs: 1                     # integer only; there is no max_steps in Soup
  batch_size: 1                 # or "auto"
  auto_batch_size_strategy: static
  gradient_accumulation_steps: 8
  lr: 2.0e-4
  scheduler: cosine
  warmup_ratio: 0.03            # Soup converts it to warmup_steps internally
  lora:
    r: 16
    alpha: 32
    dropout: 0.0
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  quantization: 4bit            # NF4 default; 8bit | none also valid
  optimizer: adamw_torch
  gradient_checkpointing: false
  logging_steps: 1
  save_steps: 100
  data_seed: 3407

output: ./out/runname
```

Run non-interactively with `soup train --config soup.yaml --yes`.

## Values that must come from the hardware probe

| Probe finding | Config consequence |
|---|---|
| sm_75/sm_70 GPU | fp16 only; leave bf16 off (Soup/TRL pick fp16 when bf16 unsupported) |
| sm_80+ | bf16 |
| VRAM < 0.9 x estimated peak at batch 1 | `gradient_checkpointing: true`, `quantization: 4bit`, batch 1 |
| Even batch 1 does not fit | `training.stream_layers: true` (frozen base streams from RAM) |
| >= 48 GB | `quantization: none`, bf16 LoRA, batch 4+ |
| Apple Silicon | `backend: mlx` |
| Many cores (>= 16) | leave `dataloader_num_workers`/`num_proc` defaults |
| Few cores (2-4, Colab/Kaggle) | pin dataset workers to 1 (see failure modes) |

## Steps, epochs, and dataset size

`steps = rows x epochs / (batch_size x gradient_accumulation_steps)`.

Soup has no `max_steps` and refuses fractional epochs. To hit an exact step
budget (e.g. for a fair A/B of two engines): cap rows. 400 rows at batch 1 x
accum 8 = exactly 50 steps. Unsloth/TRL accept `max_steps` directly; use it
there. A run that "finishes" in seconds means the step math collapsed - check
rows and accumulation.

## Data section notes

- `format: auto` detects from the first 100 rows; a file mixing shapes is
  refused. Prefer explicit formats for reproducibility.
- `val_split` holds rows out of training and evaluates at epoch end (or every
  `training.eval_steps`). With `val_split: 0` the dataset must bring its own
  validation split.
- `train_on_responses_only: true` needs a chat template containing the
  `{% generation %}` keyword. Llama-3.1's stock template does not have it -
  the flag logs a warning and silently trains on prompt tokens too. Verify in
  the log before trusting it, or use pre-rendered prompt/completion data.
- Cross-engine comparisons: emit one file both frameworks consume
  (`{"messages": [...]}` for Soup, `{"text": "<rendered>"}` for Unsloth, or the
  same `prompt_completion` rows). Identical bytes, identical order.

## LoRA settings

| Setting | Default used | When to change |
|---|---|---|
| `r` | 16 | 32-64 for more capacity (needs more VRAM for adapter grads) |
| `alpha` | 32 | keep `alpha = 2r` |
| `dropout` | 0.0 | 0.05-0.1 for small datasets or overfitting |
| targets | q,k,v,o,gate,up,down | all-linear; fewer modules = less capacity |
| LR | 2e-4 LoRA | 1e-5 to 5e-5 for full fine-tune |
| scheduler | cosine | linear is a reasonable alternative |
| warmup | 3% | raise for unstable loss in the first steps |

## Unsloth-native script anatomy

See `assets/unsloth_sft.py`. Key points:

- `import unsloth` must come **before** trl/transformers/peft.
- `load_in_4bit=True`, `dtype=None` (Unsloth resolves fp16/bf16).
- `use_gradient_checkpointing="unsloth"`.
- `dataset_text_field="text"` - do not use a `formatting_func` returning a list
  in 2026.9.14 (internal bug, see failure modes). Pass prompt/completion
  columns instead when masking matters.
- On low-core boxes pin dataset preprocessing to 1 worker.
- `max_steps` works here; use it for exact-step runs and fair comparisons.

## TRL deltas (transformers 5.x)

- `warmup_ratio` no longer exists on `TrainingArguments`/`SFTConfig`; use
  `warmup_steps`. Soup still accepts `warmup_ratio` and converts it.
- Assistant-only loss requires the `{% generation %}`-tagged template; without
  it the mask silently no-ops.
- Stock TRL materializes fp32 logits; budget ~2x the Unsloth peak, or enable a
  fused cross-entropy/Liger if available for your GPU.
- A plain TRL baseline is `assets/trl_sft.py`; it is deliberately minimal.

## Config mistakes that cost a run

| Mistake | Symptom | Fix |
|---|---|---|
| bf16 on T4 | `torch.cuda.is_bf16_supported()` True, then NaN or emulated slowness | force fp16 |
| steps controlled by epochs only | cannot hit a target step budget | cap rows or move to max_steps |
| `train_on_responses_only: true` on Llama template | trains on prompt tokens | pre-render prompt/completion, or accept full-sequence and stay consistent |
| two runs on one GPU | both OOM or throughput halves | one heavy job per GPU |
| stock TRL peak assumed Unsloth peak | OOM at ~15 GB on a 16 GB card | use 1.85 x params(B) budget |
| no `expandable_segments` | fragmentation OOM despite "enough" free memory | `PYTORCH_ALLOC_CONF=expandable_segments:True` |
