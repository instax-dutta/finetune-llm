# Hardware and Engines

All numbers below are measured, not vendor claims. Source: `docs/benchmark-2x-t4.md`
(2x Tesla T4 15.6 GB, 31 GB RAM, 4 vCPU, Llama-3.1-8B-Instruct, NF4 QLoRA
r=16 alpha=32 all-linear, seq 1024, batch 1 x accum 8, 50 steps, seed 3407).

## Measured peaks (8B, NF4 QLoRA, seq 1024, batch 1)

| Arm | Peak VRAM | Final loss | Notes |
|---|---|---|---|
| Unsloth native | 7,111 MiB | 0.8290 | fused LM head + CE |
| Soup `backend: unsloth` | 7,109 MiB | 0.8044 | 2 MiB apart from native = same engine |
| Soup `backend: transformers`, Unsloth importable | 14,867 MiB | 0.8058 | silently Unsloth-patched (see failure modes) |
| Soup `backend: transformers`, Unsloth blocked | OOM at 14,817 MiB | - | genuine stock HF; fp32 logits |

FLOPs were within 2.2% across arms (3.34e15 vs 3.26e15), so the memory gap is a
memory plan, not less work. Another reason stock TRL is heavy: it materializes
the `[1, 1024, 128256]` fp32 logits (~1 GB per copy) unless you enable a fused
cross-entropy.

Quality was a tie: held-out loss 1.0373 vs 1.0405 (0.3%) via one neutral
loader; identical 41,943,040 trainable params, all 448 tensors nonzero, norms
within 0.1%. **Do not pick engines on quality. Pick on VRAM and ergonomics.**

## Planning formula

Calibrated from the table above, $P$ = model parameters in billions:

```
peak_unsloth_GB ~= 0.90 * P     # 8B -> 7.2
peak_trl_GB     ~= 1.85 * P     # 8B -> 14.8
```

Assumptions: NF4 4-bit, LoRA r=16, seq 1024, batch 1, no full fine-tune. Treat
as lower bounds and require `peak <= 0.9 * VRAM`:

- Longer sequences raise peaks (~0.5-1 GB per extra 1K tokens at 8B).
- `r` and target modules move the adapter+optimizer term, small at 8B.
- Full fine-tune (`lora.r: 0`) changes everything: weights + grads + AdamW
  states ~= 16 bytes/param before activations. 8B does not fit on 80 GB alone.
- CPU RAM matters too: `stream_layers: true` streams the frozen base from RAM
  (NVMe overflow), bounding VRAM to about one decoder layer.

## Engine decision matrix

| Condition | Engine | dtype |
|---|---|---|
| NVIDIA sm_80+ and Unsloth installed | `backend: unsloth` | bf16 |
| NVIDIA sm_80+ no Unsloth | `backend: transformers` | bf16, consider Liger/cut-CE |
| NVIDIA sm_75 (T4) / sm_70 (V100) | `backend: unsloth` when VRAM-bound | **fp16** (no bf16 units) |
| Apple Silicon | `backend: mlx` | fp16/bf16 per mlx-lm |
| No GPU | rent a GPU | - |

T4 specifics: sm_75 has no bf16 hardware and no bf16 tensor cores. fp16 it is.
Unsloth's published 2-5x speedup is measured on Ampere+; on T4 the win shows up
as VRAM (about 2x), which is the number that decides fit.

## Version pins

Validated together on the benchmark (torch 2.10.0+cu128, driver 580):

```
torch==2.10.0+cu128
transformers==5.17.0
trl==0.29.1
peft==0.21.2
bitsandbytes==0.50.2
accelerate==1.13.0
datasets==3.6.0
unsloth==2026.9.14
unsloth_zoo==2026.9.9
xformers==0.0.35
soup-cli==0.75.0
```

Known conflicts (details in `failure-modes.md`):

- `pip install "soup-cli[train,fast]"` can resolve `unsloth==2025.9.5` against
  `transformers==5.17.0`, then `import unsloth` dies with
  `NameError: name 'auto_docstring' is not defined`.
- `unsloth==2026.9.14` declares `transformers<=5.5.0` and `trl<=0.24.0`, below
  Soup's own floors (`transformers>=5.16.1`, `trl>=0.29.0`). The combination
  above runs and imports outside that declared range - pin it deliberately and
  smoke test, rather than letting the resolver improvise.

Order that works: install the torch stack, `pip install unsloth --no-deps`,
then `pip install soup-cli[train]` with pins, then verify with
`python scripts/probe_hardware.py` and a 20-step smoke run.

## Multi-GPU

- One arm per GPU is the only clean comparison; T4s without NVLink cannot use
  each other's VRAM and cross-GPU collectives are slow.
- A single training job across N GPUs: `accelerate`/DeepSpeed data parallel.
  Batch math changes: effective batch = `batch x accum x N`.
- Run one heavy job per box. Two concurrent jobs measurably drop throughput
  (46% in one measurement) and cause address-space/communication failures.

## Apple Silicon

`backend: mlx` in Soup (SFT only; DPO/GRPO refused at config load). Peaks bundle
unified memory with the OS, so leave more headroom than on discrete GPUs;
`mlx-lm >= 0.31.3`. The CUDA tuning rules (fused CE, fp16-vs-bf16 on T4) do not
apply.
