# finetune-llm

[![skills.sh](https://skills.sh/b/instax-dutta/finetune-llm)](https://skills.sh/instax-dutta/finetune-llm)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A hardware-aware, verification-gated LLM fine-tuning skill for coding agents.
It teaches an agent to probe the GPU before writing configs, pick the right
engine (Soup / Unsloth / TRL / MLX), prepare data deterministically, smoke
test, train with instrumentation, and verify the result with a neutral
evaluator - so an automated fine-tune never OOMs blindly, silently no-ops, or
reports "it trained" without evidence.

## Install

```bash
npx skills add instax-dutta/finetune-llm
```

Works with any agent the skills CLI supports (Claude Code, OpenCode, Codex,
Cursor, Windsurf, Copilot, and more). Or install manually by copying
`skills/finetune-llm/` into your agent's skills directory.

## What the skill enforces

| Phase | Gate |
|---|---|
| Probe | `scripts/probe_hardware.py` - VRAM, compute capability, true bf16 support, model fit vs measured engine budgets |
| Engine | pick from measured peaks: Unsloth ~0.9 GB/param(B), stock TRL ~1.85 GB/param(B) |
| Data | `scripts/prep_sft_data.py` - deterministic split + sha256 + token counts in `manifest.json` |
| Config | templates in `assets/` + a cookbook of version-sensitive fields |
| Smoke | `scripts/verify_run.py` - loss must fall, adapter must be nonzero |
| Train | `scripts/sample_gpu.sh` - identical VRAM sampling for every arm |
| Verify | `scripts/eval_adapter.py` - held-out loss via a loader that imports neither Soup nor Unsloth |
| Provenance | hardware, versions, hashes, losses, and peak VRAM recorded in the repo |

## Why you can trust the numbers

Every budget in the skill comes from a controlled 4-arm benchmark
([docs/benchmark-2x-t4.md](docs/benchmark-2x-t4.md)): 2x Tesla T4, Llama-3.1-8B,
NF4 QLoRA, identical data/seed/steps across Soup-unsloth, Unsloth-native,
Soup-transformers, and truly-stock HF (unsloth import blocked).

Headline results:

| Arm | Peak VRAM | Held-out loss |
|---|---|---|
| Unsloth native | 7,111 MiB | 1.0373 |
| Soup `backend: unsloth` | 7,109 MiB | - |
| Soup `backend: transformers` | 14,867 MiB | 1.0405 |
| Stock HF (unsloth blocked) | OOM at 14.8 GB | - |

- Quality: statistically a tie (0.3% apart, identical parameter counts).
- Cost: the Unsloth engine halves VRAM; Soup's speed *is* Unsloth's.
- Defects documented with detection and fixes: silently unsloth-backed
  `backend: transformers`, silently no-op'd `train_on_responses_only`,
  emulated bf16 reporting on T4, dependency conflicts, and more - see
  `skills/finetune-llm/references/failure-modes.md`.

## Repository layout

```
skills/finetune-llm/
  SKILL.md                      # the workflow
  references/
    hardware-and-engines.md     # budgets, engine matrix, install pins
    config-cookbook.md          # Soup / Unsloth / TRL configs and knobs
    failure-modes.md            # each defect: symptom, detection, fix
  scripts/
    probe_hardware.py           # hardware + model-fit probe
    prep_sft_data.py            # deterministic data prep with provenance
    verify_run.py               # post-run gate (adapter health + loss)
    eval_adapter.py             # neutral held-out evaluator
    sample_gpu.sh               # VRAM/utilization sampler
  assets/
    soup_qlora.yaml             # Soup QLoRA template
    unsloth_sft.py              # Unsloth-native template
    trl_sft.py                  # plain-TRL template
docs/benchmark-2x-t4.md         # the evidence
```

## License

MIT
