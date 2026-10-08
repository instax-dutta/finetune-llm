# Failure Modes

Defects found and reproduced during the benchmark. Each has: symptom, why it
happens, how to detect it, how to fix it.

## 1. `backend: transformers` is silently Unsloth-backed

**Symptom:** a Soup "transformers" run uses half the VRAM you budgeted; tracebacks
show `unsloth/trainer.py` frames while the config says `backend: transformers`.

**Why:** `soup_cli/commands/train.py` calls `is_unsloth_available()` even for the
transformers backend, only to print a "try Unsloth" tip. That function does
`import unsloth`, and Unsloth patches the entire HF/TRL stack at import time.
On any machine where Unsloth is installed, you cannot get a stock-HF Soup run.

**Detect:** grep the run log for `unsloth/` in tracebacks; or check importability.

**Fix:** if you want genuine stock HF, block the import for the whole process:

```python
# no_unsloth/sitecustomize.py on PYTHONPATH
import sys
class _BlockUnsloth:
    def find_spec(self, name, path=None, target=None):
        if name == "unsloth" or name.startswith("unsloth."):
            raise ImportError("unsloth blocked on purpose")
        return None
sys.meta_path.insert(0, _BlockUnsloth())
```

```bash
PYTHONPATH=./no_unsloth soup train --config soup.yaml --yes
```

If you want Unsloth, say `backend: unsloth` and get it deliberately.

## 2. `train_on_responses_only: true` silently does nothing

**Symptom:** the flag is set, but the model trains on prompt tokens.

**Why:** assistant-only masking requires the chat template to contain the
`{% generation %}` keyword (system/user content tagged as non-generation). The
stock Llama-3.1 template does not. The trainer logs:

```
return_assistant_tokens_mask==True but chat template does not contain `{% generation %}` keyword.
```

It is a warning, not an error. The run proceeds full-sequence.

**Detect:** search the log for `{% generation %}`.

**Fix:** pre-render and pre-tokenize prompt/completion rows (supervise only
completion ids), or add a `{% generation %}` region to the template, or accept
full-sequence training and keep it consistent across every arm being compared.

## 3. Dependency conflicts on install

**Symptom A:** `pip install "soup-cli[train,fast]"` produces an env where
`import unsloth` fails: `NameError: name 'auto_docstring' is not defined`.
The resolver paired `unsloth==2025.9.5` with `transformers==5.17.0`.

**Symptom B:** installing `unsloth==2026.9.14` cleanly downgrades
`transformers<=5.5.0` and `trl<=0.24.0`, below Soup's declared floors
(`transformers>=5.16.1`, `trl>=0.29.0`).

**Fix:** install deliberately and pin the validated set from
`hardware-and-engines.md`; `pip install unsloth --no-deps` after the torch
stack, then smoke test the import before any long run. Do not let the resolver
improvise.

## 4. `warmup_ratio` is gone from transformers 5.x

**Symptom:** `TypeError`/unknown-field errors in raw TRL scripts; or silently no
warmup if you removed it.

**Why:** `TrainingArguments` (and trl's `SFTConfig`) dropped `warmup_ratio` in
transformers 5.x; only `warmup_steps` remains.

**Fix:** compute `warmup_steps = round(ratio * total_steps)` yourself. Soup still
exposes `warmup_ratio` and converts internally, so this only bites raw scripts.

## 5. No `max_steps` in Soup; integer epochs only

**Symptom:** cannot reproduce an exact step count, e.g. for a fair comparison.

**Fix:** `steps = rows / (batch x accum)`. Cap rows (400 rows at bs1 x accum8 =
50 steps) or run Unsloth/TRL, which accept `max_steps`.

## 6. Unsloth `formatting_func` is broken in 2026.9.14

**Symptom:** `TypeError: can only concatenate str (not "list") to str`.

**Why:** the validator requires the formatting function to return a list, but
the internal `.map` concatenates the result as a string.

**Fix:** avoid `formatting_func`; use `dataset_text_field` for plain text or
pre-tokenized prompt/completion columns for masked training.

## 7. bf16 "support" is emulated on T4

**Symptom:** `torch.cuda.is_bf16_supported()` returns `True` on sm_75, then
training is slow or unstable.

**Why:** the default check includes software emulation. T4 has no bf16 units.

**Fix:** `torch.cuda.is_bf16_supported(including_emulation=False)`; force fp16
on sm_70/sm_75. `scripts/probe_hardware.py` reports the truthful value.

## 8. Dataset preprocessing oversubscribes small boxes

**Symptom:** `.map` hangs in uninterruptible worker spawn on 2-4 core VMs;
training never starts.

**Why:** Unsloth defaults `num_proc=min(8, cpu_count-1)`; on a 4-core box that
oversubscribes.

**Fix:** `dataloader_num_workers=0`, and cap `unsloth_zoo.dataset_num_proc`
(`DATASET_NUM_PROC=1`) for the native script. Soup tokenizes on the main thread
already.

## 9. Measuring the run changes the run

**Symptom:** per-step time comparisons exist for one arm but not the other.

**Why:** Soup's `bench train` refuses to measure an unsloth-backend run
(`bench/train_run.py`), and whole-process timing includes ~36 s model load.

**Fix:** instrument inside the trainer - a `TrainerCallback` recording
`on_step_begin` timestamps and reporting the **median** step, so a warmup step
cannot dominate. Sample VRAM externally with `scripts/sample_gpu.sh` so the
same tool measures every arm.

## 10. Attribution: "Soup is fast" means "Unsloth is fast"

**Symptom:** comparing `soup backend: unsloth` against stock HF attributes the
speedup to Soup.

**Why:** Soup's Unsloth arm measured 7,109 MiB vs Unsloth-native 7,111 MiB - a
2 MiB difference. Soup has no custom kernels; Liger/FlashAttention are off by
default and self-measured small.

**Fix:** compare engine-to-engine (Unsloth vs TRL), not wrapper-to-wrapper.
Soup's real wins are ergonomics: one config file, validation, eval/merge/export.

## 11. Empty or zero adapters pass as success

**Symptom:** a run finishes, files exist, but the adapter is useless.

**Why:** trainers save on completion regardless of whether the adapter learned
anything; a silently no-op'd mask or a broken data path can produce near-zero
deltas while loss barely moves.

**Fix:** always run the gates: `scripts/verify_run.py` (adapter tensors nonzero,
loss trajectory sane) and `scripts/eval_adapter.py` (held-out loss via a
neutral loader). Never report success from "it exited 0".
