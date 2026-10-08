# Benchmark harness (2x T4, Soup vs Unsloth)

Original harness behind [../benchmark-2x-t4.md](../benchmark-2x-t4.md). All four
arms ran the same model, rows, rendering, LoRA, optimizer, LR, seed, and step
count (50) - one arm per T4.

## Arms

| Tag | Command essence | Config |
|---|---|---|
| `soup_hf_pure` | `PYTHONPATH=./no_unsloth soup train` | `soup_cfg_hf_pure.yaml` |
| `soup_hf_plain` | `soup train` compared against pure to prove silent patching | `soup_cfg_hf_plain.yaml` |
| `soup_unsloth` | `soup train`, `backend: unsloth` | `soup_cfg_unsloth.yaml` |
| `unsloth_native` | `python run_unsloth.py` | script args |

`no_unsloth/sitecustomize.py` blocks `import unsloth` so `soup_hf_pure` is a
genuine stock-HF arm; without it, `backend: transformers` is silently
Unsloth-backed (see failure mode 1 in the skill).

## Data chain

```
prep_data.py     # fixed 4000/200 split of yahma/alpaca-cleaned, seed 3407, sha256-stamped
prep_pc.py       # chat-template render + token ids, prompt/completion split
prep_shared.py   # chatml.jsonl (Soup) + text.jsonl (Unsloth), byte-identical renders
cap_rows.py      # 400 rows: 400 / (1 x 8) = 50 steps
```

## Run

```
bash run_matrix.sh          # both passes, one arm per GPU, GPU sampled per arm
python eval_adapter.py      # neutral held-out loss + adapter health (original version)
```

## results/

Raw artifacts: per-arm `nvidia-smi` samples (`*_gpu.csv`), `trainer_state.json`
loss histories for the HF and Unsloth arms, `unsloth_native.json` (per-step
medians, peaks), environment (`versions.txt`, `pip_freeze.txt`), and
`hardware.csv`.
