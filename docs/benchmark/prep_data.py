"""Shared prep for the Soup-vs-Unsloth benchmark.

Builds a fixed, seeded SFT dataset in Alpaca format from yahma/alpaca-cleaned.
Both arms consume the identical bytes: same file, same order, same token count.
"""
import hashlib
import json
import os
import random

from datasets import load_dataset

SEED = 3407
OUT = "/root/bench/data"
TRAIN_N = 4000
VAL_N = 200
MAX_CHARS = 4096

os.makedirs(OUT, exist_ok=True)

ds = load_dataset("yahma/alpaca-cleaned", split="train")
print("raw rows:", len(ds))

idx = list(range(len(ds)))
random.Random(SEED).shuffle(idx)

train_rows, val_rows = [], []
for i in idx:
    if len(train_rows) >= TRAIN_N and len(val_rows) >= VAL_N:
        break
    ex = ds[i]
    instr = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    out = (ex.get("output") or "").strip()
    if not instr or not out:
        continue
    if len(instr) + len(inp) + len(out) > MAX_CHARS:
        continue
    row = {"instruction": instr, "input": inp, "output": out}
    if len(train_rows) < TRAIN_N:
        train_rows.append(row)
    elif len(val_rows) < VAL_N:
        val_rows.append(row)

print("train:", len(train_rows), "val:", len(val_rows))
assert len(train_rows) == TRAIN_N and len(val_rows) == VAL_N, "short dataset"

for name, rows in [("train.jsonl", train_rows), ("val.jsonl", val_rows)]:
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    with open(path + ".sha256", "w") as fh:
        fh.write(h + "\n")
    print(f"wrote {path} {os.path.getsize(path)}B sha256={h[:16]}")
