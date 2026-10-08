#!/usr/bin/env python3
"""Deterministic SFT data preparation with provenance.

Does the boring, mandatory parts of data prep so every run is comparable:
  - detects the input format (alpaca | messages/chatml | prompt_completion | text)
  - filters rows longer than --max-len (measured with the model's tokenizer)
  - splits train/val with a fixed seed
  - writes a manifest.json with sha256 hashes and token counts

The same seed, same input, same tokenizer always produce byte-identical
outputs. Commit manifest.json with the run so a result can be traced to its data.

Works without transformers/models (falls back to a chars/4 length estimate and
says so in the manifest).
"""
import argparse
import datetime
import hashlib
import json
import os
import random
import sys


def detect_format(row):
    if "messages" in row:
        return "messages"
    if "prompt" in row and "completion" in row:
        return "prompt_completion"
    if "text" in row:
        return "text"
    if "instruction" in row and "output" in row:
        return "alpaca"
    return "unknown"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_token_counter(model_id, no_tokenize):
    if no_tokenize or not model_id:
        return None, None
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        print(f"warn: transformers unavailable ({e}); using chars/4 estimate", file=sys.stderr)
        return None, None
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print(f"warn: could not load tokenizer for {model_id} ({e}); using chars/4 estimate", file=sys.stderr)
        return None, None

    def count(row, fmt):
        if fmt == "messages":
            msgs = row["messages"]
            try:
                ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
            except Exception:
                ids = tok("\n".join(m.get("content", "") for m in msgs))["input_ids"]
            return len(ids)
        if fmt == "prompt_completion":
            p = len(tok(row["prompt"], add_special_tokens=False)["input_ids"])
            c = len(tok(row["completion"], add_special_tokens=False)["input_ids"])
            return p + c
        if fmt == "text":
            return len(tok(row["text"], add_special_tokens=False)["input_ids"])
        instr = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        out = (row.get("output") or "").strip()
        prompt = instr + (("\n\n" + inp) if inp else "")
        try:
            p = len(tok.apply_chat_template([{"role": "user", "content": prompt}],
                                            tokenize=True, add_generation_prompt=True))
        except Exception:
            p = len(tok(prompt, add_special_tokens=False)["input_ids"])
        c = len(tok(out, add_special_tokens=False)["input_ids"])
        return p + c

    return count, tok.name_or_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="input .jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", help="tokenizer/model id used for length measurement")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--val-split", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--format", default="auto", choices=["auto", "alpaca", "messages", "prompt_completion", "text"])
    ap.add_argument("--no-tokenize", action="store_true", help="skip tokenizer; estimate lengths as chars/4")
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        print(f"input not found: {args.inp}", file=sys.stderr)
        return 2

    rows = load_rows(args.inp)
    if not rows:
        print("input has no rows", file=sys.stderr)
        return 2

    fmt = args.format if args.format != "auto" else detect_format(rows[0])
    if fmt == "unknown":
        print("could not detect format; expected alpaca|messages|prompt_completion|text", file=sys.stderr)
        return 2
    bad = [i for i, r in enumerate(rows) if detect_format(r) != fmt]
    if bad and args.format == "auto":
        print(f"refusing: mixed formats (row {bad[0]} differs); pass --format explicitly", file=sys.stderr)
        return 2

    counter, tok_name = build_token_counter(args.model, args.no_tokenize)
    method = "tokenizer" if counter else "chars/4-estimate"

    kept, too_long = [], 0
    tokens_total = 0
    for r in rows:
        if counter:
            n = counter(r, fmt)
        else:
            n = max(1, len(json.dumps(r)) // 4)
        if n > args.max_len:
            too_long += 1
            continue
        kept.append(r)
        tokens_total += n

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    n_val = int(round(len(kept) * args.val_split))
    val_rows = kept[:n_val]
    train_rows = kept[n_val:]

    os.makedirs(args.out_dir, exist_ok=True)
    out_paths = {}
    for name, subset in [("train.jsonl", train_rows), ("val.jsonl", val_rows)]:
        path = os.path.join(args.out_dir, name)
        with open(path, "w") as f:
            for r in subset:
                f.write(json.dumps(r) + "\n")
        out_paths[name] = {"path": path, "rows": len(subset), "sha256": sha256_file(path)}

    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "input": {"path": args.inp, "rows": len(rows), "sha256": sha256_file(args.inp)},
        "format": fmt,
        "model_tokenizer": tok_name or args.model,
        "length_method": method,
        "max_len": args.max_len,
        "seed": args.seed,
        "val_split": args.val_split,
        "dropped_too_long": too_long,
        "tokens_total_kept": tokens_total,
        "outputs": out_paths,
        "run_hints": {
            "soup_format": {"alpaca": "alpaca", "messages": "chatml", "text": "text",
                            "prompt_completion": "not native to soup; use pre_tokenized or an alpaca/chatml remap"}[fmt],
            "soup_data_block": {"train": out_paths["train.jsonl"]["path"],
                                "format": {"alpaca": "alpaca", "messages": "chatml", "text": "text"}.get(fmt, "auto"),
                                "max_length": args.max_len,
                                "val_split": 0.0},
            "note": "val rows are in val.jsonl; keep them out of training (val_split: 0) and score with scripts/eval_adapter.py",
        },
    }
    mpath = os.path.join(args.out_dir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"format={fmt} rows_in={len(rows)} kept={len(kept)} train={len(train_rows)} val={len(val_rows)} "
          f"dropped_too_long={too_long} tokens={tokens_total} method={method}")
    print(f"wrote {out_paths['train.jsonl']['path']} sha256={out_paths['train.jsonl']['sha256'][:16]}")
    if len(val_rows):
        print(f"wrote {out_paths['val.jsonl']['path']} sha256={out_paths['val.jsonl']['sha256'][:16]}")
    print(f"wrote {mpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
