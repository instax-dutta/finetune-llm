#!/usr/bin/env python3
"""Post-run verification gate for a fine-tuning run.

Answers "did it actually train?" from the artifacts, without trusting the
trainer's exit code:
  1. adapter weights exist and are not zero / not empty
  2. training loss trajectory is finite and actually decreased
  3. run log does not contain known silent-failure signatures

Works on Soup output dirs, Unsloth output dirs, or any directory containing
`adapter_config.json` / `adapter_model.safetensors` / `trainer_state.json`
(checkpoints one level deep are found automatically).

Exit codes: 0 = pass, 2 = at least one check failed.
"""
import argparse
import glob
import json
import math
import os
import re
import struct
import sys


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data_start = 8 + n
    try:
        import numpy as np
    except Exception:
        return None
    mm = np.memmap(path, mode="r", offset=data_start, dtype=np.uint8)
    tensors = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = meta.get("dtype")
        start, end = meta.get("data_offsets", (0, 0))
        buf = mm[start:end]
        if dtype == "F32":
            arr = buf.view(np.float32)
        elif dtype == "F16":
            arr = buf.view(np.float16).astype(np.float32)
        elif dtype == "BF16":
            u16 = buf.view(np.uint16).astype(np.uint32)
            arr = (u16 << 16).view(np.float32)
        else:
            continue
        tensors[name] = arr
    return tensors


def find_first(candidates, patterns, num_key=None):
    for c in candidates:
        if os.path.exists(c):
            return c
    found = []
    for p in patterns:
        found.extend(glob.glob(p))
    if not found:
        return None
    if num_key:
        def key(p):
            m = re.search(num_key, p)
            return int(m.group(1)) if m else -1
        found.sort(key=key)
    else:
        found.sort(key=os.path.getmtime)
    return found[-1]


def check_adapter(run_dir, checks):
    adapter_dir = run_dir
    cfg = find_first(
        [os.path.join(run_dir, "adapter_config.json")],
        [os.path.join(run_dir, "*", "adapter_config.json")],
    )
    weights = find_first(
        [os.path.join(run_dir, "adapter_model.safetensors")],
        [os.path.join(run_dir, "*", "adapter_model.safetensors")],
        num_key=r"checkpoint-(\d+)",
    )
    if cfg:
        try:
            c = json.load(open(cfg))
            checks.append({"name": "adapter_config", "status": "pass",
                           "detail": f"r={c.get('r')} alpha={c.get('lora_alpha')} "
                                     f"targets={len(c.get('target_modules') or [])} base={c.get('base_model_name_or_path')}"})
        except Exception as e:
            checks.append({"name": "adapter_config", "status": "fail", "detail": f"unreadable: {e}"})
    else:
        checks.append({"name": "adapter_config", "status": "fail", "detail": "adapter_config.json not found"})

    if not weights:
        checks.append({"name": "adapter_weights", "status": "fail", "detail": "adapter_model.safetensors not found"})
        return
    tensors = read_safetensors(weights)
    if tensors is None:
        checks.append({"name": "adapter_weights", "status": "warn", "detail": "numpy unavailable; skipping tensor health"})
        return
    nonzero = 0
    total = 0
    frob2 = 0.0
    for arr in tensors.values():
        if arr.size == 0:
            continue
        total += arr.size
        amax = float(abs(arr).max())
        if amax > 0:
            nonzero += arr.size
        frob2 += float((arr.astype("float64") ** 2).sum())
    frac = (nonzero / total) if total else 0.0
    detail = (f"tensors={len(tensors)} params={total} nonzero_frac={frac:.4f} "
              f"frobenius={math.sqrt(frob2):.2f} file={os.path.relpath(weights, run_dir)}")
    if total == 0 or frac == 0.0:
        status = "fail"
    elif frac < 0.99:
        status = "warn"
    else:
        status = "pass"
    checks.append({"name": "adapter_weights", "status": status, "detail": detail})


def check_loss_trajectory(run_dir, min_decrease, checks):
    state = find_first(
        [os.path.join(run_dir, "trainer_state.json")],
        [os.path.join(run_dir, "*", "trainer_state.json"), os.path.join(run_dir, "checkpoint-*", "trainer_state.json")],
        num_key=r"checkpoint-(\d+)",
    )
    if not state:
        checks.append({"name": "loss_trajectory", "status": "fail", "detail": "trainer_state.json not found"})
        return
    try:
        d = json.load(open(state))
        losses = [e["loss"] for e in d.get("log_history", []) if "loss" in e and e["loss"] is not None]
    except Exception as e:
        checks.append({"name": "loss_trajectory", "status": "fail", "detail": f"unreadable: {e}"})
        return
    if len(losses) < 2:
        checks.append({"name": "loss_trajectory", "status": "fail", "detail": f"only {len(losses)} logged losses"})
        return
    first, last, lo = losses[0], losses[-1], min(losses)
    finite = all(math.isfinite(x) for x in losses)
    decrease = (first - last) / abs(first) if first else 0.0
    detail = (f"steps={len(losses)} first={first:.4f} last={last:.4f} min={lo:.4f} "
              f"decrease={decrease * 100:.1f}% state={os.path.relpath(state, run_dir)}")
    if not finite:
        checks.append({"name": "loss_trajectory", "status": "fail", "detail": detail + " (non-finite loss)"})
    elif decrease < min_decrease:
        checks.append({"name": "loss_trajectory", "status": "fail",
                       "detail": detail + f" (needs >= {min_decrease * 100:.0f}% decrease; flat or rising loss)"})
    else:
        checks.append({"name": "loss_trajectory", "status": "pass", "detail": detail})


LOG_PATTERNS = [
    ("oom", "fail", re.compile(r"CUDA out of memory|OutOfMemoryError|torch\.OutOfMemory|CUDA error: out of memory")),
    ("nan_loss", "fail", re.compile(r"loss['\"=:\s]+nan|nan loss", re.IGNORECASE)),
    ("traceback", "warn", re.compile(r"Traceback \(most recent call last\)")),
    ("unsloth_frames", "info", re.compile(r"unsloth/[\w./]+\.py")),
]


def check_log(log_path, checks):
    if not log_path:
        checks.append({"name": "run_log", "status": "warn", "detail": "no --log given; skipped log scan"})
        return
    if not os.path.exists(log_path):
        checks.append({"name": "run_log", "status": "warn", "detail": f"{log_path} not found"})
        return
    text = open(log_path, "r", errors="replace").read()
    hits = []
    status = "pass"
    for name, sev, rx in LOG_PATTERNS:
        n = len(rx.findall(text))
        if n:
            hits.append(f"{name}={n}")
            if sev == "fail":
                status = "fail"
            elif sev == "warn" and status == "pass":
                status = "warn"
    if "does not contain `{% generation %}`" in text or "does not contain {% generation %}" in text:
        hits.append("response_only_masking_silently_disabled")
        if status == "pass":
            status = "warn"
    checks.append({"name": "run_log", "status": status, "detail": ", ".join(hits) if hits else "no known failure signatures"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="output dir of the training run")
    ap.add_argument("--log", help="path to the captured stdout/stderr log (optional)")
    ap.add_argument("--min-decrease", type=float, default=0.05, help="required relative loss decrease (default 0.05)")
    ap.add_argument("--json", action="store_true", help="print a JSON report")
    args = ap.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"run dir not found: {args.run_dir}", file=sys.stderr)
        return 2

    checks = []
    check_adapter(args.run_dir, checks)
    check_loss_trajectory(args.run_dir, args.min_decrease, checks)
    check_log(args.log, checks)

    failed = [c for c in checks if c["status"] == "fail"]
    report = {"run_dir": args.run_dir, "pass": not failed, "checks": checks}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for c in checks:
            mark = {"pass": "PASS", "fail": "FAIL", "warn": "WARN", "info": "INFO"}[c["status"]]
            print(f"[{mark}] {c['name']}: {c['detail']}")
        print("\nRESULT:", "PASS" if not failed else f"FAIL ({len(failed)} failed check(s))")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
