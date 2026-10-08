#!/usr/bin/env python3
"""Hardware probe and fit planner for LLM fine-tuning.

Answers, before any config is written:
  - what GPU / VRAM / compute capability / true bf16 support exists here
  - how many params the target model has (HF API, no weights downloaded)
  - whether it fits, for the Unsloth engine vs stock TRL, at a given seq length

Budgets are calibrated against a measured 2x-T4 benchmark (8B NF4 QLoRA,
seq 1024, batch 1): peak ~= 0.90 GB x params(B) for Unsloth,
peak ~= 1.85 GB x params(B) for stock TRL. See references/hardware-and-engines.md.

Exit codes: 0 = fits with the recommended engine, 2 = does not fit.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import urllib.request

UNSLOTH_GB_PER_B = 0.90
TRL_GB_PER_B = 1.85
HEADROOM = 0.90


def cpu_cores():
    try:
        return os.cpu_count()
    except Exception:
        return None


def total_ram_gb():
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True)
            return round(int(out.stdout.strip()) / 1e9, 1)
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) * 1024 / 1e9, 1)
    except Exception:
        return None
    return None


def nvidia_smi_gpus():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
        if out.returncode != 0:
            return []
    except Exception:
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            cc = parts[2] if len(parts) > 2 and parts[2] not in ("", "N/A") else None
            gpus.append({
                "name": parts[0],
                "vram_gb": round(float(parts[1]) / 1024, 1),
                "compute_cap": cc,
                "source": "nvidia-smi",
            })
    return gpus


def torch_probe():
    info = {"available": False}
    try:
        import torch
    except Exception as e:
        info["error"] = f"import torch failed: {type(e).__name__}: {e}"
        return info, None
    info["available"] = True
    info["version"] = torch.__version__

    if torch.cuda.is_available():
        gpus = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            cc = f"{p.major}.{p.minor}"
            try:
                bf16 = bool(torch.cuda.is_bf16_supported(including_emulation=False))
                bf16_note = "hardware"
            except TypeError:
                bf16 = bool(torch.cuda.is_bf16_supported())
                bf16_note = "may include emulation (old torch)"
            except Exception:
                bf16 = False
                bf16_note = "unknown"
            gpus.append({
                "name": p.name,
                "vram_gb": round(p.total_memory / 1e9, 1),
                "compute_cap": cc,
                "bf16_hardware": bf16,
                "bf16_note": bf16_note,
                "source": "torch",
            })
        info["cuda"] = True
        info["devices"] = gpus
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        info["cuda"] = False
        info["mps"] = True
    else:
        info["cuda"] = False
    return info, torch


def hf_param_count(model_id):
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "finetune-llm-skill"})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r)
    except Exception as e:
        return None, f"HF API failed: {type(e).__name__}: {e}"
    st = d.get("safetensors") or {}
    total = st.get("total")
    if total:
        return float(total), "safetensors.total"
    cfg = d.get("config") or {}
    try:
        h = cfg.get("hidden_size") or cfg.get("n_embd")
        layers = cfg.get("num_hidden_layers") or cfg.get("n_layer")
        inter = cfg.get("intermediate_size") or cfg.get("n_inner") or (4 * h)
        vocab = cfg.get("vocab_size")
        if h and layers and vocab:
            per_layer = 4 * h * h + 3 * h * inter
            est = layers * per_layer + vocab * h
            return float(est), "config-estimate (tied embeddings assumed)"
    except Exception:
        pass
    return None, "no param count available from HF API; pass --params-billions"


def pick_engine(vram_gb, unsloth_peak, trl_peak):
    budget = vram_gb * HEADROOM
    fits_unsloth = unsloth_peak <= budget
    fits_trl = trl_peak <= budget
    if fits_unsloth:
        return "unsloth" if not fits_trl else "unsloth", budget
    if fits_trl:
        return "transformers", budget
    return "none", budget


def dtype_advice(gpu):
    cc = gpu.get("compute_cap")
    if not cc:
        return "fp16 (compute capability unknown)"
    major = int(str(cc).split(".")[0])
    if major >= 8 and gpu.get("bf16_hardware"):
        return "bf16"
    return "fp16 (no bf16 hardware; torch.cuda.is_bf16_supported() may still say True via emulation)"


def build_report(args):
    report = {
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_cores": cpu_cores(),
            "ram_gb": total_ram_gb(),
        },
        "torch": None,
        "gpus": [],
        "model": {},
        "plan": {},
    }
    tinfo, _ = torch_probe()
    report["torch"] = tinfo

    gpus = tinfo.get("devices") or []
    if not gpus and not tinfo.get("cuda"):
        smi = nvidia_smi_gpus()
        if smi:
            for g in smi:
                g["bf16_hardware"] = None
                g["bf16_note"] = "torch unavailable; assume fp16 on sm<80"
            gpus = smi
    report["gpus"] = gpus

    params_b = args.params_billions
    params_src = "cli"
    if params_b is None and args.model:
        params, src = hf_param_count(args.model)
        if params:
            params_b = params / 1e9
            params_src = src
    report["model"] = {
        "id": args.model,
        "params_billions": round(params_b, 3) if params_b else None,
        "param_source": params_src if params_b else None,
        "seq_len": args.seq,
    }

    if params_b and gpus:
        unsloth_peak = UNSLOTH_GB_PER_B * params_b
        trl_peak = TRL_GB_PER_B * params_b
        for gpu in gpus:
            engine, budget = pick_engine(gpu["vram_gb"], unsloth_peak, trl_peak)
            gpu["plan"] = {
                "vram_budget_gb": round(budget, 1),
                "est_peak_unsloth_gb": round(unsloth_peak, 1),
                "est_peak_trl_gb": round(trl_peak, 1),
                "recommended_engine": engine,
                "dtype": dtype_advice(gpu),
            }
        first = gpus[0]["plan"]
        report["plan"] = {
            "fits": first["recommended_engine"] != "none",
            "recommended_engine": first["recommended_engine"],
            "dtype": first["dtype"],
            "note": "estimates calibrated at seq=1024, batch=1, NF4 QLoRA r=16, 8B; "
                    "treat as lower bounds for longer sequences and require headroom",
        }
        if args.seq > 1024:
            report["plan"]["note"] += f" (seq={args.seq} will exceed these estimates)"

    return report


def human_summary(r):
    lines = []
    p = r["platform"]
    lines.append(f"platform : {p['system']} {p['machine']} | {p['cpu_cores']} cores | {p['ram_gb']} GB RAM | py {p['python']}")
    if r["gpus"]:
        for i, g in enumerate(r["gpus"]):
            bf = g.get("bf16_hardware")
            bf_s = "bf16=yes" if bf else ("bf16=no" if bf is False else "bf16=?")
            lines.append(f"gpu {i}    : {g['name']} | {g['vram_gb']} GB | cc {g.get('compute_cap')} | {bf_s} ({g.get('source')})")
    else:
        mps = (r.get("torch") or {}).get("mps")
        lines.append(f"gpu      : none detected{' | Apple MPS available (use backend: mlx)' if mps else ''}")
        if not mps:
            lines.append("           no GPU: rent one for training; CPU streaming is for tiny models only")
    m = r["model"]
    if m.get("params_billions"):
        lines.append(f"model    : {m['id']} | {m['params_billions']}B params ({m['param_source']}) | seq {m['seq_len']}")
    for g in r["gpus"]:
        pl = g.get("plan")
        if not pl:
            continue
        lines.append(
            f"plan     : budget {pl['vram_budget_gb']} GB | est peak unsloth {pl['est_peak_unsloth_gb']} GB, "
            f"trl {pl['est_peak_trl_gb']} GB -> engine: {pl['recommended_engine']} | dtype: {pl['dtype']}"
        )
    if r["plan"].get("note"):
        lines.append(f"note     : {r['plan']['note']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="HF model id, e.g. Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--params-billions", type=float, default=None, help="override param count (billions) when offline")
    ap.add_argument("--seq", type=int, default=1024, help="planned max sequence length (default 1024)")
    ap.add_argument("--out", help="write JSON report to this path")
    ap.add_argument("--json", action="store_true", help="print JSON instead of the human summary")
    args = ap.parse_args()

    if not args.model and args.params_billions is None:
        ap.error("pass --model (preferred) or --params-billions")

    report = build_report(args)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(human_summary(report))
        if args.out:
            print(f"\nwrote {args.out}")

    fits = report["plan"].get("fits")
    return 0 if fits is None or fits else 2


if __name__ == "__main__":
    sys.exit(main())
