"""Build prompt-completion rows for BOTH arms, with identical tokenization.

Rationale: unsloth 2026.9.14's `formatting_func` path is internally broken --
its validator requires a list (`UnslothSFTTrainer.py:1331`) but its `.map`
concatenates the return value as a str, so a correct list raises
`TypeError: can only concatenate str (not "list") to str`.

So we pre-render and pre-tokenize to `prompt_completion` form, which takes
unsloth's separate `_tokenize_pc` branch, and give the same rows to Soup.

Emitting prompt/completion separately (rather than one joined string) lets us
mask the prompt out of the loss. Both arms then supervise ONLY the response
tokens -- the strictest fair comparison, since it removes any argument that one
side trained on prompt tokens the other did not.
"""
import hashlib
import json

from transformers import AutoTokenizer

MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
MAX_LEN = 1024
MAX_PROMPT = 768

tok = AutoTokenizer.from_pretrained(MODEL)

kept, tok_total, prompt_tok_total = 0, 0, 0
out = "/root/bench/data/prompt_completion.jsonl"

with open("/root/bench/data/train.jsonl") as f, open(out, "w") as g:
    for line in f:
        row = json.loads(line)
        prompt = row["instruction"] + (("\n\n" + row["input"]) if row.get("input") else "")
        ptext = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        ctext = row["output"] + tok.eos_token

        p_ids = tok(ptext, add_special_tokens=False)["input_ids"]
        c_ids = tok(ctext, add_special_tokens=False)["input_ids"]
        if len(p_ids) > MAX_PROMPT or len(p_ids) + len(c_ids) > MAX_LEN:
            continue
        g.write(json.dumps({
            "prompt": ptext,
            "completion": ctext,
            "prompt_ids": p_ids,
            "completion_ids": c_ids,
        }) + "\n")
        kept += 1
        tok_total += len(p_ids) + len(c_ids)
        prompt_tok_total += len(p_ids)

h = hashlib.sha256(open(out, "rb").read()).hexdigest()
print(f"rows={kept} total_tokens={tok_total} prompt_tokens={prompt_tok_total} "
      f"completion_tokens={tok_total - prompt_tok_total} sha256={h[:16]}")

# Validation rows, same treatment.
kept_v = 0
outv = "/root/bench/data/val_pc.jsonl"
with open("/root/bench/data/val.jsonl") as f, open(outv, "w") as g:
    for line in f:
        row = json.loads(line)
        prompt = row["instruction"] + (("\n\n" + row["input"]) if row.get("input") else "")
        ptext = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        ctext = row["output"] + tok.eos_token
        p_ids = tok(ptext, add_special_tokens=False)["input_ids"]
        c_ids = tok(ctext, add_special_tokens=False)["input_ids"]
        if len(p_ids) > MAX_PROMPT or len(p_ids) + len(c_ids) > MAX_LEN:
            continue
        g.write(json.dumps({"prompt": ptext, "completion": ctext,
                            "prompt_ids": p_ids, "completion_ids": c_ids}) + "\n")
        kept_v += 1
print(f"val rows={kept_v} sha256={hashlib.sha256(open(outv,'rb').read()).hexdigest()[:16]}")