"""Emit ONE dataset both frameworks accept: chatml (Soup) -> also readable by unsloth.

Soup has no prompt_completion format, and unsloth's formatting_func path is
broken internally, so the only common denominator is a single text column built
from the chat template. Emitted twice:
  chatml.jsonl  -- {"messages":[{role,content}...]}  (Soup native)
  text.jsonl    -- {"text": "<rendered>"}           (unsloth dataset_text_field)
Both render to byte-identical strings, so the arms see identical tokens.
"""
import hashlib, json
from transformers import AutoTokenizer

MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"; MAX_LEN=1024
tok=AutoTokenizer.from_pretrained(MODEL)

kept=0; ctok=0
src="/root/bench/data/prompt_completion.jsonl"
with open(src) as f, open("/root/bench/data/chatml.jsonl","w") as gc, open("/root/bench/data/text.jsonl","w") as gt:
    for line in f:
        r=json.loads(line)
        msgs=[{"role":"user","content":r["prompt"]},{"role":"assistant","content":r["completion"]}]
        text=tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        n=len(tok(text, add_special_tokens=False)["input_ids"])
        if n>MAX_LEN: continue
        gc.write(json.dumps({"messages":msgs})+"\n")
        gt.write(json.dumps({"text":text})+"\n")
        kept+=1; ctok+=len(r["completion_ids"])

for p in ["chatml.jsonl","text.jsonl"]:
    print(p, hashlib.sha256(open(f"/root/bench/data/{p}","rb").read()).hexdigest()[:16])
print(f"rows={kept} completion_tokens_supervised={ctok}")
