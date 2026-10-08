"""Held-out loss + adapter sanity for every arm.

Held-out loss is computed by BOTH adapters through the SAME neutral loader
(no soup, no unsloth), so the comparison cannot be confounded by whichever
trainer wrote the file. Also counts non-zero adapter tensors, because a
"soup" adapter that saved zeros would otherwise look like a training run.
"""
import glob, json, os, sys
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"
VAL="/root/bench/data/val_pc.jsonl"

# 1) adapter health, no GPU needed
rows=[]
for name in ["hf_plain","unsloth"]:
    d=f"/root/bench/out/{name}"
    f=os.path.join(d,"adapter_model.safetensors")
    if not os.path.exists(f):
        rows.append({"arm":name,"status":"missing"}); continue
    sd=load_file(f)
    nz=sum(1 for v in sd.values() if v.abs().max().item()>0)
    tot=sum(v.numel() for v in sd.values())
    nrm=sum(float(v.float().norm()) for v in sd.values())
    rows.append({"arm":name,"tensors":len(sd),"nonzero":nz,"params":tot,"frobenius_norm":round(nrm,4),
                 "all_zero": nz==0})
print("=== ADAPTER HEALTH ===")
for r in rows: print(" ", r)

# 2) held-out loss, one neutral loader, fp16, both adapters
tok=AutoTokenizer.from_pretrained(MODEL)
# 4-bit to match how both arms were trained; fp16 base (14.4GB) does not fit
# on a 15.6GB card alongside activations.
from transformers import BitsAndBytesConfig
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                       bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
base=AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="cuda:0")
base.eval()
tok.padding_side="right"
if tok.pad_token is None: tok.pad_token=tok.eos_token

import datasets as hfds
ds=hfds.load_dataset("json", data_files={"val":VAL}, split="val")
def enc(r):
    p=r["prompt_ids"]; c=r["completion_ids"]
    ids=p+c
    labels=[-100]*len(p)+c[:]
    return {"input_ids":ids,"labels":labels}
ds=ds.map(enc, remove_columns=ds.column_names)
def coll(b):
    # Right-pad to the batch max. Labels padded with -100 so padding never
    # contributes to the loss.
    mx=max(len(x["input_ids"]) for x in b)
    ids=torch.full((len(b),mx), tok.pad_token_id, dtype=torch.long)
    att=torch.zeros((len(b),mx), dtype=torch.long)
    lab=torch.full((len(b),mx), -100, dtype=torch.long)
    for i,x in enumerate(b):
        n=len(x["input_ids"])
        ids[i,:n]=torch.tensor(x["input_ids"])
        att[i,:n]=1
        lab[i,:n]=torch.tensor(x["labels"])
    return {"input_ids":ids.to("cuda:0"),"attention_mask":att.to("cuda:0"),"labels":lab.to("cuda:0")}

print("\n=== HELD-OUT LOSS (same neutral loader for both) ===")
out={}
for name in ["hf_plain","unsloth"]:
    from peft import PeftModel
    m=PeftModel.from_pretrained(base, f"/root/bench/out/{name}", adapter_name=name).eval()
    tot,n=0.0,0
    with torch.no_grad():
        for i in range(0,len(ds),4):
            b=[ds[j] for j in range(i,min(i+4,len(ds)))]
            batch=coll(b)
            out_=m(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],labels=batch["labels"])
            k=sum((x!=-100).sum().item() for x in batch["labels"])
            tot+=float(out_.loss)*k; n+=k
    print(f"  {name:12s} held_out_loss={tot/n:.5f}  tokens={n}")
    out[name]=tot/n
    m.delete_adapter(name)
print("\nJSON:", json.dumps(out,indent=2))
