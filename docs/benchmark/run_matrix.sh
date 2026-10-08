#!/bin/bash
# Four-arm comparison, 2x T4, 50 steps each, identical data + hyper-parameters.
#
# Arms:
#   soup_hf_pure    soup backend=transformers, unsloth import BLOCKED via PYTHONPATH
#                   -> the honest stock-HF arm. Without the block this arm is
#                      silently unsloth-backed (commands/train.py:1517 calls
#                      is_unsloth_available(), which imports unsloth, and unsloth
#                      patches the whole HF/trl stack at import).
#   soup_hf_plain   soup backend=transformers, unsloth importable
#                   -> measured to PROVE the silent-patching effect
#   soup_unsloth    soup backend=unsloth
#   unsloth_native  unsloth's own script, no soup in the loop
#
# 400 rows / (bs1 x accum8) = 50 steps, matching unsloth's max_steps=50.
# One arm per GPU. T4s are host-connected with no PCIe path between them.
set -u
cd /root/bench
MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"
mkdir -p logs out
pkill -9 -f sample_gpu.sh 2>/dev/null
pkill -9 -f "soup train" 2>/dev/null
pkill -9 -f "run_unsloth.py" 2>/dev/null
sleep 3
rm -f logs/arm_status.txt

arm() {
  local tag="$1" gpu="$2"; shift 2
  ./sample_gpu.sh "$gpu" "logs/${tag}_gpu.csv" 2 >/dev/null 2>&1 &
  local sampler=$!
  "$@" > "logs/${tag}.log" 2>&1
  local rc=$?
  kill "$sampler" 2>/dev/null
  echo "$tag exit=$rc" >> logs/arm_status.txt
}

: > logs/arm_status.txt

# pass 1 -- gpu0: soup stock-HF (blocked)   gpu1: unsloth native
arm soup_hf_pure 0 env CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=/root/bench/no_unsloth \
    soup train --config soup_cfg_hf_pure.yaml --yes &
A=$!

arm unsloth_native 1 env CUDA_VISIBLE_DEVICES=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    python run_unsloth.py --model "$MODEL" --data data/text_50.jsonl \
    --out out/unsloth_native.json --steps 50 --arm unsloth_native &
B=$!
wait $A $B
echo "=== pass 1 done ==="

# pass 2 -- gpu0: soup backend=unsloth   gpu1: soup transformers, unsloth importable
arm soup_unsloth 0 env CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    soup train --config soup_cfg_unsloth.yaml --yes &
C=$!

arm soup_hf_plain 1 env CUDA_VISIBLE_DEVICES=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    soup train --config soup_cfg_hf_plain.yaml --yes &
D=$!
wait $C $D
pkill -9 -f sample_gpu.sh 2>/dev/null
echo "=== ALL ARMS DONE ==="
cat logs/arm_status.txt