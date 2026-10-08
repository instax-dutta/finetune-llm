"""Cap both datasets to 400 rows so 1 epoch x bs1 x accum8 == 50 steps.

Soup's schema requires an integer `epochs` and has no max_steps, so the step
count is controlled by dataset size instead. 400 rows / 8 = 50 steps, matching
the unsloth arm's max_steps=50 exactly. Same rows for every arm.
"""
import hashlib, json
N = 400
for src, dst in [("chatml", "chatml_50"), ("text", "text_50")]:
    out = f"/root/bench/data/{dst}.jsonl"
    n = 0
    with open(f"/root/bench/data/{src}.jsonl") as f, open(out, "w") as g:
        for line in f:
            if n >= N:
                break
            g.write(line)
            n += 1
    print(f"{dst}: rows={n} sha256={hashlib.sha256(open(out,'rb').read()).hexdigest()[:16]}")
