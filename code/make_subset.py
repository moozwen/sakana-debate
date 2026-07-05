import json
import random

from datasets import load_dataset

SEED, N = 42, 200
ds = load_dataset("openai/gsm8k", "main", split="test")
idx = random.Random(SEED).sample(range(len(ds)), N)
with open("gsm8k_subset.jsonl", "w") as f:
    for i in idx:
        q = ds[i]
        # GSM8K の正解は解説末尾の "#### <数>"
        gold = q["answer"].split("####")[-1].strip().replace(",", "")
        f.write(json.dumps({"idx": i, "question": q["question"], "gold": gold}) + "\n")
print(f"wrote {N} problems (seed={SEED})")
