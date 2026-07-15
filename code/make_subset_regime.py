# make_subset_regime.py — Part 1b 用 500問サブセット（入れ子構成: 既存198 + 新規302）
# 事前登録: vault/01_Inbox/04_Crafts/sakana_problem/part1b-regime-preregistration.md §3.2
# 既存200問（experiments/phase1_grid/data/gsm8k_subset.jsonl）から不良2問を除いた198問を保持し、
# 7B の既存結果を再利用できるようにする。残り302問は未使用プールから決定的に追加。
import json
import random

from datasets import load_dataset

N_TOTAL = 500
EXCLUDE = {1309, 255}  # 1309: gold ラベル誤り / 255: 問題文の自己矛盾（Phase 1 で確定）
NEW_SEED = 43          # 旧サブセット（seed=42）と独立に、新規分の抽出を決定的にする

old_idx = [json.loads(l)["idx"] for l in open("experiments/phase1_grid/data/gsm8k_subset.jsonl")]
keep = [i for i in old_idx if i not in EXCLUDE]

ds = load_dataset("openai/gsm8k", "main", split="test")
pool = [i for i in range(len(ds)) if i not in set(old_idx) and i not in EXCLUDE]
new = random.Random(NEW_SEED).sample(pool, N_TOTAL - len(keep))

with open("experiments/phase1b_1c_regime_budget/data/gsm8k_subset_regime.jsonl", "w") as f:
    for i in keep + new:
        q = ds[i]
        gold = q["answer"].split("####")[-1].strip().replace(",", "")
        f.write(json.dumps({"idx": i, "question": q["question"], "gold": gold}) + "\n")
print(f"wrote {len(keep) + len(new)} problems (reused={len(keep)}, new={len(new)}, excluded={sorted(EXCLUDE)})")
