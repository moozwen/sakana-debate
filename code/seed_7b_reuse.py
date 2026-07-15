# seed_7b_reuse.py — 7B の既存 Phase 1 結果を Part 1b の outdir にコピーして再利用する。
# run_debate.py のレジューム機構（済み idx をスキップ）がそのまま効くので、
# これを流しておけば 7B は新規302問だけ実行される。
# 使い方: cd sakana-debate && python code/seed_7b_reuse.py
import json
import os

SRC = "experiments/phase1_grid/results"
DST = "experiments/phase1b_1c_regime_budget/results/7b"
CONDS = [(1, 0), (3, 0), (3, 2)]  # 床 / SC / 討論
SEEDS = [1, 2, 3]

regime_idx = {json.loads(l)["idx"] for l in open("experiments/phase1b_1c_regime_budget/data/gsm8k_subset_regime.jsonl")}
os.makedirs(DST, exist_ok=True)

for n, r in CONDS:
    for seed in SEEDS:
        src = os.path.join(SRC, f"debate_N{n}_R{r}_seed{seed}.jsonl")
        dst = os.path.join(DST, f"debate_N{n}_R{r}_seed{seed}.jsonl")
        done = set()
        if os.path.exists(dst):
            done = {json.loads(l)["idx"] for l in open(dst) if l.strip()}
        n_copied = 0
        with open(dst, "a") as f:
            for l in open(src):
                if not l.strip():
                    continue
                rec = json.loads(l)
                if rec["idx"] in regime_idx and rec["idx"] not in done:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_copied += 1
        print(f"{dst}: +{n_copied} records reused")
print("done — 7B は run_debate.py config_regime_7b.yaml で残り（新規302問）だけ実行される")
