# analyze.py — Phase 1 集計。R曲線 / N曲線（seed平均±95%CI）と RQ5（不一致解析）
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}  # 自由度 k-1 の t 値


def ci(vals):
    k = len(vals)
    m = sum(vals) / k
    if k < 2:
        return m, 0.0
    sd = (sum((v - m) ** 2 for v in vals) / (k - 1)) ** 0.5
    return m, T95[k] * sd / k**0.5


# --- 読み込み：条件 (n,r) → {seed: [recs]} ---
data = defaultdict(dict)
for path in glob.glob("experiments/phase1_grid/results/debate_N*_R*_seed*.jsonl"):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if recs:
        data[(recs[0]["n"], recs[0]["r"])][recs[0]["seed"]] = recs

acc = {
    k: {s: sum(r["ok"] for r in v) / len(v) for s, v in d.items()}
    for k, d in data.items()
}

print("=== accuracy per condition (mean ± 95%CI over seeds) ===")
for n, r in sorted(acc):
    m, h = ci(list(acc[(n, r)].values()))
    print(f"N={n} R={r}: {m:.3f} ± {h:.3f}  (seeds={sorted(acc[(n, r)])})")


def plot_slice(pairs, xkey, fixed_label, fname):
    xs = [p[0] for p in pairs]
    ms, hs = zip(*[ci(list(acc[p[1]].values())) for p in pairs])
    plt.figure(figsize=(5, 3.5))
    plt.errorbar(xs, ms, yerr=hs, marker="o", capsize=4)
    plt.axhline(191 / 200, ls="--", lw=1, label="greedy floor 0.955 (temp=0)")
    plt.xlabel(xkey)
    plt.ylabel("GSM8K accuracy (200 problems)")
    plt.title(f"debate accuracy vs {xkey} ({fixed_label}, mean±95%CI, 3 seeds)")
    plt.xticks(xs)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    print(f"wrote {fname}")


r_slice = [(r, (3, r)) for r in [0, 1, 2, 3] if (3, r) in acc]
n_slice = [(n, (n, 2)) for n in [1, 2, 4, 6] if (n, 2) in acc]
if r_slice:
    plot_slice(r_slice, "rounds R", "N=3", "experiments/phase1_grid/figures/fig_r_curve.png")
if n_slice:
    plot_slice(n_slice, "agents N", "R=2", "experiments/phase1_grid/figures/fig_n_curve.png")

# --- RQ5：不一致は誤りの予測子か ---
print("\n=== RQ5: unanimity vs accuracy（最終ラウンドの全会一致で層別） ===")
print(f"{'cond':>10} {'coverage':>9} {'acc(unanimous)':>15} {'acc(split)':>11}")
for n, r in sorted(data):
    if n < 2:
        continue  # 不一致は N>=2 でのみ定義できる
    allrecs = [x for recs in data[(n, r)].values() for x in recs]
    una = [x for x in allrecs if x["unanimous"]]
    spl = [x for x in allrecs if not x["unanimous"]]
    cov = len(una) / len(allrecs)
    a_u = sum(x["ok"] for x in una) / len(una) if una else float("nan")
    a_s = sum(x["ok"] for x in spl) / len(spl) if spl else float("nan")
    print(f"N={n} R={r:>2} {cov:>8.1%} {a_u:>15.3f} {a_s:>11.3f}")
print(
    "→ acc(unanimous) ≫ acc(split) なら「全会一致でなければ人間へ」が有効な自己申告"
    "（coverage がそのときの自動処理率）"
)
