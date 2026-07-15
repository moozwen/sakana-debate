# plot_agreement_ternary.py — 合意三角図（agreement ternary）: {全員一致 / 2-1 割れ / 三者三様}
# 構想: vault/01_Inbox/04_Crafts/sakana_problem/slm-petrology-map-explainer-20260713.md §2
#
# 性格: 記述図・検定なし・ラベルフリー座標（床の注記のみ既公表のラベル使用値を再掲）。
#   事前登録の対象外の探索的作図。S1-D として登録に載せる場合は、本図の閲覧範囲を登録に記載すること。
#
# 描くもの:
#   fig_ternary_main.png   — 8 モデルの全岩点（モデル単位の平均組成）+ 理論等値線グリッド
#                            （iso-p: 床一定 / iso-m: 誤答相関一定）
#   fig_ternary_models.png — モデル別 3×3 面: 問題ごとの 3 seed 組成（薄片クラウド、格子点バブル）
#   stdout                 — 組成表・ā との整合・実効 (p̂, m̂) の逆算（全岩の実効値 — 混合の注意つき）
#
# 理論グリッドの生成モデル（コメントで固定 — 図の脚注にも明記）:
#   1 票 = 確率 p で正解（正解は一意）、確率 1−p で誤答。
#   誤答分布は「単一優勢誤答（質量 m）+ 塵（互いに衝突しない）」と仮定:
#     全員一致 U = p^3 + (1−p)^3 m^3
#     三者三様 D = 3p(1−p)^2 (1−m^2) + (1−p)^3 (1 − 3m^2 + 2m^3)
#     2-1 割れ S = 1 − U − D
#   誤答ペア衝突率 c = m^2。モデル単位の点は問題ごとの p の混合なので、
#   逆算される (p̂, m̂) は実効値（全岩分析）であり真の平均 p ではない。
#
# 使い方: cd sakana-debate && python code/plot_agreement_ternary.py [data_root]
import json
import os
import sys
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "experiments/phase1b_1c_regime_budget/results")
IMG = os.path.join(DATA, "img")
os.makedirs(IMG, exist_ok=True)

MODELS = [("Qwen2.5", "0.5B", "05b"), ("Qwen2.5", "1.5B", "15b"),
          ("Qwen2.5", "3B", "3b"), ("Qwen2.5", "7B", "7b"),
          ("Gemma3", "1B", "gemma1b"), ("Gemma3", "4B", "gemma4b"),
          ("Llama3.2", "1B", "llama1b"), ("Llama3.2", "3B", "llama3b")]
FAM_STYLE = {"Qwen2.5": ("tab:blue", "o"), "Gemma3": ("tab:orange", "D"),
             "Llama3.2": ("tab:green", "s")}
SEEDS = [1, 2, 3]

# 三角図の頂点: 上 = 全員一致(U)、右下 = 2-1 割れ(S)、左下 = 三者三様(D)
V_U, V_S, V_D = (0.5, 3 ** 0.5 / 2), (1.0, 0.0), (0.0, 0.0)


def to_xy(u, s, d):
    return (u * V_U[0] + s * V_S[0] + d * V_D[0],
            u * V_U[1] + s * V_S[1] + d * V_D[1])


def load(sub, n, r):
    out = {}
    for se in SEEDS:
        fn = os.path.join(DATA, sub, f"debate_N{n}_R{r}_seed{se}.jsonl")
        if os.path.exists(fn):
            out[se] = {j["idx"]: j for l in open(fn) if l.strip() for j in [json.loads(l)]}
    return out


def classify(preds):
    """3 票 → 'u'(全員一致) / 's'(2-1) / 'd'(三者三様)。None は固有の非一致票扱い
    （pair_agree と同じ規約: None は何とも一致しない。pred=None は実測 0.0% なので影響なし）。"""
    cnt = Counter(p for p in preds if p is not None)
    top = max(cnt.values()) if cnt else 0
    return "u" if top == 3 else ("s" if top == 2 else "d")


# ---------- 理論グリッド（データ不使用） ----------
def theory(p, m):
    u = p ** 3 + (1 - p) ** 3 * m ** 3
    d = 3 * p * (1 - p) ** 2 * (1 - m ** 2) + (1 - p) ** 3 * (1 - 3 * m ** 2 + 2 * m ** 3)
    return u, 1 - u - d, d


def draw_frame(ax, grid=True):
    xs, ys = zip(V_D, V_S, V_U, V_D)
    ax.plot(xs, ys, color="0.35", lw=1.0, zorder=1)
    # 図中テキストは英語のみ（matplotlib 既定フォントに日本語グリフがないため — リポジトリの図の規約）
    ax.text(*V_U, "  unanimous", ha="left", va="bottom", fontsize=7, color="0.25")
    ax.text(*V_S, "2-1 split ", ha="left", va="top", fontsize=7, color="0.25")
    ax.text(*V_D, " 3-way ", ha="right", va="top", fontsize=7, color="0.25")
    if grid:
        ts = [i / 60 for i in range(61)]
        for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            pts = [to_xy(*theory(p, m)) for m in ts]
            ax.plot([x for x, _ in pts], [y for _, y in pts], color="0.75", lw=0.7, zorder=1)
            ax.annotate(f"p={p:.1f}", pts[-1], fontsize=5.5, color="0.45",
                        xytext=(3, 0), textcoords="offset points")
        for m in [0.0, 0.25, 0.5, 0.75]:
            pts = [to_xy(*theory(p, m)) for p in ts]
            ax.plot([x for x, _ in pts], [y for _, y in pts], color="0.75", lw=0.7,
                    ls="--", zorder=1)
            mid = pts[18]
            ax.annotate(f"m={m:g}", mid, fontsize=5.5, color="0.45",
                        xytext=(-2, -7), textcoords="offset points")
    ax.set_xlim(-0.12, 1.14)
    ax.set_ylim(-0.12, 0.99)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------- 実測の読み込みと組成 ----------
comp = {}      # (fam,label) -> (U,S,D) 全 (seed,idx) 平均 = 全岩点
cloud = {}     # (fam,label) -> Counter{(u3,s3,d3): 問題数}（問題ごと 3 seed の格子組成 = 薄片）
floor_acc = {}
have = []
for fam, label, sub in MODELS:
    sc = load(sub, 3, 0)
    if not sc:
        continue
    have.append((fam, label))
    fl = load(sub, 1, 0)
    if fl:
        oks = [r["ok"] for recs in fl.values() for r in recs.values()]
        floor_acc[(fam, label)] = sum(oks) / len(oks)
    tot = Counter()
    per_idx = defaultdict(Counter)
    for se in SEEDS:
        if se not in sc:
            continue
        for i, r in sc[se].items():
            k = classify(r["rounds"][0]["preds"])
            tot[k] += 1
            per_idx[i][k] += 1
    n = sum(tot.values())
    comp[(fam, label)] = (tot["u"] / n, tot["s"] / n, tot["d"] / n)
    cloud[(fam, label)] = Counter((c["u"], c["s"], c["d"]) for c in per_idx.values())

print(f"data_root: {DATA}")
print(f"モデル: {len(have)}/8 — " + ", ".join(f"{f} {l}" for f, l in have))

# ---------- 組成表 + 実効 (p̂, m̂) の逆算（全岩。記述のみ） ----------
print("\n=== 合意三角図の組成（ラベルフリー）と実効 (p̂, m̂)（単一優勢誤答+塵モデルでの逆算）===")
print(f"  {'モデル':<16} {'一致U':>6} {'割れS':>6} {'三様D':>6} {'ā(=U+S/3)':>10} "
      f"{'床[既公表]':>10} {'p̂':>5} {'m̂':>5}")
GRID = [i / 200 for i in range(201)]
for fam, label in have:
    u, s, d = comp[(fam, label)]
    abar = u + s / 3
    best = min(((p, m) for p in GRID for m in GRID),
               key=lambda pm: (theory(*pm)[0] - u) ** 2 + (theory(*pm)[2] - d) ** 2)
    fl = floor_acc.get((fam, label))
    print(f"  {fam+' '+label:<16} {u:6.3f} {s:6.3f} {d:6.3f} {abar:10.3f} "
          f"{fl if fl is not None else float('nan'):10.3f} {best[0]:5.2f} {best[1]:5.2f}")
print("  ※ p̂/m̂ は問題混合の実効値（全岩分析）。問題ごとの p のばらつきは畳み込まれている。")

# ---------- 図 1: 全岩点 + 理論グリッド ----------
fig, ax = plt.subplots(figsize=(7.5, 6.5))
draw_frame(ax, grid=True)
for fam, (color, marker) in FAM_STYLE.items():
    ms = [m for m in have if m[0] == fam]
    if not ms:
        continue
    xs, ys = zip(*[to_xy(*comp[m]) for m in ms])
    ax.scatter(xs, ys, c=color, marker=marker, s=70, zorder=3, label=fam,
               edgecolors="white", linewidths=0.8)
    for m, x, y in zip(ms, xs, ys):
        fl = floor_acc.get(m)
        note = f"{m[1]}" + (f" (floor {fl:.2f})" if fl is not None else "")
        ax.annotate(note, (x, y), fontsize=6.5, color="0.15",
                    xytext=(6, 4), textcoords="offset points")
ax.legend(fontsize=8, loc="upper left", frameon=False)
ax.set_title("Agreement ternary — SC3 3-vote outcomes (label-free)", fontsize=11)
fig.text(0.5, 0.015,
         "grid: single-dominant-wrong-answer + dust error model "
         "(solid: iso-floor p, dashed: iso-dominance m; pair-collision c = m²) — descriptive, no tests",
         ha="center", fontsize=6.5, color="0.4")
fig.tight_layout()
fig.savefig(os.path.join(IMG, "fig_ternary_main.png"), dpi=150)

# ---------- 図 2: モデル別の薄片クラウド（問題ごと 3 seed の格子組成） ----------
fig, axes = plt.subplots(3, 3, figsize=(11, 9.5))
for ax, m in zip(axes.flat, have):
    draw_frame(ax, grid=False)
    # 参考に iso-p 3 本だけ薄く
    ts = [i / 60 for i in range(61)]
    for p in [0.3, 0.5, 0.7]:
        pts = [to_xy(*theory(p, mm)) for mm in ts]
        ax.plot([x for x, _ in pts], [y for _, y in pts], color="0.85", lw=0.6, zorder=1)
    color, marker = FAM_STYLE[m[0]]
    total = sum(cloud[m].values())
    for (cu, cs, cd), n in sorted(cloud[m].items()):
        se = cu + cs + cd  # 通常 3（seed 欠損時はそれ未満）
        x, y = to_xy(cu / se, cs / se, cd / se)
        ax.scatter([x], [y], s=8 + 600 * n / total, c=color, alpha=0.45, zorder=2,
                   edgecolors="none")
    x, y = to_xy(*comp[m])
    ax.scatter([x], [y], c="0.1", marker="*", s=90, zorder=4)
    fl = floor_acc.get(m)
    ax.set_title(f"{m[0]} {m[1]}" + (f"  floor={fl:.3f}" if fl is not None else ""), fontsize=9)
for ax in axes.flat[len(have):]:
    ax.axis("off")
fig.suptitle("Per-problem agreement compositions (3 seeds/problem, bubble = #problems; ★ = model mean)",
             y=0.995, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "fig_ternary_models.png"), dpi=150)
print(f"\nwrote {IMG}/fig_ternary_{{main,models}}.png")
