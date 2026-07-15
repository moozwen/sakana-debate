# fit_ternary_mixture.py — 合意三角図の混合フィット（p の 2 点混合 + 共有 m、4 パラメータ最尤）
# 定式化: vault/01_Inbox/04_Crafts/sakana_problem/slm-ternary-reading-guide-20260714.md 付録 A.5
# 背景:   同 §4 — 均質 (p, m) 逆算は問題の二峰性に棄却された（p̂→0 縮退）。本スクリプトはその宿題。
#
# 性格: 記述・検定なし・登録外の探索フィット。S1-D として登録する場合は本出力の閲覧範囲を開示すること。
#
# モデル:
#   問題 i の正解率 p_i ~ π·δ(p_hi) + (1−π)·δ(p_lo)（2 点混合）、誤答は「優勢誤答 m + 塵」を全問題で共有。
#   3 seed の帰結カウント (n_u, n_s, n_d)（和=3）の尤度:
#     L = Π_i [ π·Mult(n_i | θ(p_hi, m)) + (1−π)·Mult(n_i | θ(p_lo, m)) ],  θ = (U, S, D)（付録 A.2）
#   問題は 10 格子セルにしか落ちないので、尤度はセル別ヒストグラムの重み付き和で計算する。
#
# 出すもの（モデル別）:
#   π̂, p̂_hi, p̂_lo, m̂ / 含意床 E[p] = π p_hi + (1−π) p_lo（← 既公表の実測床との突合が主検証）/
#   含意 ā / 均質フィットとの ΔAIC（正 = 混合の勝ち）/ m の近似プロファイル区間（ΔlogL ≤ 2）/
#   図: fig_ternary_mixfit.png（10 セルの観測ヒストグラム vs フィット期待値、3×3 面）
#
# 使い方: cd sakana-debate && python code/fit_ternary_mixture.py [data_root]   （数十秒〜数分）
import json
import math
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
EPS = 1e-12

# 10 格子セル（3 seed の帰結カウント (n_u, n_s, n_d)、和 = 3）と多項係数
CELLS = [(a, b, 3 - a - b) for a in range(3, -1, -1) for b in range(3 - a, -1, -1)]
COEF = [6 // (math.factorial(a) * math.factorial(b) * math.factorial(c)) for a, b, c in CELLS]


def load(sub, n, r):
    out = {}
    for se in SEEDS:
        fn = os.path.join(DATA, sub, f"debate_N{n}_R{r}_seed{se}.jsonl")
        if os.path.exists(fn):
            out[se] = {j["idx"]: j for l in open(fn) if l.strip() for j in [json.loads(l)]}
    return out


def classify(preds):
    cnt = Counter(p for p in preds if p is not None)
    top = max(cnt.values()) if cnt else 0
    return "u" if top == 3 else ("s" if top == 2 else "d")


def theory(p, m):
    """付録 A.2: U/S/D（単一優勢誤答 + 塵）。"""
    u = p ** 3 + (1 - p) ** 3 * m ** 3
    d = 3 * p * (1 - p) ** 2 * (1 - m ** 2) + (1 - p) ** 3 * (1 - 3 * m ** 2 + 2 * m ** 3)
    return u, 1 - u - d, d


def cell_probs(p, m):
    u, s, d = (max(EPS, v) for v in theory(p, m))
    return [k * u ** a * s ** b * d ** c for k, (a, b, c) in zip(COEF, CELLS)]


def loglik_mix(hist, pi, ph, pl, m):
    Ph, Pl = cell_probs(ph, m), cell_probs(pl, m)
    return sum(n * math.log(max(EPS, pi * Ph[j] + (1 - pi) * Pl[j]))
               for j, n in hist)


def fit_model(hist):
    """粗い格子 → 局所細分化 2 回。ph ≥ pl を強制（ラベル入れ替えの排除）。"""
    PS = [i * 0.025 for i in range(41)]
    MS = [i * 0.05 for i in range(21)]
    PIS = [0.05 + i * 0.075 for i in range(13)]
    best = (-1e18, None)
    for m in MS:
        probs = {p: cell_probs(p, m) for p in PS}
        for hi in range(len(PS)):
            Ph = probs[PS[hi]]
            for lo in range(hi + 1):
                Pl = probs[PS[lo]]
                for pi in PIS:
                    ll = sum(n * math.log(max(EPS, pi * Ph[j] + (1 - pi) * Pl[j]))
                             for j, n in hist)
                    if ll > best[0]:
                        best = (ll, (pi, PS[hi], PS[lo], m))
    ll, (pi, ph, pl, m) = best
    step = 0.025
    for _ in range(3):  # 局所細分化（±step を 1/5 刻みで）
        step /= 5
        cand = [(pi + i * step * 3, ph + j * step, pl + k * step, m + l * step * 2)
                for i in range(-2, 3) for j in range(-2, 3)
                for k in range(-2, 3) for l in range(-2, 3)]
        for c_pi, c_ph, c_pl, c_m in cand:
            c_pi = min(1 - EPS, max(EPS, c_pi))
            c_ph, c_pl = min(1.0, max(0.0, c_ph)), min(1.0, max(0.0, c_pl))
            c_m = min(1.0, max(0.0, c_m))
            if c_pl > c_ph:
                continue
            l2 = loglik_mix(hist, c_pi, c_ph, c_pl, c_m)
            if l2 > ll:
                ll, (pi, ph, pl, m) = l2, (c_pi, c_ph, c_pl, c_m)
    return ll, pi, ph, pl, m


def fit_homog(hist):
    best = (-1e18, None)
    for p in (i * 0.01 for i in range(101)):
        for m in (i * 0.02 for i in range(51)):
            P = cell_probs(p, m)
            ll = sum(n * math.log(max(EPS, P[j])) for j, n in hist)
            if ll > best[0]:
                best = (ll, (p, m))
    return best


def profile_m(hist, pi, ph, pl, ll_max):
    """m を固定し (π, p_hi, p_lo) を局所再適合した近似プロファイル。ΔlogL ≤ 2 の m 範囲を返す。"""
    ok = []
    for m in (i * 0.05 for i in range(21)):
        ll, (cpi, cph, cpl) = -1e18, (pi, ph, pl)
        step = 0.05
        for _ in range(3):
            for dpi in (-2, -1, 0, 1, 2):
                for dh in (-2, -1, 0, 1, 2):
                    for dl in (-2, -1, 0, 1, 2):
                        t_pi = min(1 - EPS, max(EPS, cpi + dpi * step))
                        t_ph = min(1.0, max(0.0, cph + dh * step))
                        t_pl = min(1.0, max(0.0, cpl + dl * step))
                        if t_pl > t_ph:
                            continue
                        l2 = loglik_mix(hist, t_pi, t_ph, t_pl, m)
                        if l2 > ll:
                            ll, (cpi, cph, cpl) = l2, (t_pi, t_ph, t_pl)
            step /= 3
        if ll >= ll_max - 2.0:
            ok.append(m)
    return (min(ok), max(ok)) if ok else (None, None)


print(f"data_root: {DATA}")
print("\n=== 混合フィット（2 点混合 p + 共有 m、最尤）— 記述・登録外 ===")
print(f"  {'モデル':<16} {'π̂':>5} {'p̂hi':>5} {'p̂lo':>5} {'m̂':>5} {'m範囲(ΔlL≤2)':>13} "
      f"{'E[p]':>6} {'床[既公表]':>10} {'ā(fit)':>7} {'ā(obs)':>7} {'ΔAIC':>7}")
results = {}
for fam, label, sub in MODELS:
    sc = load(sub, 3, 0)
    if not sc:
        continue
    fl = load(sub, 1, 0)
    floor = None
    if fl:
        oks = [r["ok"] for recs in fl.values() for r in recs.values()]
        floor = sum(oks) / len(oks)
    per_idx = defaultdict(Counter)
    for se in SEEDS:
        if se not in sc:
            continue
        for i, r in sc[se].items():
            per_idx[i][classify(r["rounds"][0]["preds"])] += 1
    cellcnt = Counter()
    skipped = 0
    for i, c in per_idx.items():
        if sum(c.values()) != 3:  # seed 欠損問題は除外（和=3 の前提を保つ）
            skipped += 1
            continue
        cellcnt[(c["u"], c["s"], c["d"])] += 1
    hist = [(j, cellcnt.get(cell, 0)) for j, cell in enumerate(CELLS) if cellcnt.get(cell, 0)]
    n_prob = sum(n for _, n in hist)
    obs = Counter()
    for j, n in hist:
        a, b, c = CELLS[j]
        obs["u"] += a * n
        obs["s"] += b * n
        obs["d"] += c * n
    a_obs = (obs["u"] + obs["s"] / 3) / (3 * n_prob)

    ll_mix, pi, ph, pl, m = fit_model(hist)
    ll_hom, (p_h, m_h) = fit_homog(hist)
    d_aic = (2 * 2 - 2 * ll_hom) - (2 * 4 - 2 * ll_mix)  # 正 = 混合の勝ち
    m_lo, m_hi = profile_m(hist, pi, ph, pl, ll_mix)
    ep = pi * ph + (1 - pi) * pl
    a_fit = pi * (ph ** 2 + (1 - ph) ** 2 * m ** 2) + (1 - pi) * (pl ** 2 + (1 - pl) ** 2 * m ** 2)
    results[(fam, label)] = dict(hist=hist, n=n_prob, pi=pi, ph=ph, pl=pl, m=m,
                                 ll=ll_mix, floor=floor)
    mr = f"[{m_lo:.2f},{m_hi:.2f}]" if m_lo is not None else "—"
    warn = " ⚠m弱識別" if m_lo is not None and m_hi - m_lo > 0.5 else ""
    sk = f"（欠損除外 {skipped} 問）" if skipped else ""
    print(f"  {fam+' '+label:<16} {pi:5.2f} {ph:5.2f} {pl:5.2f} {m:5.2f} {mr:>13} "
          f"{ep:6.3f} {floor if floor is not None else float('nan'):10.3f} "
          f"{a_fit:7.3f} {a_obs:7.3f} {d_aic:7.1f}{warn}{sk}")
print("  ※ E[p] = π·p_hi + (1−π)·p_lo（ラベル不使用の含意床）。床列は既公表の実測（突合用）。")
print("  ※ ΔAIC = 均質 − 混合（正 = 混合が単純さのペナルティ込みで優位）。")
print("  ※ m̂ は p_lo 側（誤答が出る問題）でのみ識別される。範囲が広いモデルでは読まないこと。")

# ---------- 図: 観測ヒストグラム vs フィット期待値 ----------
fig, axes = plt.subplots(3, 3, figsize=(13, 9.5))
labels = ["".join(str(x) for x in cell) for cell in CELLS]  # "300" = (u3,s0,d0) など
for ax, (mkey, res) in zip(axes.flat, results.items()):
    color = FAM_STYLE[mkey[0]][0]
    obs_n = {j: n for j, n in res["hist"]}
    xs = range(len(CELLS))
    ax.bar(xs, [obs_n.get(j, 0) for j in xs], color=color, alpha=0.6, label="observed")
    Ph, Pl = cell_probs(res["ph"], res["m"]), cell_probs(res["pl"], res["m"])
    exp_n = [res["n"] * (res["pi"] * Ph[j] + (1 - res["pi"]) * Pl[j]) for j in xs]
    ax.plot(xs, exp_n, "k.-", ms=6, lw=1, label="mixture fit")
    ax.set_xticks(list(xs), labels, fontsize=6, rotation=45)
    ax.set_title(f"{mkey[0]} {mkey[1]}  (pi={res['pi']:.2f}, m={res['m']:.2f})", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
for ax in axes.flat[len(results):]:
    ax.axis("off")
axes.flat[0].legend(fontsize=7)
fig.suptitle("Ternary mixture fit — observed per-problem cell counts vs fitted expectation\n"
             "(cells 'usd' = #seeds unanimous/split/3-way, e.g. 300 = unanimous in all 3 seeds)",
             y=0.995, fontsize=10)
fig.supylabel("#problems")
fig.tight_layout()
fig.savefig(os.path.join(IMG, "fig_ternary_mixfit.png"), dpi=150)
print(f"\nwrote {IMG}/fig_ternary_mixfit.png")
