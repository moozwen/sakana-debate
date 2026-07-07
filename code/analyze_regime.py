# analyze_regime.py — Part 1b（能力相対レジーム同定）の事前登録解析
# 事前登録: vault/01_Inbox/04_Crafts/sakana_problem/part1b-regime-preregistration.md（追補 A1 含む）
# 出すもの:
#   (1) サイズ×条件 accuracy（3seed平均±95%CI）
#   (2) 討論 vs SC / SC vs 床 の McNemar（各サイズ、Holm はファミリ内 debate>sc に適用）
#   (3) ラウンド推移・フリップ・全会一致誤答率・format失敗率（raw から事後計算）
#   (4) レジーム表 + 図3実行ゲート判定（追補 A1.3）
#   (5) fig_regime.png（追補 A1.1 の 2×2 レジーム図。横軸=床精度）
#       fig_regime_raw.png（元 §5 のサイズ軸 3曲線＋デルタ。補助図）
# 使い方: cd sakana-debate && python code/analyze_regime.py
import json
import os
import random
import re
import sys
from collections import Counter
from math import comb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from grading import _NUM, grade

# (family, label, size_B, outdir)。Gemma はゲート成立時のみ実行（A1.3）— 未実行なら自動スキップ
SIZES = [("Qwen2.5", "0.5B", 0.5, "results_regime/05b"),
         ("Qwen2.5", "1.5B", 1.5, "results_regime/15b"),
         ("Qwen2.5", "3B", 3.0, "results_regime/3b"),
         ("Qwen2.5", "7B", 7.0, "results_regime/7b"),
         ("Gemma3", "1B", 1.0, "results_regime/gemma1b"),
         ("Gemma3", "4B", 4.0, "results_regime/gemma4b")]
CONDS = {"floor": (1, 0), "sc": (3, 0), "debate": (3, 2)}
SEEDS = [1, 2, 3]
T95_3 = 4.303  # 自由度2の t 値（3 seed）
GATE_D = 0.03  # 図3ゲートの効果量しきい値（§3.2 の検出力設計と同値）

# --- format 失敗の分類（採点器と同じパターン階層。raw から事後計算） ---
_PATS = [("hash", re.compile(r"####\s*(" + _NUM + r")")),
         ("boxed", re.compile(r"\\boxed\{?\s*(" + _NUM + r")")),
         ("answer_is", re.compile(r"answer\s+is[^\d\-]{0,15}(" + _NUM + r")"))]


def extract_method(text):
    if not text:
        return "none"
    for name, pat in _PATS:
        if pat.findall(text):
            return name
    return "fallback" if re.findall(_NUM, text) else "none"


def majority(preds):
    votes = Counter(p for p in preds if p is not None)
    if not votes:
        return None
    top = votes.most_common(1)[0][1]
    tied = [v for v, c in votes.items() if c == top]
    if len(tied) == 1:
        return tied[0]
    for p in preds:
        if p in tied:
            return p


def load(outdir, n, r):
    out = {}
    for s in SEEDS:
        fn = os.path.join(outdir, f"debate_N{n}_R{r}_seed{s}.jsonl")
        if os.path.exists(fn):
            out[s] = {j["idx"]: j for l in open(fn) if l.strip() for j in [json.loads(l)]}
    return out


def ci3(vals):
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return m, T95_3 * sd / len(vals) ** 0.5


def mcnemar(A, B):
    """A/B: {seed: {idx: rec}}。返り値 (Aのみ正解, Bのみ正解, 両側p)。"""
    b = c = 0
    for s in SEEDS:
        if s not in A or s not in B:
            continue
        for idx in A[s]:
            if idx not in B[s]:
                continue
            a_ok, b_ok = A[s][idx]["ok"], B[s][idx]["ok"]
            b += a_ok and not b_ok
            c += b_ok and not a_ok
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = min(1.0, sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n * 2)
    return b, c, p


rng = random.Random(0)


def paired_delta(A, B):
    """A−B のペア差平均と問題単位ブートストラップ 95%CI。ペアなしなら None。"""
    diffs = [A[s][i]["ok"] - B[s][i]["ok"]
             for s in SEEDS if s in A and s in B for i in A[s] if i in B[s]]
    if not diffs:
        return None
    d = sum(diffs) / len(diffs)
    boots = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(2000))
    return d, boots[49], boots[1949]


data = {}  # (family, label) -> cond_label -> {seed: {idx: rec}}
for fam, label, _, outdir in SIZES:
    data[(fam, label)] = {cl: load(outdir, n, r) for cl, (n, r) in CONDS.items()}

# ---------- (1) accuracy 表 ----------
print("=== accuracy（3seed 平均 ± 95%CI）===")
acc = {}
for fam, label, _, _ in SIZES:
    for cl in CONDS:
        per_seed = [sum(r["ok"] for r in recs.values()) / len(recs)
                    for recs in data[(fam, label)][cl].values() if recs]
        if per_seed:
            acc[(fam, label, cl)] = ci3(per_seed)
    row = "  ".join(f"{cl}={acc[(fam, label, cl)][0]:.3f}±{acc[(fam, label, cl)][1]:.3f}"
                    for cl in CONDS if (fam, label, cl) in acc)
    if row:
        print(f"  {fam} {label:>5}: {row}")

# ---------- (2) McNemar + Holm（ファミリ内 debate>sc。事前登録 §4・A1.3） ----------
print("\n=== McNemar（討論 vs SC / SC vs 床、各サイズ）===")
tests = []
for fam, label, _, _ in SIZES:
    D = data[(fam, label)]
    if D["debate"] and D["sc"]:
        tests.append((fam, label, "debate>sc", *mcnemar(D["debate"], D["sc"])))
    if D["sc"] and D["floor"]:
        tests.append((fam, label, "sc>floor", *mcnemar(D["sc"], D["floor"])))
holm = {}  # (fam, label, kind) -> 補正p。debate>sc / sc>floor それぞれファミリ内で補正
for fam in {t[0] for t in tests}:
    for kind in ("debate>sc", "sc>floor"):
        fam_tests = sorted([t for t in tests if t[0] == fam and t[2] == kind],
                           key=lambda t: t[5])
        for rank, t in enumerate(fam_tests):
            holm[(fam, t[1], kind)] = min(1.0, t[5] * (len(fam_tests) - rank))
for fam, label, kind, b, c, p in tests:
    print(f"  {fam} {label:>5} {kind}: +{b} / -{c}  p={p:.3f}  "
          f"Holm p={holm[(fam, label, kind)]:.3f}")

# ---------- (3) ラウンド推移・フリップ・全会一致誤答・format ----------
print("\n=== 討論（N3_R2）: ラウンド推移 / フリップ / unanimous-wrong / format ===")
for fam, label, _, _ in SIZES:
    D = data[(fam, label)]["debate"]
    if not D:
        continue
    n_rounds = 3
    accs, fg, fb, wrong, unan_wrong = [[] for _ in range(n_rounds)], 0, 0, 0, 0
    fmt = Counter()
    n_resp = 0
    for recs in D.values():
        for rnd in range(n_rounds):
            accs[rnd].append(sum(grade(majority(r["rounds"][rnd]["preds"]), r["gold"])
                                 for r in recs.values()) / len(recs))
        for r in recs.values():
            first = grade(majority(r["rounds"][0]["preds"]), r["gold"])
            fg += (not first) and r["ok"]
            fb += first and (not r["ok"])
            if not r["ok"]:
                wrong += 1
                unan_wrong += bool(r["unanimous"])
            for rnd in r["rounds"]:
                for t in rnd["raw"]:
                    fmt[extract_method(t)] += 1
                    n_resp += 1
    traj = " -> ".join(f"{sum(a)/len(a):.3f}" for a in accs)
    uw = f"{unan_wrong}/{wrong}" if wrong else "0/0"
    fallback = (fmt["fallback"] + fmt["none"]) / n_resp if n_resp else 0
    none_rate = fmt["none"] / n_resp if n_resp else 0
    print(f"  {fam} {label:>5}: {traj}  w->r:{fg} r->w:{fb}  unan-wrong:{uw}  "
          f"format逸脱(#### 不在):{fallback:.1%}  抽出不能:{none_rate:.1%}")

# ---------- (4) レジーム表（A1.1 の 4面の数値）＋図3ゲート判定（A1.3） ----------
print("\n=== レジーム表（横軸=床精度。図1/2/4 の数値）===")
reg = {}  # (fam, label) -> dict
for fam, label, size, _ in SIZES:
    D = data[(fam, label)]
    if not D["floor"]:
        continue
    floor_m, floor_h = acc[(fam, label, "floor")]
    r = {"size": size, "floor": floor_m, "floor_ci": floor_h,
         "sc_d": paired_delta(D["sc"], D["floor"]) if D["sc"] else None,
         "db_d": paired_delta(D["debate"], D["sc"]) if D["debate"] and D["sc"] else None}
    # 図4: SC 条件（N3_R0）の誤答中 unanimous 率
    w = uw = 0
    for recs in D["sc"].values():
        for rec in recs.values():
            if not rec["ok"]:
                w += 1
                uw += bool(rec["unanimous"])
    r["uw"] = (uw / w, uw, w) if w else None
    reg[(fam, label)] = r
    sc_s = f"{r['sc_d'][0]:+.3f} [{r['sc_d'][1]:+.3f},{r['sc_d'][2]:+.3f}]" if r["sc_d"] else "—"
    db_s = f"{r['db_d'][0]:+.3f} [{r['db_d'][1]:+.3f},{r['db_d'][2]:+.3f}]" if r["db_d"] else "—"
    uw_s = f"{r['uw'][0]:.1%} ({r['uw'][1]}/{r['uw'][2]})" if r["uw"] else "—"
    print(f"  {fam} {label:>5}: 床={floor_m:.3f}  SC−床={sc_s}  討論−SC={db_s}  "
          f"unan-wrong(SC)={uw_s}")

gate_hits = [label for (fam, label), r in reg.items()
             if fam == "Qwen2.5" and r["sc_d"] and r["sc_d"][0] >= GATE_D
             and holm.get((fam, label, "sc>floor"), 1.0) < 0.05]
print(f"\n図3実行ゲート（A1.3: SC>床 Holm p<0.05 かつ SC−床 ≥ +{GATE_D:.0%}）: "
      + (f"成立（{', '.join(gate_hits)}）→ Gemma-3 1B/4B を実行" if gate_hits
         else "不成立 → 図3（Gemma）は実行しない"))

# ---------- (5) 判定図 ----------
BANDS = [(0.0, 0.35, "collapse"), (0.40, 0.85, "mid"), (0.90, 1.0, "saturated")]
FAM_STYLE = {"Qwen2.5": ("tab:blue", "o"), "Gemma3": ("tab:orange", "D")}


def band_axes(ax):
    for lo, hi, name in BANDS:
        ax.axvspan(lo, hi, color="gray", alpha=0.08)
        ax.text((lo + hi) / 2, 0.98, name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7, color="gray")
    ax.set_xlim(0, 1)
    ax.axhline(0, ls="--", lw=1, color="gray")


def plot_delta(ax, key, fams):
    for fam in fams:
        color, marker = FAM_STYLE[fam]
        pts = sorted((r["floor"], r["floor_ci"], *r[key])
                     for (f, _), r in reg.items() if f == fam and r[key])
        if not pts:
            continue
        x, xe, d, lo, hi = zip(*pts)
        ax.errorbar(x, d, xerr=xe, yerr=[[m - l for m, l in zip(d, lo)],
                                         [h - m for m, h in zip(d, hi)]],
                    fmt=marker + ("-" if fam == "Qwen2.5" else ""), color=color,
                    capsize=3, ms=5, label=fam)
    band_axes(ax)


fig, axes = plt.subplots(2, 2, figsize=(11, 8))
ax1, ax2, ax3, ax4 = axes.flat

plot_delta(ax1, "sc_d", ["Qwen2.5"])
ax1.set_ylabel("SC − floor (paired, 95% bootstrap CI)")
ax1.set_title("Fig1  SC gain vs capability — predicted: asymmetric inverted-U")

plot_delta(ax2, "db_d", FAM_STYLE)
ax2.set_ylabel("debate − SC (paired)")
ax2.set_title("Fig2  debate − SC — predicted: ≤ 0 everywhere")

plot_delta(ax3, "sc_d", FAM_STYLE)
ax3.set_ylabel("SC − floor (paired)")
ax3.set_xlabel("single-agent floor accuracy (N1_R0)")
ax3.set_title("Fig3  cross-family universality (Gemma overlaid, gated)")
ax3.legend(fontsize=8)

for fam, (color, marker) in FAM_STYLE.items():
    pts = sorted((r["floor"], r["uw"][0]) for (f, _), r in reg.items()
                 if f == fam and r["uw"])
    if pts:
        x, y = zip(*pts)
        ax4.plot(x, y, marker + ("-" if fam == "Qwen2.5" else ""), color=color, ms=5)
band_axes(ax4)
ax4.set_ylim(0, 1)
ax4.set_ylabel("unanimous share of wrong answers (SC)")
ax4.set_xlabel("single-agent floor accuracy (N1_R0)")
ax4.set_title("Fig4  mechanism: correlated errors — predicted: rising")

fig.suptitle("capability-relative regime figure (preregistered, addendum A1)", y=0.995)
fig.tight_layout()
fig.savefig("fig_regime.png", dpi=150)
print("wrote fig_regime.png")

# 補助図: 元 §5 のサイズ軸（Qwen のみ）
fig, (bx1, bx2) = plt.subplots(2, 1, figsize=(6, 6.5), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
xs = [s for f, _, s, _ in SIZES if f == "Qwen2.5"]
for cl, style in [("floor", "s--"), ("sc", "o-"), ("debate", "^-")]:
    pts = [(s, *acc[(f, l, cl)]) for f, l, s, _ in SIZES
           if f == "Qwen2.5" and (f, l, cl) in acc]
    if pts:
        px, pm, ph = zip(*pts)
        bx1.errorbar(px, pm, yerr=ph, fmt=style, capsize=3, label=cl)
bx1.set_xscale("log")
bx1.set_xticks(xs, [l for f, l, _, _ in SIZES if f == "Qwen2.5"])
bx1.set_ylabel("GSM8K accuracy (500 problems)")
bx1.legend()
bx1.set_title("raw view: floor / SC / debate vs model size (Qwen2.5)")
for (f, l), r in reg.items():
    if f == "Qwen2.5" and r["db_d"]:
        d, lo, hi = r["db_d"]
        bx2.errorbar([r["size"]], [d], yerr=[[d - lo], [hi - d]], fmt="o",
                     color="tab:red", capsize=3)
bx2.axhline(0, ls="--", lw=1, color="gray")
bx2.set_xlabel("model size (B params, log)")
bx2.set_ylabel("debate − SC")
fig.tight_layout()
fig.savefig("fig_regime_raw.png", dpi=150)
print("wrote fig_regime_raw.png")
