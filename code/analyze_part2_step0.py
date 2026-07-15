# analyze_part2_step0.py — Part 2 Step 0（ラベルフリー・レジーム推定の kill-screen）事前登録解析
# 事前登録: vault/01_Inbox/04_Crafts/sakana_problem/phase_2/part2-adaptive-preregistration.md
# 手順書:   同 phase_2/part2-adaptive-guide.md（実装細目はそちらで固定）
# 生成ゼロ・ローカル CPU のみ。⚠ 実行 = 数値の閲覧なので sign-off ①〜⑥ 後に実データで実行すること。
#
# 出すもの:
#   (0) 構成 assert（床 = SC agent0 / 討論 R0 = SC の一致率。破れても停止しない — 登録済み
#       フォールバック: プールは N3_R0 のみ。1c 追補 A2 の既知挙動）
#   (A1) 較正曲線: 8 点の ā（平均ペア一致率、ラベルフリー）vs 床精度。Spearman ρ（n=8 正確
#        permutation、片側）。LOO 床予測 MAE（線形内挿・端クランプ）。副次: 1b ゲート LOO 分類（記述）
#   (A2) 各モデル: m ∈ {1,2,3} → SC 多数決正誤の AUC（Mann-Whitney、タイ 0.5）。
#        問題単位ブートストラップ B=10,000 の 95%CI
#   (A3) ポリシー・パレート: 9 プール（N3_R0 の 3seed×3agent）から問題ごと 1,000 順列の逐次抽選。
#        固定 N∈{1,3,5,7,9} / 早期打ち切り「先に 2 票一致で停止、上限 C∈{3,5,9}」。
#        タイ処理は run_debate.majority と同一（抽選順最先）。判定は C3 vs 固定 N3（P3）
#   (A4) seed 整合エスカレーション: スクリーニング = seed s の SC agent{0,1}（一致 = 両方非 None
#        かつ等値）。不一致 → (a) 同 seed 討論最終多数決 / (b) 9 プール多数決（コスト完全一致 9）。
#        McNemar、8 モデル Holm。⚠ 相互作用帰属には使わない（1c が正 — 事前登録 §4 従属規則）
#   (A5) 探索的: 同調率 c（R0 少数派の R1 多数派乗り換え率）と churn。形式検定なし
#   図 3 枚 → <data_root>/img/fig_step0_{calibration,pareto,escalation}.png
#   末尾: P1〜P5 と K1/K2 の判定文（8 モデル揃わないうちは暫定表示）
#
# 使い方: cd sakana-debate && python code/analyze_part2_step0.py [data_root]
#   data_root 省略時 = experiments/phase1b_1c_regime_budget/results（実データ）。合成データでのスモークテスト用に差し替え可。
import json
import os
import random
import sys
from collections import Counter, defaultdict
from itertools import permutations as iperm
from math import comb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from grading import grade

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "experiments/phase1b_1c_regime_budget/results")
IMG = os.path.join(DATA, "img")
os.makedirs(IMG, exist_ok=True)

MODELS = [("Qwen2.5", "0.5B", "05b"), ("Qwen2.5", "1.5B", "15b"),
          ("Qwen2.5", "3B", "3b"), ("Qwen2.5", "7B", "7b"),
          ("Gemma3", "1B", "gemma1b"), ("Gemma3", "4B", "gemma4b"),
          ("Llama3.2", "1B", "llama1b"), ("Llama3.2", "3B", "llama3b")]
BAND = {("Qwen2.5", "0.5B"), ("Gemma3", "1B"), ("Llama3.2", "1B"), ("Qwen2.5", "1.5B")}
# Part 1b 確定値のハードコード: ゲート通過（SC−床 ≥ +3pt & Holm 有意）と 討論−SC の有意符号
GATE_1B = {("Qwen2.5", "0.5B"): 1, ("Qwen2.5", "1.5B"): 1, ("Qwen2.5", "3B"): 1,
           ("Qwen2.5", "7B"): 0, ("Gemma3", "1B"): 1, ("Gemma3", "4B"): 0,
           ("Llama3.2", "1B"): 1, ("Llama3.2", "3B"): 1}
SIGN_1B = {("Gemma3", "1B"): +1, ("Qwen2.5", "0.5B"): +1, ("Llama3.2", "1B"): -1}
FAM_STYLE = {"Qwen2.5": ("tab:blue", "o"), "Gemma3": ("tab:orange", "D"),
             "Llama3.2": ("tab:green", "s")}
SEEDS = [1, 2, 3]
# ④ で固定した kill 閾値（sign-off 後は変更不可）
RHO_MAIN, RHO_KILL, AUC_KILL, DROP_MAX, SAVE_MIN = 0.8, 0.7, 0.55, 0.010, 0.10
B_BOOT = 10000   # 問題単位ブートストラップ
N_PERM = 1000    # A3 の抽選順列/問題
FIXED_NS = [1, 3, 5, 7, 9]
CAPS = [3, 5, 9]
rng = random.Random(0)  # 乱数固定（再実行で再現）


def load(sub, n, r):
    out = {}
    for s in SEEDS:
        fn = os.path.join(DATA, sub, f"debate_N{n}_R{r}_seed{s}.jsonl")
        if os.path.exists(fn):
            out[s] = {j["idx"]: j for l in open(fn) if l.strip() for j in [json.loads(l)]}
    return out


def boot_ci(groups, stat, b=B_BOOT):
    """groups: 問題単位のリスト。stat(list_of_groups) -> float。95%CI を返す。"""
    vals = sorted(stat(rng.choices(groups, k=len(groups))) for _ in range(b))
    return vals[int(b * 0.025)], vals[int(b * 0.975) - 1]


data = {}
for fam, label, sub in MODELS:
    data[(fam, label)] = {"floor": load(sub, 1, 0), "sc": load(sub, 3, 0),
                          "debate": load(sub, 3, 2)}
have = [(f, l) for f, l, _ in MODELS if data[(f, l)]["sc"] and data[(f, l)]["floor"]]
print(f"data_root: {DATA}")
print(f"モデル: {len(have)}/8 — " + ", ".join(f"{f} {l}" for f, l in have))
if len(have) < 8:
    print("⚠ 8 点未満: P1/K1 と主張レベル対応表の確定は不可（暫定表示のみ）")

# ---------- (0) 構成 assert（一致率の報告。停止しない — 事前登録 §3 フォールバック） ----------
print("\n=== 構成確認: リクエスト seed 共有条件の R0 予測一致率（1c 追補 A2 の既知挙動）===")
for fam, label in have:
    D = data[(fam, label)]
    mf = tf = md = td = 0
    for s in SEEDS:
        if s not in D["sc"]:
            continue
        for i, r in D["sc"][s].items():
            p3 = r["rounds"][0]["preds"]
            if s in D["floor"] and i in D["floor"][s]:
                tf += 1
                mf += D["floor"][s][i]["rounds"][0]["preds"][0] == p3[0]
            if s in D["debate"] and i in D["debate"][s]:
                pd = D["debate"][s][i]["rounds"][0]["preds"]
                td += len(p3)
                md += sum(a == b for a, b in zip(pd, p3))
    fl = f"床 {mf/tf:.1%}" if tf else "床 —"
    db = f"討論R0 {md/td:.1%}" if td else "討論R0 —"
    print(f"  {fam} {label:>5}: {fl}  {db}   → プールは N3_R0 のみ（登録どおり、影響なし）")

# ---------- 信号の計算（gold 不使用 — このブロックにラベルを混入させないこと） ----------
def pair_agree(preds):
    """3 票の一致ペア割合 ∈ {0, 1/3, 1}。両方非 None かつ等値のみ一致（実装細目、手順書 §0.2）。"""
    return sum(preds[i] is not None and preds[i] == preds[j]
               for i, j in ((0, 1), (0, 2), (1, 2))) / 3


def modal_count(preds):
    c = Counter(p for p in preds if p is not None)
    return c.most_common(1)[0][1] if c else 0


sig = {}   # (fam,label) -> dict(abar, floor_acc, events=[(idx, m, a, ok)])
for fam, label in have:
    D = data[(fam, label)]
    ev = [(i, modal_count(r["rounds"][0]["preds"]), pair_agree(r["rounds"][0]["preds"]),
           r["ok"]) for s in SEEDS if s in D["sc"] for i, r in D["sc"][s].items()]
    fl = [r["ok"] for s in SEEDS if s in D["floor"] for r in D["floor"][s].values()]
    sig[(fam, label)] = {"abar": sum(a for _, _, a, _ in ev) / len(ev),
                         "floor": sum(fl) / len(fl), "events": ev}

# ---------- (A1) 較正曲線: ā vs 床（P1） ----------
def rankdata(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    rk = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            rk[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return rk


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = (sum((v - mx) ** 2 for v in x)) ** 0.5
    sy = (sum((v - my) ** 2 for v in y)) ** 0.5
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx * sy else 0.0


print("\n=== A1: 較正（ā → 床精度、ラベルフリー配備診断）— P1 ===")
pts = sorted((sig[m]["abar"], sig[m]["floor"], m) for m in have)
for ab, fl, (fam, label) in pts:
    print(f"  {fam} {label:>5}: ā={ab:.3f}  床={fl:.3f}")
rho = p_perm = None
if len(pts) >= 4:
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    rx, ry = rankdata(xs), rankdata(ys)
    rho = pearson(rx, ry)
    n_ge = total = 0
    for pm in iperm(range(len(ry))):          # n=8 → 40,320 通りの正確 permutation
        total += 1
        n_ge += pearson(rx, [ry[i] for i in pm]) >= rho - 1e-12
    p_perm = n_ge / total                      # 片側（正の単調を予測 — §2 P1）
    # LOO 床予測（他 7 点の ā ソート線形内挿、端はクランプ）
    errs = []
    for k in range(len(pts)):
        others = [p for j, p in enumerate(pts) if j != k]
        ab, fl, _ = pts[k]
        if ab <= others[0][0]:
            pred = others[0][1]
        elif ab >= others[-1][0]:
            pred = others[-1][1]
        else:
            for (a0, f0, _), (a1, f1, _) in zip(others, others[1:]):
                if a0 <= ab <= a1:
                    pred = f0 + (f1 - f0) * (ab - a0) / (a1 - a0) if a1 > a0 else (f0 + f1) / 2
                    break
        errs.append(abs(pred - fl))
    # 副次（記述のみ）: ā ≤ θ → 1b ゲート通過、の LOO 分類
    loo_ok = 0
    for k in range(len(pts)):
        others = [(p[0], GATE_1B[p[2]]) for j, p in enumerate(pts) if j != k]
        cands = [(others[i][0] + others[i + 1][0]) / 2 for i in range(len(others) - 1)]
        best_th = max(cands, key=lambda th: sum((ab <= th) == g for ab, g in others))
        loo_ok += (pts[k][0] <= best_th) == GATE_1B[pts[k][2]]
    print(f"  Spearman ρ = {rho:.3f}（正確 permutation 片側 p = {p_perm:.4f}）  "
          f"LOO 床予測 MAE = {sum(errs)/len(errs):.3f}")
    print(f"  副次（記述）: 1b ゲート LOO 分類 {loo_ok}/{len(pts)} 正解")

# ---------- (A2) インスタンスレベル AUC（P2） ----------
print("\n=== A2: m（3 票の最頻値票数）→ SC 多数決正誤の AUC — P2 ===")
auc_res = {}
for fam, label in have:
    # idx -> 8 カウント（pos の m=0..3, neg の m=0..3）。スコアは 4 値なので総和から AUC が閉形式
    per_idx = defaultdict(lambda: [0] * 8)
    for i, m, _, ok in sig[(fam, label)]["events"]:
        per_idx[i][m if ok else 4 + m] += 1
    vecs = list(per_idx.values())

    def auc_from(tot):
        pos, neg = tot[:4], tot[4:]
        P, N = sum(pos), sum(neg)
        if not P or not N:
            return 0.5
        num = sum(pos[s] * neg[t] for s in range(4) for t in range(s))
        num += 0.5 * sum(pos[s] * neg[s] for s in range(4))
        return num / (P * N)

    tot0 = [sum(v[a] for v in vecs) for a in range(8)]
    a = auc_from(tot0)
    bs = []
    for _ in range(B_BOOT):
        mult = Counter(rng.choices(range(len(vecs)), k=len(vecs)))
        tot = [0] * 8
        for j, mlt in mult.items():
            vj = vecs[j]
            for k in range(8):
                tot[k] += vj[k] * mlt
        bs.append(auc_from(tot))
    bs.sort()
    lo, hi = bs[int(B_BOOT * 0.025)], bs[int(B_BOOT * 0.975) - 1]
    auc_res[(fam, label)] = (a, lo, hi)
    band = "（帯内 — P2 判定対象）" if (fam, label) in BAND else ""
    print(f"  {fam} {label:>5}: AUC = {a:.3f} [CI {lo:.3f}, {hi:.3f}]{band}")

# ---------- (A3) ポリシーシミュレーション（P3） ----------
print("\n=== A3: 9 プールの配分ポリシー（1,000 順列/問。タイ = 抽選順最先）— P3 ===")
pol = {}  # (fam,label) -> idx -> {"fixed": {N: acc}, "es": {C: (acc, cost)}}
for fam, label in have:
    D = data[(fam, label)]["sc"]
    pools = defaultdict(list)
    gold = {}
    for s in SEEDS:
        if s not in D:
            continue
        for i, r in D[s].items():
            pools[i].extend(r["rounds"][0]["preds"])
            gold[i] = r["gold"]
    res = {}
    for i, vals in pools.items():
        if len(vals) != 9:
            continue
        okmap = {v: grade(v, gold[i]) for v in set(vals) if v is not None}
        fx = {N: 0 for N in FIXED_NS}
        es = {C: [0, 0] for C in CAPS}
        for _ in range(N_PERM):
            perm = vals[:]
            rng.shuffle(perm)
            counts, first = {}, {}
            best = None
            stop_v = stop_t = None
            pref = {}
            for t, v in enumerate(perm, 1):
                if v is not None:
                    c = counts[v] = counts.get(v, 0) + 1
                    if v not in first:
                        first[v] = t
                    if best is None or c > counts[best] or (c == counts[best]
                                                            and first[v] < first[best]):
                        best = v
                    if c == 2 and stop_v is None:
                        stop_v, stop_t = v, t
                if t in fx:
                    pref[t] = best
            for N in FIXED_NS:
                fx[N] += okmap.get(pref[N], False)
            for C in CAPS:
                if stop_t is not None and stop_t <= C:
                    es[C][0] += okmap.get(stop_v, False)
                    es[C][1] += stop_t
                else:
                    es[C][0] += okmap.get(pref[C], False)
                    es[C][1] += C
        res[i] = {"fixed": {N: k / N_PERM for N, k in fx.items()},
                  "es": {C: (k / N_PERM, c / N_PERM) for C, (k, c) in es.items()}}
    pol[(fam, label)] = res
    fxs = "  ".join(f"N{N}={sum(r['fixed'][N] for r in res.values())/len(res):.3f}"
                    for N in FIXED_NS)
    ess = "  ".join(f"C{C}={sum(r['es'][C][0] for r in res.values())/len(res):.3f}"
                    f"@{sum(r['es'][C][1] for r in res.values())/len(res):.2f}calls"
                    for C in CAPS)
    print(f"  {fam} {label:>5}: 固定 {fxs}")
    print(f"          早期打切 {ess}")

# P3 判定（プール、層化ブートストラップ: モデル内で問題を resample）
p3_verdict = {}
if pol:
    print("  --- P3 判定（プール、C vs 固定 N=C。drop = 固定 − 早期打切）---")
    for C in CAPS:
        per_model = {m: [(r["fixed"][C] - r["es"][C][0], r["es"][C][1])
                         for r in pol[m].values()] for m in pol}

        def stat(quant):
            def f(_):
                tot_d = tot_c = tot_n = 0
                for m, rows in per_model.items():
                    pick = rng.choices(rows, k=len(rows))
                    tot_d += sum(d for d, _ in pick)
                    tot_c += sum(c for _, c in pick)
                    tot_n += len(pick)
                return (tot_d / tot_n) if quant == "drop" else (tot_c / tot_n)
            return f

        n_all = sum(len(v) for v in per_model.values())
        drop = sum(d for rows in per_model.values() for d, _ in rows) / n_all
        cost = sum(c for rows in per_model.values() for _, c in rows) / n_all
        dvals = sorted(stat("drop")(None) for _ in range(B_BOOT))
        d_hi = dvals[int(B_BOOT * 0.975) - 1]
        ok = d_hi < DROP_MAX and cost <= C * (1 - SAVE_MIN)
        p3_verdict[C] = ok
        print(f"    C={C}: drop={drop*100:+.2f}pt (CI上限 {d_hi*100:.2f}pt)  "
              f"cost={cost:.2f}/{C}  {'✓ 成立' if ok else '✗ 不成立'}"
              f"{'（P3 本命の登録ポリシー）' if C == 3 else ''}")

# ---------- (A4) seed 整合エスカレーション（P4。⚠ 相互作用帰属には使わない — 1c が正） ----------
print("\n=== A4: エスカレーション 討論(a) vs SC9(b)（コスト完全一致）— P4 整合チェック ===")
esc = {}
tests = []
for fam, label in have:
    D = data[(fam, label)]
    if not D["debate"]:
        continue
    pools = defaultdict(list)
    for s in SEEDS:
        if s not in D["sc"]:
            continue
        for i, r in D["sc"][s].items():
            pools[i].extend(r["rounds"][0]["preds"])  # seed 昇順 × agent 昇順の固定順

    def pool_maj(i):
        vals = pools[i]
        cnt = Counter(v for v in vals if v is not None)
        if not cnt:
            return None
        top = cnt.most_common(1)[0][1]
        tied = {v for v, c in cnt.items() if c == top}
        for v in vals:
            if v in tied:
                return v

    b = c = n = agree_n = 0
    for s in SEEDS:
        if s not in D["sc"] or s not in D["debate"]:
            continue
        for i, r in D["sc"][s].items():
            if i not in D["debate"][s] or len(pools[i]) != 9:
                continue
            n += 1
            p0, p1 = r["rounds"][0]["preds"][:2]
            if p0 is not None and p0 == p1:  # スクリーニング一致 → 両腕同一（コスト 2）
                agree_n += 1
                continue
            ok_a = D["debate"][s][i]["ok"]
            ok_b = grade(pool_maj(i), r["gold"])
            b += ok_a and not ok_b
            c += ok_b and not ok_a
    nd = b + c
    p = min(1.0, sum(comb(nd, k) for k in range(min(b, c) + 1)) / 2 ** nd * 2) if nd else 1.0
    tests.append([fam, label, b, c, p, n, agree_n])
for rank, t in enumerate(sorted(tests, key=lambda t: t[4])):
    t.append(min(1.0, t[4] * (len(tests) - rank)))
for fam, label, b, c, p, n, agree_n, hp in tests:
    esc[(fam, label)] = (b, c, hp)
    exp = SIGN_1B.get((fam, label))
    tag = ""
    if exp is not None:
        match = (b > c) if exp > 0 else (c > b)
        tag = f"  1b 符号({'+' if exp > 0 else '−'}): {'一致 ✓' if match else '不一致 ✗ → §5-4 構成疑い'}"
    print(f"  {fam} {label:>5}: 討論勝ち +{b} / SC9勝ち -{c}  Holm p={hp:.3f}  "
          f"(一致即答 {agree_n}/{n}){tag}")

# ---------- (A5) 探索的: 同調率・churn（P5。形式検定なし） ----------
print("\n=== A5（探索的）: 同調率 c と churn（討論 R0→R1）===")
conf = {}
for fam, label in have:
    D = data[(fam, label)]["debate"]
    if not D:
        continue
    minority = adopt = agents = churn = 0
    for s in SEEDS:
        if s not in D:
            continue
        for r in D[s].values():
            p0, p1 = r["rounds"][0]["preds"], r["rounds"][1]["preds"]
            cnt = Counter(p for p in p0 if p is not None)
            if not cnt:
                continue
            top = cnt.most_common(1)[0][1]
            tied = [v for v, c2 in cnt.items() if c2 == top]
            modal = tied[0] if len(tied) == 1 else None  # 最頻値が一意のときだけ評価
            for a in range(len(p0)):
                agents += 1
                churn += p1[a] != p0[a]
                if modal is not None and p0[a] != modal:
                    minority += 1
                    adopt += p1[a] == modal
    if agents:
        c_rate = adopt / minority if minority else 0.0
        conf[(fam, label)] = c_rate
        print(f"  {fam} {label:>5}: 同調率 c = {adopt}/{minority} = {c_rate:.1%}  "
              f"churn = {churn/agents:.1%}")

# ---------- 図 3 枚 ----------
# 1) 較正
fig, ax = plt.subplots(figsize=(6, 4.5))
for fam, (color, marker) in FAM_STYLE.items():
    ps = [(sig[m]["abar"], sig[m]["floor"], m[1]) for m in have if m[0] == fam]
    if ps:
        ax.scatter([p[0] for p in ps], [p[1] for p in ps], c=color, marker=marker,
                   s=60, label=fam)
        for ab, fl, lb in ps:
            ax.annotate(lb, (ab, fl), fontsize=7, xytext=(4, 3),
                        textcoords="offset points")
ax.set_xlabel("mean pairwise agreement ā (label-free, SC N3)")
ax.set_ylabel("floor accuracy (N1_R0)")
ax.set_title(f"Step0 A1: label-free calibration"
             + (f"  Spearman ρ={rho:.2f} (p={p_perm:.3f})" if rho is not None else ""))
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "fig_step0_calibration.png"), dpi=150)

# 2) パレート（モデル別 8 面 + プール 1 面）
fig, axes = plt.subplots(3, 3, figsize=(13, 10))
panels = [(m, pol[m]) for m in have if m in pol]
pooled = defaultdict(list)
for _, res in panels:
    for r in res.values():
        for N in FIXED_NS:
            pooled[("f", N)].append(r["fixed"][N])
        for C in CAPS:
            pooled[("e", C)].append(r["es"][C])
for ax, item in zip(axes.flat, panels + [("pooled", None)]):
    if item[1] is not None:
        (fam, label), res = item
        fx = [(N, sum(r["fixed"][N] for r in res.values()) / len(res)) for N in FIXED_NS]
        es = [(sum(r["es"][C][1] for r in res.values()) / len(res),
               sum(r["es"][C][0] for r in res.values()) / len(res), C) for C in CAPS]
        ax.set_title(f"{fam} {label}", fontsize=9)
    else:
        fx = [(N, sum(pooled[("f", N)]) / len(pooled[("f", N)])) for N in FIXED_NS]
        es = [(sum(c for _, c in pooled[("e", C)]) / len(pooled[("e", C)]),
               sum(a for a, _ in pooled[("e", C)]) / len(pooled[("e", C)]), C) for C in CAPS]
        ax.set_title("pooled (all settings)", fontsize=9)
    ax.plot([n for n, _ in fx], [a for _, a in fx], "o-", color="tab:blue", ms=4,
            label="fixed N")
    for cost, a, C in es:
        ax.plot([cost], [a], "*", color="tab:red", ms=11)
        ax.annotate(f"C{C}", (cost, a), fontsize=7, xytext=(3, -9),
                    textcoords="offset points", color="tab:red")
    ax.grid(alpha=0.3)
for ax in axes.flat[len(panels) + 1:]:
    ax.axis("off")
axes.flat[0].legend(fontsize=8)
fig.suptitle("Step0 A3: accuracy vs calls — fixed-N vs early-stop (first-to-2)", y=0.995)
fig.supxlabel("mean calls / problem")
fig.supylabel("accuracy")
fig.tight_layout()
fig.savefig(os.path.join(IMG, "fig_step0_pareto.png"), dpi=150)

# 3) エスカレーション
if esc:
    fig, ax = plt.subplots(figsize=(7, 4))
    ms = [m for m in have if m in esc]
    dv = [(esc[m][0] - esc[m][1]) for m in ms]
    cols = [FAM_STYLE[m[0]][0] for m in ms]
    ax.barh(range(len(ms)), dv, color=cols)
    ax.set_yticks(range(len(ms)), [f"{f} {l}" for f, l in ms], fontsize=8)
    ax.axvline(0, color="gray", lw=1)
    ax.set_xlabel("discordant pairs: debate-escalation wins − SC9-escalation wins")
    ax.set_title("Step0 A4: seed-coherent escalation (equal cost) — sign check vs Part 1b")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig_step0_escalation.png"), dpi=150)
print(f"\nwrote {IMG}/fig_step0_{{calibration,pareto,escalation}}.png")

# ---------- 判定（事前登録 §5 の機械適用） ----------
print("\n=== 事前登録 §5 判定 " + ("" if len(have) == 8 else "【暫定: 8 点未満】") + "===")
p1_ok = p1_dead = None
if rho is not None and len(have) == 8:
    p1_ok = rho >= RHO_MAIN and p_perm <= 0.05
    p1_dead = rho < RHO_KILL or p_perm > 0.05
    print(f"  P1: ρ={rho:.3f} (p={p_perm:.4f}) → "
          + ("本命成立（ρ≥0.8）" if p1_ok else
             ("不成立（ρ<0.7 または p>0.05）" if p1_dead else "中間（0.7≤ρ<0.8）— 弱い成立")))
band_have = [m for m in BAND if m in auc_res]
p2_pass = [m for m in band_have if auc_res[m][1] > AUC_KILL]
p2_ok = p2_dead = None
if band_have:
    p2_ok = len(p2_pass) == len(band_have) == 4
    p2_dead = len(p2_pass) == 0
    print(f"  P2: 帯内 AUC CI 下限 > {AUC_KILL}: {len(p2_pass)}/{len(band_have)} 点 → "
          + ("本命成立" if p2_ok else ("全滅" if p2_dead else
             f"部分成立（CL-B は {', '.join(f'{f} {l}' for f, l in p2_pass)} に限定 — §5-3）")))
if p1_dead is not None and p2_dead is not None:
    if p1_dead and p2_dead:
        print("  ★ K1 発火: ラベルフリー推定に信号なし → Part 2 撤収（§5-1）")
    else:
        p3_any = any(p3_verdict.values()) if p3_verdict else False
        if p3_verdict:
            print(f"  P3: 登録ポリシー族 {'成立あり' if p3_any else '全滅'} → "
                  + ("CL-B 存続" if p3_any else "★ K2 発火: CL-A（配備診断）のみで Step 1 へ（§5-2）"))
        row = {(False, True): "最大主張（in-sample 限定 → Step 2 で確証）",
               (False, False): "配備レベル診断のみ — Step 1 は為替表のみ",
               (True, True): "配分信号は問題単位でのみ有効（想定薄 — 追補で考察）",
               (True, False): "撤収（K1）"}[(bool(p1_dead), bool(p2_ok or p3_any))]
        print(f"  → 主張レベル対応表の行: 「{row}」")
if conf:
    top = max(conf, key=conf.get)
    print(f"  P5（探索的）: 同調率最大 = {top[0]} {top[1]}（{conf[top]:.1%}）"
          + (" — Llama 1B 予測どおり（示唆止まり）" if top == ("Llama3.2", "1B") else ""))
print("  ※ 全結論に「Part 1b データ上の / in-sample」を冠する（⑥）。A4 は相互作用帰属に使わない（1c が正）")
