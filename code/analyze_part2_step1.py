# analyze_part2_step1.py — Part 2 Step 1（為替レート表・lead-k 停止則・家系混成多数決）事前登録解析
# 事前登録: vault/01_Inbox/04_Crafts/sakana_problem/phase_2/part2-step1-preregistration.md
# 手順書:   同 phase_2/part2-step1-guide.md（実装細目はそちらで固定）
# 生成ゼロ・ローカル CPU のみ。⚠ 実行 = 数値の閲覧（S1-B lead-k / S1-C 混成は未閲覧量）。
#   sign-off ①〜⑥ 前に実データで実行しないこと。py_compile のみ可。
#
# 出すもの:
#   (0) 構成確認（Step 0 と同一の一致率報告。停止しない — 1c 追補 A2 の既知挙動）
#   (B) 停止則: 9 プール（N3_R0 の 3seed×3agent）、問題ごと 1,000 順列の逐次抽選。
#       固定 N∈{1,3,5,7,9} / first-to-2 C∈{3,5,9} / lead-k（首位−次点 ≥ k で停止）k∈{2,3}×cap∈{5,9}。
#       判定は lead-2/cap-9 vs 固定 N9 のみ（PB1。閾値は Step 0 ④ と同一）。PB2 = vs first-to-2 C9（方向）
#   (A) 為替レート表: 床/SC3/自己再考/適応C3/討論9/SC9 の精度・コール数。推奨 = 点推定 argmax +
#       1c/1b 既登録 Holm 判定のハードコード SIG_1C で確定/暫定マーク（新規検定なし — 登録 ②）
#   (C) 混成多数決: 帯内 1B 級トリオの 3×3（9 コール）vs 単独 9 票 ×3（Holm-3、PC1/PC2）。
#       3 コール版は副次（別ファミリ Holm-3）。家系内/間 誤答一致率（PC3、記述）。
#       探索的（隔離）: ā 重み付き投票・ā 選択単独
#   図 3 枚 → <data_root>/img/fig_step1_{exchange,pareto,mixed}.png
#
# 使い方: cd sakana-debate && python code/analyze_part2_step1.py [data_root]
import json
import os
import random
import sys
from collections import Counter, defaultdict

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
CORE5 = {("Qwen2.5", "0.5B"), ("Qwen2.5", "1.5B"), ("Qwen2.5", "3B"),
         ("Gemma3", "1B"), ("Llama3.2", "1B")}
TRIO = [("Qwen2.5", "0.5B"), ("Gemma3", "1B"), ("Llama3.2", "1B")]  # 帯内 1B 級（登録 §3）
FAM_STYLE = {"Qwen2.5": ("tab:blue", "o"), "Gemma3": ("tab:orange", "D"),
             "Llama3.2": ("tab:green", "s")}
CONDS = {"floor": (1, 0), "sc3": (3, 0), "self": (1, 2), "debate": (3, 2), "sc9": (9, 0)}
CALLS = {"floor": 1, "sc3": 3, "self": 3, "debate": 9, "sc9": 9}
SEEDS = [1, 2, 3]
B_BOOT = 10000
N_PERM = 1000
FIXED_NS = [1, 3, 5, 7, 9]
FT2_CAPS = [3, 5, 9]
LEAD_KS = [2, 3]
LEAD_CAPS = [5, 9]
MAIN_LEAD = (2, 9)                 # PB1 の判定対象（登録 ③ — これ以外は曲線表示のみ）
DROP_MAX, SAVE_MIN = 0.010, 0.10   # Step 0 ④ と同一（変更なし）
FT2_C9_DROP_KNOWN = 0.0140         # PB2 の参照: Step 0 実測の first-to-2 C9 drop（既知・閲覧済み）
rng = random.Random(0)             # 乱数固定（再実行で再現）

# 1c/1b の既登録 Holm 判定のハードコード（確定/暫定マーク用。新規検定はしない — 登録 ②）
# 予算段 -> (勝者cond, Holm 有意か)。出典: part1c-budget-results-20260713 §2（P3 と P2）
SIG_1C = {("Qwen2.5", "0.5B"): {3: ("sc3", True), 9: ("sc9", True)},
          ("Qwen2.5", "1.5B"): {3: ("sc3", False), 9: ("sc9", True)},
          ("Gemma3", "1B"): {3: ("self", True), 9: ("debate", True)},
          ("Llama3.2", "1B"): {3: ("sc3", True), 9: ("sc9", True)},
          ("Qwen2.5", "3B"): {3: ("sc3", True), 9: ("sc9", True)}}


def load(sub, n, r):
    out = {}
    for s in SEEDS:
        fn = os.path.join(DATA, sub, f"debate_N{n}_R{r}_seed{s}.jsonl")
        if os.path.exists(fn):
            out[s] = {j["idx"]: j for l in open(fn) if l.strip() for j in [json.loads(l)]}
    return out


def acc_all(D):
    oks = [r["ok"] for recs in D.values() for r in recs.values()]
    return sum(oks) / len(oks) if oks else None


def boot_vals(groups, b=B_BOOT):
    """groups: 問題単位の値リスト。resample 平均の分布（ソート済み）を返す。"""
    out = []
    for _ in range(b):
        pick = rng.choices(groups, k=len(groups))
        out.append(sum(pick) / len(pick))
    out.sort()
    return out


def boot_p_ci(groups):
    """平均・95%CI・両側ブートストラップ p（+1 平滑化）。"""
    d = sum(groups) / len(groups)
    bs = boot_vals(groups)
    lo, hi = bs[int(B_BOOT * 0.025)], bs[int(B_BOOT * 0.975) - 1]
    n_le = sum(v <= 0 for v in bs)
    n_ge = sum(v >= 0 for v in bs)
    p = min(1.0, 2 * min((n_le + 1) / (B_BOOT + 1), (n_ge + 1) / (B_BOOT + 1)))
    return d, lo, hi, p


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out = [0.0] * len(ps)
    mx = 0.0
    for r, i in enumerate(order):
        mx = max(mx, min(1.0, ps[i] * (len(ps) - r)))
        out[i] = mx
    return out


data = {}
for fam, label, sub in MODELS:
    data[(fam, label)] = {cl: load(sub, n, r) for cl, (n, r) in CONDS.items()}
have = [(f, l) for f, l, _ in MODELS if data[(f, l)]["sc3"] and data[(f, l)]["floor"]]
print(f"data_root: {DATA}")
print(f"モデル: {len(have)}/8 — " + ", ".join(f"{f} {l}" for f, l in have))

# ---------- (0) 構成確認（Step 0 と同一。報告のみ、停止しない） ----------
print("\n=== 構成確認: リクエスト seed 共有条件の R0 予測一致率（既知挙動 — 1c 追補 A2）===")
for fam, label in have:
    D = data[(fam, label)]
    mf = tf = md = td = 0
    for s in SEEDS:
        if s not in D["sc3"]:
            continue
        for i, r in D["sc3"][s].items():
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

# ---------- (B) 停止則シミュレーション（PB1/PB2） ----------
print("\n=== S1-B: 停止則（固定 N / first-to-2 / lead-k、1,000 順列/問）===")
LEAD_COMBOS = [(k, C) for k in LEAD_KS for C in LEAD_CAPS]
pol = {}  # m -> idx -> {"fixed":{N:frac}, "ft2":{C:(frac,cost)}, "lead":{(k,C):(frac,cost)}}
for fam, label in have:
    D = data[(fam, label)]["sc3"]
    pools, gold = defaultdict(list), {}
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
        ft2 = {C: [0, 0] for C in FT2_CAPS}
        ld = {kc: [0, 0] for kc in LEAD_COMBOS}
        for _ in range(N_PERM):
            perm = vals[:]
            rng.shuffle(perm)
            counts, first = {}, {}
            best = None
            c1 = c2 = 0                       # 首位/次点の票数（先出現タイ規則つき増分追跡）
            ft2_v = ft2_t = None
            lstop = {}                        # k -> (解答, 停止時刻)。cap に依存しない（手順書 §1）
            pref = {}
            for t, v in enumerate(perm, 1):   # None もコールは消費（票には数えない）
                if v is not None:
                    c = counts[v] = counts.get(v, 0) + 1
                    if v not in first:
                        first[v] = t
                    if best is None:
                        best, c1 = v, c
                    elif v == best:
                        c1 = c
                    elif c > c1 or (c == c1 and first[v] < first[best]):
                        best, c2, c1 = v, c1, c
                    elif c > c2:
                        c2 = c
                    if c == 2 and ft2_v is None:
                        ft2_v, ft2_t = v, t
                    lead = c1 - c2
                    for k in LEAD_KS:
                        if k not in lstop and lead >= k:
                            lstop[k] = (best, t)
                pref[t] = best
            for N in FIXED_NS:
                fx[N] += okmap.get(pref[N], False)
            for C in FT2_CAPS:
                if ft2_t is not None and ft2_t <= C:
                    ft2[C][0] += okmap.get(ft2_v, False)
                    ft2[C][1] += ft2_t
                else:
                    ft2[C][0] += okmap.get(pref[C], False)
                    ft2[C][1] += C
            for k, C in LEAD_COMBOS:
                if k in lstop and lstop[k][1] <= C:
                    ld[(k, C)][0] += okmap.get(lstop[k][0], False)
                    ld[(k, C)][1] += lstop[k][1]
                else:
                    ld[(k, C)][0] += okmap.get(pref[C], False)
                    ld[(k, C)][1] += C
        res[i] = {"fixed": {N: n / N_PERM for N, n in fx.items()},
                  "ft2": {C: (a / N_PERM, c / N_PERM) for C, (a, c) in ft2.items()},
                  "lead": {kc: (a / N_PERM, c / N_PERM) for kc, (a, c) in ld.items()}}
    pol[(fam, label)] = res
    nn = len(res)
    fxs = "  ".join(f"N{N}={sum(r['fixed'][N] for r in res.values())/nn:.3f}"
                    for N in FIXED_NS)
    fts = "  ".join(f"C{C}={sum(r['ft2'][C][0] for r in res.values())/nn:.3f}"
                    f"@{sum(r['ft2'][C][1] for r in res.values())/nn:.2f}"
                    for C in FT2_CAPS)
    lds = "  ".join(f"L{k}/{C}={sum(r['lead'][(k,C)][0] for r in res.values())/nn:.3f}"
                    f"@{sum(r['lead'][(k,C)][1] for r in res.values())/nn:.2f}"
                    for k, C in LEAD_COMBOS)
    print(f"  {fam} {label:>5}: 固定 {fxs}")
    print(f"          first-to-2 {fts}")
    print(f"          lead-k     {lds}")

# PB1 判定（プール、モデル内層化ブートストラップ — Step 0 P3 と同一機構）
pb1_ok = None
if pol:
    k0, C0 = MAIN_LEAD
    per_model = {m: [(r["fixed"][C0] - r["lead"][(k0, C0)][0], r["lead"][(k0, C0)][1])
                     for r in pol[m].values()] for m in pol}
    n_all = sum(len(v) for v in per_model.values())
    drop = sum(d for rows in per_model.values() for d, _ in rows) / n_all
    cost = sum(c for rows in per_model.values() for _, c in rows) / n_all
    dvals = []
    for _ in range(B_BOOT):
        tot_d = tot_n = 0
        for rows in per_model.values():
            pick = rng.choices(rows, k=len(rows))
            tot_d += sum(d for d, _ in pick)
            tot_n += len(pick)
        dvals.append(tot_d / tot_n)
    dvals.sort()
    d_hi = dvals[int(B_BOOT * 0.975) - 1]
    pb1_ok = d_hi < DROP_MAX and cost <= C0 * (1 - SAVE_MIN)
    print(f"\n  --- PB1 判定（lead-{k0}/cap-{C0} vs 固定 N{C0}、プール）---")
    print(f"    drop={drop*100:+.2f}pt (CI上限 {d_hi*100:.2f}pt < 1.0pt?)  "
          f"cost={cost:.2f}/{C0} (≤{C0*(1-SAVE_MIN):.1f}?)  → {'✓ 成立' if pb1_ok else '✗ 不成立'}")
    print(f"    PB2（方向のみ）: lead drop {drop*100:.2f}pt vs first-to-2 C9 drop "
          f"{FT2_C9_DROP_KNOWN*100:.2f}pt（既知） → {'✓ 減少' if drop < FT2_C9_DROP_KNOWN else '✗'}")

# ---------- (A) 為替レート表（記述のみ。新規検定なし — 登録 ②） ----------
print("\n=== S1-A: 為替レート表（GSM8K in-sample。全行この限定つき）===")
print(f"  {'モデル':<16} {'床(1)':>7} {'SC3(3)':>7} {'self(3)':>8} {'適応C3(~2.4)':>13} "
      f"{'討論(9)':>8} {'SC9(9)':>7}")
table = {}
for fam, label in have:
    D = data[(fam, label)]
    accs = {cl: acc_all(D[cl]) for cl in CONDS}
    r = pol.get((fam, label), {})
    nn = len(r)
    c3a = sum(v["ft2"][3][0] for v in r.values()) / nn if nn else None
    c3c = sum(v["ft2"][3][1] for v in r.values()) / nn if nn else None
    if accs["sc9"] is None and nn:  # 拡張 3 モデル: SC9 は 9 プール多数決の sim 値
        accs["sc9"] = sum(v["fixed"][9] for v in r.values()) / nn
        accs["_sc9sim"] = True
    table[(fam, label)] = dict(accs, c3=(c3a, c3c))
    f = lambda x: f"{x:.3f}" if x is not None else "—"
    sim = " sim" if accs.get("_sc9sim") else ""
    print(f"  {fam+' '+label:<16} {f(accs['floor']):>7} {f(accs['sc3']):>7} "
          f"{f(accs['self']):>8} {f(c3a):>7}@{c3c:.2f}    {f(accs['debate']):>7} "
          f"{f(accs['sc9']):>6}{sim}")

print("\n  --- 推奨（機械的規則: 点推定 argmax。確定 = 1c/1b 既登録 Holm 有意のハードコード）---")
for fam, label in have:
    if (fam, label) not in CORE5:
        continue
    t = table[(fam, label)]
    recs = []
    for budget, cands in [(3, ["sc3", "self"]), (9, ["debate", "sc9"])]:
        win = max(cands, key=lambda c: t[c])
        reg = SIG_1C[(fam, label)][budget]
        if reg[0] != win:
            mark = "⚠ argmax≠登録勝者 → 暫定"
        else:
            mark = "確定" if reg[1] else "暫定"
        recs.append(f"予算{budget}: {win}({t[win]:.3f}) [{mark}]")
    ex13 = (max(t["sc3"], t["self"]) - t["floor"]) / 2 * 100
    ex39 = (max(t["debate"], t["sc9"]) - max(t["sc3"], t["self"])) / 6 * 100
    print(f"  {fam} {label:>5}: " + "  ".join(recs)
          + f"  限界為替 1→3: {ex13:+.1f}pt/コール, 3→9: {ex39:+.1f}pt/コール")

# ---------- (C) 混成多数決（PC1/PC2/PC3） ----------
print("\n=== S1-C: 家系混成多数決（帯内 1B 級トリオ、1,000 反復/問）===")
trio_ok = all(m in pol and data[m]["sc3"] for m in TRIO)
mix_res = None
if trio_ok:
    pools, gold = {}, {}
    for m in TRIO:
        D = data[m]["sc3"]
        pools[m] = defaultdict(list)
        for s in SEEDS:
            if s not in D:
                continue
            for i, r in D[s].items():
                pools[m][i].extend(r["rounds"][0]["preds"])
                gold[i] = r["gold"]
    common = [i for i in gold if all(len(pools[m][i]) == 9 for m in TRIO)]

    def pair_agree(preds):
        return sum(preds[i] is not None and preds[i] == preds[j]
                   for i, j in ((0, 1), (0, 2), (1, 2))) / 3

    abar = {}
    for m in TRIO:  # ā（ラベルフリー）— 探索的 ā 選択/重み付き用に再計算
        D = data[m]["sc3"]
        ev = [pair_agree(r["rounds"][0]["preds"]) for s in SEEDS if s in D
              for r in D[s].values()]
        abar[m] = sum(ev) / len(ev)

    def pool_maj(vals):  # 固定順（seed 昇順 × agent 昇順）、タイ = プール順最先 — Step 0 A4 と同一
        cnt = Counter(v for v in vals if v is not None)
        if not cnt:
            return None
        top = cnt.most_common(1)[0][1]
        tied = {v for v, c in cnt.items() if c == top}
        for v in vals:
            if v in tied:
                return v

    def draw_maj(vals):  # 無作為順リスト、タイ = 抽選順最先
        cnt = Counter(v for v in vals if v is not None)
        if not cnt:
            return None
        top = cnt.most_common(1)[0][1]
        tied = {v for v, c in cnt.items() if c == top}
        for v in vals:
            if v in tied:
                return v

    single9 = {m: {} for m in TRIO}
    for m in TRIO:
        for i in common:
            single9[m][i] = grade(pool_maj(pools[m][i]), gold[i])
    mix9, mix3, wmix9 = {}, {}, {}
    for i in common:
        gmaps = {m: {v: grade(v, gold[i]) for v in set(pools[m][i]) if v is not None}
                 for m in TRIO}
        ok9 = ok3 = okw = 0
        for _ in range(N_PERM):
            samp = {m: rng.sample(pools[m][i], 3) for m in TRIO}
            comb = [v for m in TRIO for v in samp[m]]
            rng.shuffle(comb)
            v9 = draw_maj(comb)
            ok9 += any(gmaps[m].get(v9, False) for m in TRIO) if v9 is not None else 0
            ones = [samp[m][0] for m in TRIO]
            rng.shuffle(ones)
            v3 = draw_maj(ones)
            ok3 += any(gmaps[m].get(v3, False) for m in TRIO) if v3 is not None else 0
            # 探索的: ā 重み付き（票の重み = そのモデルの ā。タイ → 素の票数 → 抽選順最先）
            sc = defaultdict(float)
            for m in TRIO:
                for v in samp[m]:
                    if v is not None:
                        sc[v] += abar[m]
            if sc:
                mx = max(sc.values())
                cand = {v for v, s in sc.items() if s >= mx - 1e-12}
                if len(cand) > 1:
                    cnt = Counter(v for v in comb if v in cand)
                    mtop = cnt.most_common(1)[0][1]
                    cand = {v for v in cand if cnt[v] == mtop}
                vw = next(v for v in comb if v in cand)
                okw += any(gmaps[m].get(vw, False) for m in TRIO)
        mix9[i], mix3[i], wmix9[i] = ok9 / N_PERM, ok3 / N_PERM, okw / N_PERM

    acc9 = {m: sum(single9[m].values()) / len(common) for m in TRIO}
    accm9 = sum(mix9.values()) / len(common)
    best = max(TRIO, key=lambda m: acc9[m])
    print("  単独 9 票（プール多数決）: "
          + "  ".join(f"{f} {l}={acc9[(f,l)]:.3f}" for f, l in TRIO))
    print(f"  混成 3×3 = {accm9:.3f}   最良単独（機械的 argmax）= {best[0]} {best[1]}")

    # 9 コールファミリ: 混成 − 各単独（Holm-3）
    stats9 = {}
    for m in TRIO:
        groups = [mix9[i] - single9[m][i] for i in common]
        stats9[m] = boot_p_ci(groups)
    hp = holm([stats9[m][3] for m in TRIO])
    print("  --- 9 コールファミリ（混成 − 単独、問題ブートストラップ、Holm-3）---")
    pc1_row = None
    for m, h in zip(TRIO, hp):
        d, lo, hi, _ = stats9[m]
        tag = "（PC1 判定対象 = 最良単独）" if m == best else "（PC2）"
        print(f"    vs {m[0]} {m[1]:>5}: Δ={d*100:+.2f}pt [CI {lo*100:+.2f}, {hi*100:+.2f}] "
              f"Holm p={h:.4f} {tag}")
        if m == best:
            if lo > 0 and h < 0.05:
                pc1_row = "勝ち: 家系混成は budget-fair で最強単独を超える（in-sample）→ Step 2 昇格候補"
            elif hi < 0 and h < 0.05:
                pc1_row = "負け: 混成は最強単独に勝てない → 混成は Step 2 に持ち込まない（PC3 は法則データとして残す）"
            else:
                pc1_row = "非決定: 混成の価値は ā 選択への頑健性に限定して記述"
    print(f"    → PC1 分岐表の行: 「{pc1_row}」")

    # 3 コールファミリ（副次）: 混成 1×3 − 各単独固定 N3
    print("  --- 3 コールファミリ（副次、Holm-3）---")
    stats3 = {}
    for m in TRIO:
        groups = [mix3[i] - pol[m][i]["fixed"][3] for i in common if i in pol[m]]
        stats3[m] = boot_p_ci(groups)
    hp3 = holm([stats3[m][3] for m in TRIO])
    for m, h in zip(TRIO, hp3):
        d, lo, hi, _ = stats3[m]
        print(f"    vs {m[0]} {m[1]:>5} SC3: Δ={d*100:+.2f}pt [CI {lo*100:+.2f}, {hi*100:+.2f}] "
              f"Holm p={h:.4f}")

    # PC3: 誤答一致率（両方誤答のうち同一解答の割合。None は誤答だが等値にならない）
    print("  --- PC3: 誤答一致率（家系内 vs 家系間、閉形式 + 問題ブートストラップ CI）---")
    def wrong_counts(m, i):
        cnt = Counter(pools[m][i])          # None 含む
        wrong = {v: c for v, c in cnt.items()
                 if v is None or not grade(v, gold[i])}
        w_tot = sum(wrong.values())
        return wrong, w_tot

    pair_rates = {}
    keys = [(m, m) for m in TRIO] + [(TRIO[a], TRIO[b])
                                     for a in range(3) for b in range(a + 1, 3)]
    for X, Y in keys:
        per = []  # 問題ごと (eq_pairs, both_wrong_pairs)
        for i in common:
            wx, tx = wrong_counts(X, i)
            wy, ty = wrong_counts(Y, i)
            if X == Y:
                eq = sum(c * (c - 1) for v, c in wx.items() if v is not None)
                tot = tx * (tx - 1)
            else:
                eq = sum(c * wy.get(v, 0) for v, c in wx.items() if v is not None)
                tot = tx * ty
            per.append((eq, tot))
        num = sum(e for e, _ in per)
        den = sum(t for _, t in per)
        rate = num / den if den else 0.0
        bs = []
        for _ in range(B_BOOT):
            pick = rng.choices(per, k=len(per))
            tn = sum(t for _, t in pick)
            bs.append(sum(e for e, _ in pick) / tn if tn else 0.0)
        bs.sort()
        pair_rates[(X, Y)] = (rate, bs[int(B_BOOT * 0.025)], bs[int(B_BOOT * 0.975) - 1])
        kind = "家系内" if X == Y else "家系間"
        nm = f"{X[0]} {X[1]}" if X == Y else f"{X[0]} {X[1]} × {Y[0]} {Y[1]}"
        print(f"    {kind} {nm:<28}: {rate:.3f} [CI {bs[int(B_BOOT*0.025)]:.3f}, "
              f"{bs[int(B_BOOT*0.975)-1]:.3f}]")
    n_dir = ok_dir = 0
    for a in range(3):
        for b in range(a + 1, 3):
            X, Y = TRIO[a], TRIO[b]
            cr = pair_rates[(X, Y)][0]
            for W in (X, Y):
                n_dir += 1
                ok_dir += cr < pair_rates[(W, W)][0]
    print(f"    → PC3 方向（家系間 < 当事者の家系内）: {ok_dir}/{n_dir} 一致"
          + ("（本命どおり）" if ok_dir == n_dir else ""))

    # 探索的（隔離 — この節から出さない）
    accw = sum(wmix9.values()) / len(common)
    apick = max(TRIO, key=lambda m: abar[m])
    print("  --- 探索的（隔離）: ā 重み付き・ā 選択 ---")
    print(f"    ā: " + "  ".join(f"{f} {l}={abar[(f,l)]:.3f}" for f, l in TRIO))
    print(f"    ā 重み付き混成 = {accw:.3f}（素の混成 {accm9:.3f}）")
    print(f"    ā 選択単独 = {apick[0]} {apick[1]} の 9 票 = {acc9[apick]:.3f}"
          f"（ラベルフリー選択の失敗確認: 混成との差 {(accm9-acc9[apick])*100:+.1f}pt）")
    mix_res = (acc9, accm9, accw, best)
else:
    print("  ⚠ トリオのデータが揃っていない — S1-C はスキップ")

# ---------- 図 3 枚 ----------
# 1) 為替表
core = [m for m in have if m in CORE5]
if core:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    usages = [("floor", "floor (1)"), ("sc3", "SC3 (3)"), ("self", "self-refine (3)"),
              ("c3", "adaptive C3 (~2.4)"), ("debate", "debate (9)"), ("sc9", "SC9 (9)")]
    w = 0.13
    for u, (key, lab) in enumerate(usages):
        xs, ys = [], []
        for x, m in enumerate(core):
            v = table[m]["c3"][0] if key == "c3" else table[m].get(key)
            if v is not None:
                xs.append(x + (u - 2.5) * w)
                ys.append(v)
        ax.bar(xs, ys, width=w, label=lab)
    ax.set_xticks(range(len(core)), [f"{f}\n{l}" for f, l in core], fontsize=8)
    ax.set_ylabel("accuracy")
    ax.set_title("Step1 S1-A: exchange-rate table (GSM8K, in-sample)")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig_step1_exchange.png"), dpi=150)

# 2) パレート（8 面 + プール）
fig, axes = plt.subplots(3, 3, figsize=(13, 10))
panels = [(m, pol[m]) for m in have if m in pol]
pooled = defaultdict(list)
for _, res in panels:
    for r in res.values():
        for N in FIXED_NS:
            pooled[("f", N)].append(r["fixed"][N])
        for C in FT2_CAPS:
            pooled[("t", C)].append(r["ft2"][C])
        for kc in LEAD_COMBOS:
            pooled[("l", kc)].append(r["lead"][kc])
for ax, item in zip(axes.flat, panels + [("pooled", None)]):
    if item[1] is not None:
        (fam, label), res = item
        nn = len(res)
        fx = [(N, sum(r["fixed"][N] for r in res.values()) / nn) for N in FIXED_NS]
        ft = [(sum(r["ft2"][C][1] for r in res.values()) / nn,
               sum(r["ft2"][C][0] for r in res.values()) / nn, f"C{C}") for C in FT2_CAPS]
        le = [(sum(r["lead"][kc][1] for r in res.values()) / nn,
               sum(r["lead"][kc][0] for r in res.values()) / nn,
               f"L{kc[0]}/{kc[1]}") for kc in LEAD_COMBOS]
        ax.set_title(f"{fam} {label}", fontsize=9)
    else:
        fx = [(N, sum(pooled[("f", N)]) / len(pooled[("f", N)])) for N in FIXED_NS]
        ft = [(sum(c for _, c in pooled[("t", C)]) / len(pooled[("t", C)]),
               sum(a for a, _ in pooled[("t", C)]) / len(pooled[("t", C)]), f"C{C}")
              for C in FT2_CAPS]
        le = [(sum(c for _, c in pooled[("l", kc)]) / len(pooled[("l", kc)]),
               sum(a for a, _ in pooled[("l", kc)]) / len(pooled[("l", kc)]),
               f"L{kc[0]}/{kc[1]}") for kc in LEAD_COMBOS]
        ax.set_title("pooled (all settings)", fontsize=9)
    ax.plot([n for n, _ in fx], [a for _, a in fx], "o-", color="tab:blue", ms=4,
            label="fixed N")
    for cost, a, lb in ft:
        ax.plot([cost], [a], "*", color="tab:red", ms=10)
        ax.annotate(lb, (cost, a), fontsize=6, xytext=(3, -9),
                    textcoords="offset points", color="tab:red")
    for cost, a, lb in le:
        ax.plot([cost], [a], "^", color="tab:green", ms=7)
        ax.annotate(lb, (cost, a), fontsize=6, xytext=(3, 4),
                    textcoords="offset points", color="tab:green")
    ax.grid(alpha=0.3)
for ax in axes.flat[len(panels) + 1:]:
    ax.axis("off")
axes.flat[0].legend(fontsize=8)
fig.suptitle("Step1 S1-B: fixed-N vs first-to-2 (red) vs lead-k (green)", y=0.995)
fig.supxlabel("mean calls / problem")
fig.supylabel("accuracy")
fig.tight_layout()
fig.savefig(os.path.join(IMG, "fig_step1_pareto.png"), dpi=150)

# 3) 混成
if mix_res:
    acc9, accm9, accw, best = mix_res
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [f"{f} {l}\nSC9-pool" for f, l in TRIO] + ["mixed 3×3", "ā-weighted\n(exploratory)"]
    vals = [acc9[m] for m in TRIO] + [accm9, accw]
    cols = [FAM_STYLE[m[0]][0] for m in TRIO] + ["tab:purple", "tab:gray"]
    bars = ax.bar(range(len(vals)), vals, color=cols)
    bars[-1].set_hatch("//")
    ax.set_xticks(range(len(vals)), names, fontsize=8)
    ax.set_ylabel("accuracy")
    ax.set_ylim(min(vals) - 0.05, max(vals) + 0.03)
    ax.set_title("Step1 S1-C: mixed-family majority vs single-model (9 calls, budget-fair)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig_step1_mixed.png"), dpi=150)
print(f"\nwrote {IMG}/fig_step1_{{exchange,pareto,mixed}}.png")

print("\n※ 全結論に「GSM8K in-sample（シミュレーション構成）」を冠する。相互作用帰属の出典は 1c。"
      "\n※ S1-C の混成は討論ではない（見せ合いなし・多数決のみ）— 報告で必ず区別する。")
