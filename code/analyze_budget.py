# analyze_budget.py — Part 1c（budget-fair N スケーリング）の事前登録解析
# 事前登録: vault/01_Inbox/04_Crafts/sakana_problem/phase_1/1c/part1c-budget-preregistration.md
# 出すもの:
#   (0) 入れ子サニティ（N9 の agent0-2 vs N3_R0、床 vs N3_R0 agent0 の一致率。
#       リクエスト seed は設計上同一だがビット再現は主張しない — 報告のみ、§3.1）
#   (1) モデル×条件 accuracy（3seed 平均±95%CI、コール数つき）
#   (2) McNemar + Holm（検定タイプごとに 5 モデルで 1 ファミリ、§4）:
#       主検定 P2 討論 vs SC9 / P1 SC9 vs SC3 / P3 自己再考 vs SC3 / P6 Δ_self 自己再考 vs 床
#   (3) 創発フリップ率（P4）+ SC9 の「9票中正解ゼロ」率
#   (4) 票の多様性（N3 vs N9: ユニーク解答数・正解含有率）
#   (5) format / prompt_tokens・コンテキスト逼迫の監視（§3.3）
#   (6) 解析C: SC 天井シミュレーション（§4.1。N9 の 3seed×9agent=27 サンプルプール。
#       床/N3/討論R0 は N9 agent0-2 と同一リクエスト seed のため足さない = 重複排除は構成上保証）
#   (7) fig_budget.png（予算曲線、モデルごと 1 面）→ results_regime/img/ に明示保存
#   (8) §5 判定ルールの機械適用（主判定 / §5-4 前提 / P6 / 3B 対照 / C-1・C-2）
# 第 2 段（N15_R0 / N5_R2）はファイルがあれば曲線に重畳するが、検定には入れない
# （追加検定の Holm ファミリは実行前に追補で固定する — 事前登録 §4）。
# 使い方: cd sakana-debate && python code/analyze_budget.py
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from math import comb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from grading import _NUM, grade

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "results_regime", "img")
os.makedirs(IMG, exist_ok=True)

# コア 5 モデル（事前登録 §3.1。実行順 = この並び）
MODELS = [("Qwen2.5", "0.5B", "results_regime/05b"),
          ("Qwen2.5", "1.5B", "results_regime/15b"),
          ("Gemma3", "1B", "results_regime/gemma1b"),
          ("Llama3.2", "1B", "results_regime/llama1b"),
          ("Qwen2.5", "3B", "results_regime/3b")]
POSITIVE = {("Qwen2.5", "0.5B"), ("Qwen2.5", "1.5B"), ("Gemma3", "1B")}  # §5-1 主判定の対象
BAND = POSITIVE | {("Llama3.2", "1B")}  # §5-4 前提チェックの対象（帯内 4 点）
CONTROL = ("Qwen2.5", "3B")  # §5-5 境界対照
CONDS = {"floor": (1, 0), "sc3": (3, 0), "debate": (3, 2), "sc9": (9, 0), "self": (1, 2),
         "sc15": (15, 0), "debate5": (5, 2)}  # sc15/debate5 は第 2 段（存在すれば曲線のみ）
CALLS = {"floor": 1, "sc3": 3, "debate": 9, "sc9": 9, "self": 3, "sc15": 15, "debate5": 15}
SEEDS = [1, 2, 3]
T95_3 = 4.303          # 自由度 2 の t 値（3 seed）
CTX_HEADROOM = 8192 - 1024  # プロンプトがこれを超えると生成が詰まる（§3.3）
B_DELTA = 2000         # ペア差 CI（Part 1b と同値）
B_CEIL = 10000         # 解析C 天井 CI（§4.1）
B_FLIP = 10000         # 創発フリップ CI（§5-3）
SIM = 300              # 解析C の問題別モンテカルロ数（平均への MC ノイズは CI 幅より 1 桁小さい）
NS_SIM = [3, 5, 9, 15, 25, 101]  # §4.1 の N グリッド

# --- format 分類（analyze_regime.py と同一の階層） ---
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


def load(outdir, n, r):
    out = {}
    for s in SEEDS:
        fn = os.path.join(ROOT, outdir, f"debate_N{n}_R{r}_seed{s}.jsonl")
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


def paired_delta(A, B, boots=B_DELTA):
    """A−B のペア差平均と問題単位ブートストラップ 95%CI。"""
    per_idx = defaultdict(list)  # 問題単位でまとめて resample（seed 相関を保つ）
    for s in SEEDS:
        if s not in A or s not in B:
            continue
        for i in A[s]:
            if i in B[s]:
                per_idx[i].append(A[s][i]["ok"] - B[s][i]["ok"])
    if not per_idx:
        return None
    groups = list(per_idx.values())
    n_pairs = sum(len(g) for g in groups)
    d = sum(sum(g) for g in groups) / n_pairs
    bs = []
    for _ in range(boots):
        pick = rng.choices(groups, k=len(groups))
        tot = sum(sum(g) for g in pick)
        cnt = sum(len(g) for g in pick)
        bs.append(tot / cnt)
    bs.sort()
    return d, bs[int(boots * 0.025)], bs[int(boots * 0.975) - 1]


def acc_all(D):
    """全 (seed, idx) プールの accuracy。"""
    oks = [r["ok"] for recs in D.values() for r in recs.values()]
    return sum(oks) / len(oks) if oks else None


data = {}  # (fam, label) -> cond -> {seed: {idx: rec}}
for fam, label, outdir in MODELS:
    data[(fam, label)] = {cl: load(outdir, n, r) for cl, (n, r) in CONDS.items()}

# ---------- (0) 入れ子サニティ（§3.1。ビット再現は主張しない — 一致率の報告のみ） ----------
print("=== 入れ子サニティ: リクエスト seed 共有条件の R0 予測一致率 ===")
for fam, label, _ in MODELS:
    D = data[(fam, label)]
    if not D["sc9"] or not D["sc3"]:
        continue
    m9 = t9 = mf = tf = 0
    for s in SEEDS:
        if s not in D["sc9"] or s not in D["sc3"]:
            continue
        for i, r9 in D["sc9"][s].items():
            if i in D["sc3"][s]:
                p9, p3 = r9["rounds"][0]["preds"][:3], D["sc3"][s][i]["rounds"][0]["preds"]
                t9 += len(p3)
                m9 += sum(a == b for a, b in zip(p9, p3))
            if s in D["floor"] and i in D["floor"][s]:
                tf += 1
                mf += r9["rounds"][0]["preds"][0] == D["floor"][s][i]["rounds"][0]["preds"][0]
    fl = f"{mf}/{tf} ({mf/tf:.1%})" if tf else "—"
    print(f"  {fam} {label:>5}: N9[0:3] vs N3_R0 {m9}/{t9} ({m9/t9:.1%})  "
          f"N9[0] vs 床 {fl}" if t9 else f"  {fam} {label:>5}: —")
    if t9 and m9 / t9 < 0.95:
        print("    注: 一致率 <95% は vLLM 継続バッチング非決定性による既知の挙動（1c 追補 A2 に開示済み。"
              "ビット再現は非主張）。入れ子の分散低減は効かないが、検定・解析C の妥当性には影響なし")

# ---------- (1) accuracy 表（コール数つき） ----------
print("\n=== accuracy（3seed 平均 ± 95%CI。カッコ内 = コール数/問）===")
acc = {}
for fam, label, _ in MODELS:
    parts = []
    for cl in CONDS:
        D = data[(fam, label)][cl]
        per_seed = [sum(r["ok"] for r in recs.values()) / len(recs)
                    for recs in D.values() if recs]
        if per_seed:
            acc[(fam, label, cl)] = ci3(per_seed)
            m, h = acc[(fam, label, cl)]
            parts.append(f"{cl}({CALLS[cl]})={m:.3f}±{h:.3f}")
    if parts:
        print(f"  {fam} {label:>5}: " + "  ".join(parts))

# ---------- (2) McNemar + Holm（検定タイプごとに 1 ファミリ、§4） ----------
FAMILIES = [("P2 主検定: 討論(9コール) vs SC9(9コール)", "debate", "sc9"),
            ("P1: SC9 vs SC3（サンプリングのスケーリング）", "sc9", "sc3"),
            ("P3: 自己再考 N1_R2(3コール) vs SC3(3コール)", "self", "sc3"),
            ("P6: Δ_self = 自己再考 N1_R2 vs 床 N1_R0（主対象は Llama）", "self", "floor")]
results = {}  # (family_idx, fam, label) -> (b, c, p, holm_p, delta)
for fi, (title, ca, cb) in enumerate(FAMILIES):
    tests = []
    for fam, label, _ in MODELS:
        D = data[(fam, label)]
        if D[ca] and D[cb]:
            b, c, p = mcnemar(D[ca], D[cb])
            tests.append([fam, label, b, c, p])
    if not tests:
        continue
    print(f"\n=== {title} — Holm ファミリ {len(tests)} 本 ===")
    for rank, t in enumerate(sorted(tests, key=lambda t: t[4])):
        t.append(min(1.0, t[4] * (len(tests) - rank)))
    for fam, label, b, c, p, hp in tests:
        d = paired_delta(data[(fam, label)][FAMILIES[fi][1]], data[(fam, label)][FAMILIES[fi][2]])
        ds = f"Δ={d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]" if d else ""
        print(f"  {fam} {label:>5}: +{b} / -{c}  p={p:.3f}  Holm p={hp:.3f}  {ds}")
        results[(fi, fam, label)] = (b, c, p, hp, d)

# ---------- (3) 創発フリップ率（P4）+ SC9 正解ゼロ率 ----------
print("\n=== P4: 創発フリップ（討論 R0 全票誤り → 最終多数決正解）===")
for fam, label, _ in MODELS:
    D = data[(fam, label)]["debate"]
    if not D:
        continue
    per_idx = defaultdict(list)  # idx -> [(R0全誤, 最終ok)]
    for s, recs in D.items():
        for i, r in recs.items():
            allwrong = not any(grade(p, r["gold"]) for p in r["rounds"][0]["preds"])
            per_idx[i].append((allwrong, r["ok"]))
    events = [(aw, ok) for evs in per_idx.values() for aw, ok in evs if aw]
    n_aw, n_flip = len(events), sum(ok for _, ok in events)
    rate = n_flip / n_aw if n_aw else 0.0
    # 問題単位ブートストラップ CI（§5-3）
    groups = list(per_idx.values())
    bs = []
    for _ in range(B_FLIP):
        pick = rng.choices(groups, k=len(groups))
        aw = sum(1 for g in pick for a, _ in g if a)
        fl = sum(1 for g in pick for a, o in g if a and o)
        bs.append(fl / aw if aw else 0.0)
    bs.sort()
    lo, hi = bs[int(B_FLIP * 0.025)], bs[int(B_FLIP * 0.975) - 1]
    # SC9 側: 9 票中正解ゼロの率（初期票に正解が無ければ SC は定義上救えない — P4 の対）
    D9 = data[(fam, label)]["sc9"]
    zero = tot = 0
    for s, recs in D9.items():
        for r in recs.values():
            tot += 1
            zero += not any(grade(p, r["gold"]) for p in r["final_preds"])
    z = f"  SC9 正解ゼロ率: {zero/tot:.1%} ({zero}/{tot})" if tot else ""
    print(f"  {fam} {label:>5}: {n_flip}/{n_aw} = {rate:.1%} [CI {lo:.1%}, {hi:.1%}]"
          f"{'（CI がゼロを含まない → 独立報告対象）' if lo > 0 else ''}{z}")

# ---------- (4) 票の多様性（P1 の機構） ----------
print("\n=== 票の多様性（(seed,idx) 平均ユニーク解答数 / 正解含有率）===")
for fam, label, _ in MODELS:
    parts = []
    for cl in ("sc3", "sc9"):
        D = data[(fam, label)][cl]
        if not D:
            continue
        uniq, has_gold, tot = 0, 0, 0
        for recs in D.values():
            for r in recs.values():
                tot += 1
                uniq += len({p for p in r["final_preds"] if p is not None})
                has_gold += any(grade(p, r["gold"]) for p in r["final_preds"])
        parts.append(f"{cl}: uniq={uniq/tot:.2f} 正解含有={has_gold/tot:.1%}")
    if parts:
        print(f"  {fam} {label:>5}: " + "  |  ".join(parts))

# ---------- (5) format / prompt_tokens（§3.3 監視） ----------
print("\n=== format（新条件）と prompt_tokens ===")
for fam, label, _ in MODELS:
    for cl in ("sc9", "self", "sc15", "debate5"):
        D = data[(fam, label)][cl]
        if not D:
            continue
        fmt_rnd = None
        none_ct = tot_p = 0
        ptok_sum = ptok_n = ptok_max = over = recs_n = 0
        for recs in D.values():
            for r in recs.values():
                if fmt_rnd is None:
                    fmt_rnd = [Counter() for _ in r["rounds"]]
                for rnd, rd in enumerate(r["rounds"]):
                    for t in rd["raw"]:
                        fmt_rnd[rnd][extract_method(t)] += 1
                for p in r["final_preds"]:
                    tot_p += 1
                    none_ct += p is None
                recs_n += 1
                if "max_prompt_tokens" in r:
                    ptok_n += 1
                    ptok_sum += r["prompt_tokens"]
                    ptok_max = max(ptok_max, r["max_prompt_tokens"])
                    over += r["max_prompt_tokens"] > CTX_HEADROOM
        hash_traj = " -> ".join(f"{fr['hash']/sum(fr.values()):.1%}" for fr in fmt_rnd if fr)
        pt = (f"  prompt_tokens: 平均計{ptok_sum/ptok_n:.0f}/問 最大{ptok_max} "
              f"逼迫(>{CTX_HEADROOM}) {over}/{ptok_n}" if ptok_n else "  prompt_tokens: 未記録")
        print(f"  {fam} {label:>5} {cl}: ####遵守 {hash_traj}  pred=None {none_ct/tot_p:.1%}{pt}")

# ---------- (6) 解析C: SC 天井シミュレーション（§4.1） ----------
print("\n=== 解析C: SC 天井（N9 の 27 サンプルプール。重複排除は構成上保証 = プールは sc9 のみ）===")
ceil_result = {}  # (fam,label) -> (ceiling, lo, hi, judgment, plural_wrong_idx_set)
for fam, label, _ in MODELS:
    D9 = data[(fam, label)]["sc9"]
    if len(D9) < len(SEEDS) or any(len(recs) < 500 for recs in D9.values()):
        print(f"  {fam} {label:>5}: N9 未完走 — スキップ")
        continue
    pools = {}  # idx -> (values, gold)
    for s in SEEDS:
        for i, r in D9[s].items():
            pools.setdefault(i, ([], r["gold"]))[0].extend(r["final_preds"])
    bad = [i for i, (v, _) in pools.items() if len(v) != 9 * len(SEEDS)]
    assert not bad, f"プールサイズ異常（27 でない）: idx {bad[:5]} …重複/欠落を調査"
    qN = {N: [] for N in NS_SIM}   # 問題別 SC(N) 正答確率
    lim, lim_j = [], []            # 経験極限 / Jeffreys 平滑化極限（頑健性チェック）
    for i, (vals, gold) in sorted(pools.items()):
        okmap = {v: grade(v, gold) for v in set(vals) if v is not None}
        k = sum(okmap.get(v, False) for v in vals)
        n = len(vals)
        # 極限: 非 None 票の最頻値が正解か（同率タイは 0.5）
        mass = Counter(v for v in vals if v is not None)
        if mass:
            top = max(mass.values())
            tied_ok = [okmap[v] for v, c in mass.items() if c == top]
            l_emp = 1.0 if all(tied_ok) else (0.5 if any(tied_ok) else 0.0)
        else:
            l_emp = 0.0
        lim.append(l_emp)
        # Jeffreys 平滑化（正答確率のみ縮約。正解/誤答の質量比を (k+.5)/(n+1) に置き直す）
        pj = (k + 0.5) / (n + 1)
        l_j = l_emp
        if 0 < k < n:  # 縮約で最頻値の正誤が入れ替わるケースだけ再判定
            wmass = {v: c * ((1 - pj) / ((n - k) / n)) / n for v, c in mass.items()
                     if not okmap[v]}
            cmass = {v: c * (pj / (k / n)) / n for v, c in mass.items() if okmap[v]}
            allm = {**wmass, **cmass}
            top = max(allm.values())
            tied_ok = [okmap[v] for v, m in allm.items() if abs(m - top) < 1e-12]
            l_j = 1.0 if all(tied_ok) else (0.5 if any(tied_ok) else 0.0)
        lim_j.append(l_j)
        # SC(N) モンテカルロ（タイ処理は実装と同一: 引いた順で最初のタイ解答）
        for N in NS_SIM:
            if k == n:
                qN[N].append(1.0)
                continue
            if k == 0:
                qN[N].append(0.0)
                continue
            wins = 0
            for _ in range(SIM):
                sample = rng.choices(vals, k=N)
                votes = Counter(v for v in sample if v is not None)
                if not votes:
                    continue
                top = votes.most_common(1)[0][1]
                tied = {v for v, c in votes.items() if c == top}
                if len(tied) == 1:
                    wins += okmap[next(iter(tied))]
                else:
                    for v in sample:
                        if v in tied:
                            wins += okmap[v]
                            break
            qN[N].append(wins / SIM)
    npb = len(lim)
    ceiling = sum(lim) / npb
    bs = sorted(sum(rng.choices(lim, k=npb)) / npb for _ in range(B_CEIL))
    lo, hi = bs[int(B_CEIL * 0.025)], bs[int(B_CEIL * 0.975) - 1]
    curve = "  ".join(f"N{N}={sum(q)/npb:.3f}" for N, q in qN.items())
    deb = acc_all(data[(fam, label)]["debate"])
    verdict = "—（討論データなし）"
    if deb is not None:
        c1 = deb > hi
        verdict = (f"C-1 天井超え（討論 {deb:.3f} > CI 上限 {hi:.3f}）→ サンプリング原理限界の外"
                   if c1 else
                   f"C-2 天井以下（討論 {deb:.3f} ≤ CI 上限 {hi:.3f}）→ 主張はサンプル効率に格下げ")
        ceil_result[(fam, label)] = (ceiling, lo, hi, c1)
    print(f"  {fam} {label:>5}: {curve}")
    print(f"          極限 = {ceiling:.3f} [CI {lo:.3f}, {hi:.3f}]"
          f"（Jeffreys 頑健性: {sum(lim_j)/npb:.3f}）  判定: {verdict}")
    # 補助解析（C-1 成立時のみ、§4.1）: 討論正解のうち経験相対多数が誤答だった問題の割合
    if deb is not None and ceil_result.get((fam, label), (0, 0, 0, False))[3]:
        pw = {i for (i, _), l in zip(sorted(pools.items()), lim) if l == 0.0}
        Dd = data[(fam, label)]["debate"]
        okev = [(r["idx"] in pw) for recs in Dd.values() for r in recs.values() if r["ok"]]
        print(f"          補助: 討論正解 (seed,idx) のうち相対多数が誤答の問題 "
              f"{sum(okev)}/{len(okev)} ({sum(okev)/len(okev):.1%})")

# ---------- (7) 予算曲線（主図） ----------
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=False)
for ax, (fam, label, _) in zip(axes.flat, MODELS):
    def pt(cl):
        return acc.get((fam, label, cl))
    sc_pts = [(CALLS[cl], *pt(cl)) for cl in ("floor", "sc3", "sc9", "sc15") if pt(cl)]
    if sc_pts:
        x, m, h = zip(*sc_pts)
        ax.errorbar(x, m, yerr=h, fmt="o-", color="tab:blue", capsize=3,
                    label="SC majority (N1→N3→N9→[N15])")
    for cl, style, lab in [("debate", "^", "debate N3_R2"), ("debate5", "v", "debate N5_R2"),
                           ("self", "D", "self-refine N1_R2")]:
        p = pt(cl)
        if p:
            ax.errorbar([CALLS[cl]], [p[0]], yerr=[p[1]], fmt=style, capsize=3, ms=8,
                        color="tab:red" if cl.startswith("debate") else "tab:green", label=lab)
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 9, 15], ["1", "3", "9", "15"])
    ax.set_title(f"{fam} {label}", fontsize=10)
    ax.grid(alpha=0.3)
axes.flat[0].set_ylabel("GSM8K accuracy (500 problems)")
axes.flat[3].set_ylabel("GSM8K accuracy (500 problems)")
for ax in axes.flat[3:]:
    ax.set_xlabel("calls / problem = N×(R+1), log")
axes.flat[-1].axis("off")
h, l = axes.flat[0].get_legend_handles_labels()
axes.flat[-1].legend(h, l, loc="center", fontsize=9)
fig.suptitle("Part 1c budget curves (preregistered): equal-budget debate vs SC vs self-refine",
             y=0.995)
fig.tight_layout()
fig.savefig(os.path.join(IMG, "fig_budget.png"), dpi=150)
print(f"\nwrote {os.path.join(IMG, 'fig_budget.png')}")

# ---------- (8) §5 判定ルールの機械適用 ----------
print("\n=== 事前登録 §5 判定 ===")
fired, premise_fail = [], []
for fam, label, _ in MODELS:
    r = results.get((0, fam, label))  # P2 主検定
    if r and (fam, label) in POSITIVE and r[3] < 0.05 and r[0] > r[1]:
        fired.append(f"{fam} {label}")
    p1 = results.get((1, fam, label))
    if p1 and (fam, label) in BAND and p1[0] <= p1[1]:  # SC9 が SC3 に勝てていない
        premise_fail.append(f"{fam} {label}")
    if r and (fam, label) == CONTROL and r[3] < 0.05 and r[0] > r[1]:
        print(f"  ⚠ §5-5: 3B（帯の外）で主検定が正 — 「探索的・要追試」と明記、確証扱いしない")
done_p2 = [f"{fam} {label}" for fam, label, _ in MODELS if results.get((0, fam, label))]
if done_p2:
    final = len(done_p2) == len(MODELS)
    print(f"  [{'最終判定' if final else f'途中経過: P2 完了 {len(done_p2)}/{len(MODELS)}'}]")
    if fired:
        print(f"  §5-1 主判定: 相互作用固有の効果あり（{', '.join(fired)}）"
              f" → 該当サイズのみ第 2 段（N15_R0/N5_R2。Holm 追補を書いてから）")
    elif final:
        print("  §5-2: どのモデルも討論が SC N9 を有意に超えない → サンプリング帰着"
              "（第 2 段なし。Part 1b の主結果は毀損しない完結した報告）")
    else:
        print(f"  ここまで（{', '.join(done_p2)}）討論>SC9 のモデルなし — "
              f"§5-1/5-2 の確定は全 {len(MODELS)} モデル完走後。Holm p は完了分のみの暫定値")
    if premise_fail:
        print(f"  §5-4 前提破綻: SC9 ≤ SC3（{', '.join(premise_fail)}）→ 該当サイズは第 2 段に進まない")
    p6 = results.get((3, "Llama3.2", "1B"))
    if p6:
        b, c, p, hp, d = p6
        if hp >= 0.05:
            print(f"  P6（Llama Δ_self）: ≈0（Holm p={hp:.3f}）→ 本命成立 = "
                  f"害は「他者の解答を見ること」に固有（同調ドリフト解釈が機構レベルで確定）")
        elif c > b:
            print(f"  P6（Llama Δ_self）: 有意に負（+{b}/−{c}, Holm p={hp:.3f}）→ "
                  f"再考そのものが不安定（同調ではなく再生成ノイズが主因の絵）")
        else:
            print(f"  P6（Llama Δ_self）: 有意に正（+{b}/−{c}, Holm p={hp:.3f}）→ "
                  f"想定外（自己再考が単体で有効）。探索的として報告")
    print("  中心主張は主判定 × 解析C の組み合わせで §5 の主張レベル対応表から選ぶこと")
else:
    print("  新条件（N9_R0/N1_R2）のデータ未着 — 実行後に再実行")
