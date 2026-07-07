# run_debate.py — N×R 討論ハーネス（Du et al. 2023 の stateless 変種。YAML設定・逐次保存・レジューム）
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from grading import extract, grade
from openai import OpenAI

CFG = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "config.yaml"))
client = OpenAI(base_url=CFG["base_url"], api_key="dummy")

INITIAL = (
    "Solve the following grade-school math problem. Explain your reasoning briefly, "
    "then give the final answer as a single number in the exact form '#### <number>' "
    "on the last line.\n\nProblem: {q}"
)
DEBATE = (
    "You previously solved this problem. Other agents have also proposed solutions.\n\n"
    "Your previous solution:\n{own}\n\n"
    "Solutions from other agents:\n{others}\n\n"
    "Using these solutions as additional information, carefully re-examine the problem. "
    "Explain your reasoning briefly, then give the final answer as a single number in the "
    "exact form '#### <number>' on the last line.\n\nProblem: {q}"
)


def req_seed(seed, idx, agent, rnd):
    # (seed, 問題, エージェント, ラウンド) から決定的にリクエストseedを作る
    h = hashlib.sha256(f"{seed}/{idx}/{agent}/{rnd}".encode()).hexdigest()
    return int(h[:8], 16)


def call(prompt, seed):
    resp = client.chat.completions.create(
        model=CFG["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=CFG["temperature"],
        max_tokens=CFG["max_tokens"],
        seed=seed,
    )
    usage = getattr(resp, "usage", None)
    tokens = getattr(usage, "completion_tokens", 0) or 0
    return resp.choices[0].message.content or "", tokens


def majority(preds):
    """最終ラウンドの多数決。タイは最年少番号エージェントの答え（決定的）。"""
    votes = Counter(p for p in preds if p is not None)
    if not votes:
        return None, False
    top_count = votes.most_common(1)[0][1]
    tied = {v for v, c in votes.items() if c == top_count}
    if len(tied) == 1:
        return tied.pop(), False
    for p in preds:  # 若い番号のエージェント順に走査
        if p in tied:
            return p, True
    return None, True


def run_problem(prob, n, r_rounds, seed):
    q = prob["question"]
    rounds, prev = [], None
    gen_tokens = 0  # budget-fair 比較の素地（Part 1b 事前登録 §4。本パートでは主張しない）
    for rnd in range(r_rounds + 1):
        texts = []
        for a in range(n):
            if rnd == 0:
                prompt = INITIAL.format(q=q)
            else:
                others = (
                    "\n\n".join(
                        f"Agent {j + 1}:\n{prev[j]}" for j in range(n) if j != a
                    )
                    or "(no other agents)"
                )
                prompt = DEBATE.format(q=q, own=prev[a], others=others)
            text, tok = call(prompt, req_seed(seed, prob["idx"], a, rnd))
            texts.append(text)
            gen_tokens += tok
        rounds.append({"raw": texts, "preds": [extract(t) for t in texts]})
        prev = texts
    final = rounds[-1]["preds"]
    maj, tie = majority(final)
    return {
        "idx": prob["idx"],
        "gold": prob["gold"],
        "n": n,
        "r": r_rounds,
        "seed": seed,
        "rounds": rounds,  # ← RQ5用：全ラウンド×全エージェント
        "final_preds": final,
        "majority": maj,
        "tie": tie,
        "unanimous": None not in final and len(set(final)) == 1,
        "ok": grade(maj, prob["gold"]),
        "gen_tokens": gen_tokens,
    }


def run_condition(n, r_rounds, seed, rows):
    os.makedirs(CFG["outdir"], exist_ok=True)
    out = os.path.join(CFG["outdir"], f"debate_N{n}_R{r_rounds}_seed{seed}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["idx"] for l in open(out) if l.strip()}
    todo = [p for p in rows if p["idx"] not in done]
    print(f"[N={n} R={r_rounds} seed={seed}] done={len(done)} todo={len(todo)}")
    with open(out, "a") as f, ThreadPoolExecutor(CFG["concurrency"]) as ex:
        futs = {ex.submit(run_problem, p, n, r_rounds, seed): p for p in todo}
        n_done, n_ok = 0, 0
        for fut in as_completed(futs):
            rec = fut.result()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            n_done += 1
            n_ok += rec["ok"]
            print(
                f"  {n_done:>3}/{len(todo)} idx={rec['idx']:>4} ok={rec['ok']} "
                f"maj={rec['majority']} unanimous={rec['unanimous']}"
            )
    if os.path.exists(out):
        recs = [json.loads(l) for l in open(out) if l.strip()]
        acc = sum(x["ok"] for x in recs) / len(recs)
        print(f"[N={n} R={r_rounds} seed={seed}] accuracy = {acc:.3f} ({len(recs)}問)")


rows = [json.loads(l) for l in open(CFG["subset"])]
for cond in CFG["grid"]:
    for seed in CFG["seeds"]:
        run_condition(cond["n"], cond["r"], seed, rows)
print("all conditions done")
