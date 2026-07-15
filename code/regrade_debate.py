# regrade_debate.py — 保存済み討論ログ(*.jsonl)を修正版 extract で採点し直す。
# GPU再実行不要。rounds[].raw の生テキストから preds/majority/ok を再計算する。
# 使い方: cd sakana-debate && python code/regrade_debate.py [results_dir]
import glob
import json
import os
import sys
from collections import Counter

from grading import extract, grade

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "experiments/phase1_grid/results"


def majority(preds):
    """run_debate.py と同一の多数決。タイは最年少番号エージェント（決定的）。"""
    votes = Counter(p for p in preds if p is not None)
    if not votes:
        return None, False
    top_count = votes.most_common(1)[0][1]
    tied = {v for v, c in votes.items() if c == top_count}
    if len(tied) == 1:
        return tied.pop(), False
    for p in preds:
        if p in tied:
            return p, True
    return None, True


def regrade_record(rec):
    """rounds[].raw から preds を引き直し、majority/tie/unanimous/ok を再計算。"""
    for rnd in rec["rounds"]:
        rnd["preds"] = [extract(t) for t in rnd["raw"]]
    final = rec["rounds"][-1]["preds"]
    maj, tie = majority(final)
    rec["final_preds"] = final
    rec["majority"] = maj
    rec["tie"] = tie
    rec["unanimous"] = None not in final and len(set(final)) == 1
    rec["ok"] = grade(maj, rec["gold"])
    return rec


def main():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "debate_*.jsonl")))
    if not files:
        print(f"no result files in {RESULTS_DIR}/")
        return
    print(f"{'file':30} {'#Q':>4} {'old_acc':>8} {'new_acc':>8} {'nullR_last':>11}")
    for fn in files:
        recs = [json.loads(l) for l in open(fn) if l.strip()]
        if not recs:
            continue
        old_acc = sum(r["ok"] for r in recs) / len(recs)
        recs = [regrade_record(r) for r in recs]
        new_acc = sum(r["ok"] for r in recs) / len(recs)
        last_preds = [p for r in recs for p in r["rounds"][-1]["preds"]]
        null_rate = sum(1 for p in last_preds if p is None) / len(last_preds)
        # インプレース上書き（レジューム用の1行1レコード形式を保つ）
        with open(fn, "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(
            f"{os.path.basename(fn):30} {len(recs):>4} "
            f"{old_acc:>8.3f} {new_acc:>8.3f} {null_rate:>11.3f}"
        )
    print(f"done — {RESULTS_DIR}/*.jsonl を修正版 extract で上書き採点しました")


if __name__ == "__main__":
    main()
