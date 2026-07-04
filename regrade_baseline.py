import json

from grading import extract, grade

FIXED = {
    1149,
    919,
    440,
    334,
    114,
    435,
    283,
    743,
    638,
}  # A+B: 採点バグ9件 → 正解になるはず
STILL_WRONG = {
    1309,
    689,
    255,
    781,
    539,
    611,
    1226,
    1059,
    570,
}  # C: 真の誤り9件 → 誤りのまま

rows = [json.loads(l) for l in open("baseline_results.jsonl") if l.strip()]
ok = {r["idx"]: grade(extract(r["raw"]), r["gold"]) for r in rows}
total = sum(ok.values())
print(f"regraded accuracy = {total}/{len(rows)} = {total / len(rows):.3f}")

assert all(ok[i] for i in FIXED), (
    f"救済されるはずが誤りのまま: {[i for i in FIXED if not ok[i]]}"
)
assert not any(ok[i] for i in STILL_WRONG), (
    f"過剰修正（誤りが正になった）: {[i for i in STILL_WRONG if ok[i]]}"
)
assert total == 191, f"合計が期待と不一致: {total} != 191"
print("✅ regression OK: 9件救済・9件は誤りのまま・過剰修正なし")
