import json
import os
import re

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUT = "baseline_results.jsonl"

PROMPT = (
    "Solve the following grade-school math problem. Explain briefly, then give "
    "the final answer as a single number in the exact form '#### <number>' on the last line.\n\nProblem: {q}"
)


def extract(text: str):
    m = re.findall(r"####\s*(-?[\d,]+)", text)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*", text)  # フォールバック：末尾の数値
    return nums[-1].replace(",", "") if nums else None


rows = [json.loads(l) for l in open("gsm8k_subset.jsonl")]

# すでに済んだ idx を読み込む（レジューム用）
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        if l.strip():
            done.add(json.loads(l)["idx"])
if done:
    print(f"resume: {len(done)} 件は済み、残り {len(rows) - len(done)} 件")

with open(OUT, "a") as f:
    for r in rows:
        if r["idx"] in done:
            continue
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT.format(q=r["question"])}],
            temperature=0,
            max_tokens=512,
        )
        text = resp.choices[0].message.content
        pred = extract(text)
        ok = pred == r["gold"]
        rec = {"idx": r["idx"], "gold": r["gold"], "pred": pred, "ok": ok, "raw": text}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()  # ← OSバッファへ
        os.fsync(f.fileno())  # ← ディスクへ確定（切断に強くする）
        print(f"idx={r['idx']:>4}  ok={ok}  pred={pred}")

# 集計は最後にファイル全体から（レジュームでも全件を数える）
results = [json.loads(l) for l in open(OUT) if l.strip()]
n = len(results)
correct = sum(x["ok"] for x in results)
extract_fail = sum(x["pred"] is None for x in results)
print(f"accuracy = {correct}/{n} = {correct / n:.3f}")
print(f"extraction failures = {extract_fail}/{n}")
