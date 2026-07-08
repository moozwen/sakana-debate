#!/usr/bin/env python3
# check_progress.py — Part 1b 進捗確認。SSH 復帰後にこれ一発。
# 使い方: cd ~/sakana-debate && python scripts/check_progress.py
# 見るもの: vLLM の現モデル / tmux セッション / GPU / 条件×seed ごとの done数と暫定acc
#           / 実行中ファイルの更新時刻と完走ETA（概算）
import json
import os
import subprocess
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZES = [("0.5B", "results_regime/05b"), ("1.5B", "results_regime/15b"),
         ("3B", "results_regime/3b"), ("7B", "results_regime/7b"),
         ("Gemma-3 1B", "results_regime/gemma1b"), ("Gemma-3 4B", "results_regime/gemma4b")]
CONDS = [(1, 0), (3, 0), (3, 2)]
SEEDS = [1, 2, 3]


def n_expected():
    fn = os.path.join(ROOT, "data/gsm8k_subset_regime.jsonl")
    try:
        return sum(1 for l in open(fn) if l.strip())
    except OSError:
        return 500


def vllm_model():
    try:
        with urllib.request.urlopen("http://localhost:8000/v1/models", timeout=3) as r:
            return json.load(r)["data"][0]["id"]
    except Exception:
        return None


def tmux_sessions():
    try:
        out = subprocess.run(["tmux", "ls"], capture_output=True, text=True, timeout=5)
        return [l.split(":")[0] for l in out.stdout.splitlines()]
    except Exception:
        return []


def gpu_line():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        u, m, t = [x.strip() for x in out.stdout.strip().split(",")]
        return f"GPU {u}% / VRAM {m}/{t} MiB"
    except Exception:
        return "nvidia-smi 取得不可"


def ago(sec):
    if sec < 90:
        return f"{sec:.0f}秒前"
    if sec < 5400:
        return f"{sec/60:.0f}分前"
    return f"{sec/3600:.1f}時間前"


exp = n_expected()
now = time.time()
model = vllm_model()
print(f"vLLM(:8000): {model or '応答なし（未起動 or 入れ替え中）'}")
print(f"tmux: {', '.join(tmux_sessions()) or 'セッションなし'}   {gpu_line()}")
print(f"サブセット: {exp}問\n")

n_done_files = n_files = 0
running = None  # (label, cond, seed, lines, mtime, ctime)
for label, outdir in SIZES:
    d = os.path.join(ROOT, outdir)
    if not os.path.isdir(d):
        continue  # 未着手サイズ（Gemma はゲート成立時のみ）
    print(f"[{label}]")
    for n, r in CONDS:
        cells = []
        for s in SEEDS:
            n_files += 1
            fn = os.path.join(d, f"debate_N{n}_R{r}_seed{s}.jsonl")
            if not os.path.exists(fn):
                cells.append(f"s{s} —")
                continue
            oks, lines = 0, 0
            for l in open(fn):
                if l.strip():
                    lines += 1
                    oks += json.loads(l)["ok"]
            mark = "✔" if lines >= exp else f"{lines}/{exp}"
            acc = f" acc={oks/lines:.3f}" if lines else ""
            cells.append(f"s{s} {mark}{acc}")
            n_done_files += lines >= exp
            st = os.stat(fn)
            if lines < exp and (running is None or st.st_mtime > running[4]):
                running = (label, f"N{n}_R{r}", s, lines, st.st_mtime, st.st_ctime)
        print(f"  N{n}_R{r}: " + " | ".join(cells))
print(f"\n完了ファイル: {n_done_files}/{n_files}（Qwen 4サイズで36が満了）")

if running:
    label, cond, s, lines, mtime, ctime = running
    line = f"実行中とみられる: {label} {cond} seed{s} — {lines}/{exp}問、最終書き込み {ago(now - mtime)}"
    if lines > 5 and mtime > ctime:
        rate = (mtime - ctime) / lines  # 秒/問（レジューム再開後はズレるので概算）
        line += f"  ETA≈{(exp - lines) * rate / 60:.0f}分（概算）"
    print(line)
    if now - mtime > 600:
        print("  ⚠ 10分以上書き込みが無い。Spot 中断/vLLM 停止の可能性 → "
              "`tmux attach -t regime` と `tmux attach -t vllm` でログ確認")
else:
    print("書き込み中のファイルなし（全部完了 or 未開始）。次の config を流すか analyze へ。")
print("\nライブで見るなら: tmux attach -t regime（デタッチは Ctrl-b d）")
