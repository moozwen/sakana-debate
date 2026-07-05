import re

# 先頭は必ず数字。旧版 [\d,]+ はカンマ単独文字列にマッチし、散文末尾の
# カンマを拾って None を返すバグがあった（Phase 1 討論ログで null 多発の主因）。
_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def _norm(s):
    s = s.replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
    except ValueError:
        return None
    return f


def extract(text):
    """答えの数値を抽出。優先度: '#### N' → '\\boxed{N}' → 'answer is N' → 末尾数値。

    討論ラウンドではフォーマット指示が弱まり #### が出ないことがあるため、
    LaTeX の \\boxed{} や 'answer is' も拾う。返り値は float か None。
    """
    if not text:
        return None
    for pat in (
        r"####\s*(" + _NUM + r")",
        r"\\boxed\{?\s*(" + _NUM + r")",
        r"answer\s+is[^\d\-]{0,15}(" + _NUM + r")",
    ):
        m = re.findall(pat, text)
        if m:
            return _norm(m[-1])
    m = re.findall(_NUM, text)  # fallback 先: 本文中の最後の数値
    return _norm(m[-1]) if m else None


def grade(pred, gold, eps=1e-6):
    """pred は extract() の返り値 (float/None)、gold は文字列。数値として比較する。"""
    if pred is None:
        return False
    g = _norm(str(gold))
    return g is not None and abs(pred - g) < eps
