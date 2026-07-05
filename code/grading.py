import re

_NUM = r"-?\$?[\d,]+(?:\.\d+)?"


def _norm(s):
    s = s.replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
    except ValueError:
        return None
    return f


def extract(text):
    """'#### <number>' を最優先で抽出、無ければ末尾の数値。返り値は float か None。"""
    if not text:
        return None
    m = re.findall(r"####\s*(" + _NUM + r")", text)
    if not m:
        m = re.findall(_NUM, text)  # fallback 先: 本文中の最後の数値
    return _norm(m[-1]) if m else None


def grade(pred, gold, eps=1e-6):
    """pred は extract() の返り値 (float/None)、gold は文字列。数値として比較する。"""
    if pred is None:
        return False
    g = _norm(str(gold))
    return g is not None and abs(pred - g) < eps
