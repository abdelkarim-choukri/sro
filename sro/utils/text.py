import re

_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

def years(s: str) -> set[int]:
    return set(int(y) for y in _YEAR.findall(s or ""))

def year_conflict(a: str, b: str) -> bool:
    ya, yb = years(a), years(b)
    return bool(ya and yb and ya.isdisjoint(yb))
