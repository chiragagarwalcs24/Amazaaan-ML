"""
similarity.py
-------------
Person 2 – String Similarity Metrics

Provides:
- levenshtein(a, b)        → normalised edit-distance ratio [0, 1]
- jaccard(a, b, n_gram)    → word-token or char-ngram Jaccard [0, 1]
- token_sort_ratio(a, b)   → sort tokens then compare [0, 1]
- token_set_ratio(a, b)    → set-based token comparison [0, 1]

Uses rapidfuzz for speed (C extension); falls back to pure Python if not available.
All functions are safe for empty strings.
"""

from __future__ import annotations
from typing import Set

# ─── Try to load rapidfuzz (fast C implementation) ───────────────────────────
try:
    from rapidfuzz.distance import Levenshtein as _RFL
    from rapidfuzz.fuzz import token_sort_ratio as _rf_tsort
    from rapidfuzz.fuzz import token_set_ratio  as _rf_tset
    _HAS_RF = True
except ImportError:
    _HAS_RF = False


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _lev_distance_pure(a: str, b: str) -> int:
    """Pure-Python Levenshtein distance (O(m*n) DP)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1] + [0] * len(b)
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev = curr
    return prev[len(b)]


def _safe(a, b):
    """Return (a_str, b_str) — both stripped strings. Returns ('', '') for non-strings."""
    a = a if isinstance(a, str) else ""
    b = b if isinstance(b, str) else ""
    return a.strip(), b.strip()


# ─── Public API ───────────────────────────────────────────────────────────────

def levenshtein(a: str, b: str) -> float:
    """
    Normalised Levenshtein similarity in [0, 1].

    1.0 = identical strings, 0.0 = completely different.
    Returns 1.0 when both strings are empty (both missing ≡ same).

    Examples
    --------
    levenshtein("inc seafood", "inc seafood") → 1.0
    levenshtein("seafood",     "sea food")    → 0.857
    levenshtein("",            "abc")         → 0.0
    """
    a, b = _safe(a, b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if _HAS_RF:
        return _RFL.normalized_similarity(a, b)
    dist = _lev_distance_pure(a, b)
    return 1.0 - dist / max(len(a), len(b))


def _word_set(text: str) -> Set[str]:
    return set(text.split()) if text else set()


def _char_ngrams(text: str, n: int) -> Set[str]:
    if not text or len(text) < n:
        return {text} if text else set()
    padded = f"${text}$"
    return {padded[i:i+n] for i in range(len(padded) - n + 1)}


def jaccard(a: str, b: str, n_gram: int = 1) -> float:
    """
    Jaccard similarity between two strings.

    Parameters
    ----------
    n_gram : int
        1  → word-level Jaccard  (default)
        2+ → character n-gram Jaccard

    Returns
    -------
    float in [0, 1].  1.0 when both empty.

    Examples
    --------
    jaccard("holloway peak inc", "holloway peak seafood inc") → 0.75
    jaccard("elmst", "elm street", n_gram=3) → ~0.45
    """
    a, b = _safe(a, b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    sa = _word_set(a)  if n_gram == 1 else _char_ngrams(a, n_gram)
    sb = _word_set(b)  if n_gram == 1 else _char_ngrams(b, n_gram)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def token_sort_ratio(a: str, b: str) -> float:
    """
    Sort tokens alphabetically then compute normalised Levenshtein.
    Handles word-order transpositions, e.g.:
      "Holloway Peak Inc Seafood" ↔ "Holloway Peak Seafood Inc" → 1.0

    Returns float in [0, 1].
    """
    a, b = _safe(a, b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if _HAS_RF:
        return _rf_tsort(a, b) / 100.0
    as_ = " ".join(sorted(a.split()))
    bs_ = " ".join(sorted(b.split()))
    return levenshtein(as_, bs_)


def token_set_ratio(a: str, b: str) -> float:
    """
    Compare common-token intersection against each remainder.
    Handles substring matches, e.g.:
      "Ram Marketing" ↔ "Ram Marketing Private Ltd" → 1.0

    Returns float in [0, 1].
    """
    a, b = _safe(a, b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if _HAS_RF:
        return _rf_tset(a, b) / 100.0
    ta, tb = set(a.split()), set(b.split())
    common   = " ".join(sorted(ta & tb))
    sorted_a = " ".join(sorted(ta))
    sorted_b = " ".join(sorted(tb))
    return max(
        levenshtein(common, sorted_a),
        levenshtein(common, sorted_b),
        levenshtein(sorted_a, sorted_b),
    )
