"""
feature_engineering.py
-----------------------
Person 2 – Feature Extraction Pipeline  (v2)

20 similarity features across 4 groups:
  NAME     (9)   – levenshtein, token sort/set, jaccard word/char, exact,
                   first token, length diff, length ratio
  ADDRESS  (7)   – levenshtein, token sort, jaccard word/char, house match,
                   empty flag, length diff
  TF-IDF   (3)   – cosine via name-char, name-word, addr-char vectorisers
  MISC     (1)   – country match

v2 additions
------------
- Phonetic code feature: Soundex of first token match
- Number token overlap: Jaccard of digit sequences in addresses
- Token count match: similar number of words → likely same entity
- Address prefix hash: hash of first 2 address tokens for quick match
"""

from __future__ import annotations

import re
import hashlib
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

from .normalization import normalize_name, normalize_address
from .similarity import levenshtein, jaccard, token_sort_ratio, token_set_ratio


# ─── Helpers ──────────────────────────────────────────────────────────────────

_HOUSE_RE  = re.compile(r"^\s*(\d[\w/-]*)")
_DIGIT_RE  = re.compile(r"\b(\d+)\b")


def _house_number(addr: str) -> str:
    m = _HOUSE_RE.match(addr)
    return m.group(1) if m else ""


def _number_tokens(text: str) -> set:
    """All digit sequences found in text."""
    return set(_DIGIT_RE.findall(text))


def _soundex(s: str) -> str:
    """
    Soundex phonetic code for English words.
    Returns '' for empty or non-ASCII strings.
    """
    s = s.strip().upper()
    if not s or not s[0].isalpha() or ord(s[0]) > 127:
        return ""
    # Correct table: only consonants get codes
    _MAP = {"B":"1","F":"1","P":"1","V":"1",
            "C":"2","G":"2","J":"2","K":"2","Q":"2","S":"2","X":"2","Z":"2",
            "D":"3","T":"3",
            "L":"4",
            "M":"5","N":"5",
            "R":"6"}
    code = s[0]
    prev = _MAP.get(s[0], "0")
    for ch in s[1:]:
        c = _MAP.get(ch, "0")
        if c != "0" and c != prev:
            code += c
        prev = c
        if len(code) == 4:
            break
    return code.ljust(4, "0")


def _addr_prefix_hash(addr: str, n_tokens: int = 3) -> str:
    """Hash of first n address tokens — quick locality key."""
    toks = addr.split()[:n_tokens]
    return hashlib.md5(" ".join(toks).encode()).hexdigest()[:8] if toks else ""


# ─── Step 1: Preprocess sources ───────────────────────────────────────────────

def preprocess_sources(
    sources: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Apply normalisation + add helper columns to each source DataFrame.

    Added columns
    -------------
    name_norm   : cleaned business name
    addr_norm   : cleaned address
    house_num   : leading house number token
    num_tokens  : frozenset of all digit sequences in addr_norm (stored as str)
    name_soundex: Soundex of first name token

    Returns new DataFrames (originals not mutated).
    """
    out = {}
    for key, df in sources.items():
        d = df.copy()
        d["name_norm"]    = d["business_name"].apply(normalize_name)
        d["addr_norm"]    = d["business_address"].apply(normalize_address)
        d["house_num"]    = d["addr_norm"].apply(_house_number)
        d["name_soundex"] = d["name_norm"].apply(
            lambda n: _soundex(n.split()[0]) if n.split() else ""
        )
        out[key] = d
    return out


# ─── Step 2: TF-IDF vectorisers ───────────────────────────────────────────────

class TFIDFBundle:
    """
    Three TF-IDF vectorisers:
      name_char  – char 2-4-gram on business names
      name_word  – word 1-2-gram on business names
      addr_char  – char 2-4-gram on addresses
    """

    def __init__(self, max_features: int = 50_000):
        self.name_char = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 4),
            max_features=max_features, sublinear_tf=True, min_df=2,
        )
        self.name_word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            max_features=max_features, sublinear_tf=True, min_df=2,
        )
        self.addr_char = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 4),
            max_features=max_features, sublinear_tf=True, min_df=2,
        )
        self.fitted = False

    def fit(self, names: List[str], addrs: List[str]) -> "TFIDFBundle":
        """Fit all three vectorisers on the corpus."""
        clean_names = [n for n in names if n]
        clean_addrs = [a for a in addrs if a]
        if clean_names:
            self.name_char.fit(clean_names)
            self.name_word.fit(clean_names)
        if clean_addrs:
            self.addr_char.fit(clean_addrs)
        self.fitted = True
        return self

    def cosine_batch(
        self,
        na: List[str], nb: List[str],
        aa: List[str], ab: List[str],
    ) -> np.ndarray:
        """
        Return (N, 3) array: [name_cos_char, name_cos_word, addr_cos_char].
        Uses L2-normalised row dot product (correct cosine similarity).
        """
        if not self.fitted:
            return np.zeros((len(na), 3), dtype=np.float32)

        def _cos(v1, v2):
            v1n = sk_normalize(v1, norm="l2")
            v2n = sk_normalize(v2, norm="l2")
            return np.asarray(v1n.multiply(v2n).sum(axis=1)).ravel()

        nc = _cos(self.name_char.transform(na), self.name_char.transform(nb))
        nw = _cos(self.name_word.transform(na), self.name_word.transform(nb))
        ac = _cos(self.addr_char.transform(aa), self.addr_char.transform(ab))
        return np.column_stack([nc, nw, ac]).astype(np.float32)


# ─── Step 3: Feature extraction ───────────────────────────────────────────────

def compute_features(
    pairs_df: pd.DataFrame,
    all_records: pd.DataFrame,
    tfidf: Optional[TFIDFBundle] = None,
    chunksize: int = 50_000,
) -> pd.DataFrame:
    """
    Compute the full 23-feature matrix for every candidate pair.

    Parameters
    ----------
    pairs_df    : DataFrame with [source1_entity_id, candidate_entity_id]
                  Optionally also contains 'label' (1/0) for training pairs.
    all_records : pd.concat of all preprocessed sources; must have columns
                  [entity_id, name_norm, addr_norm, house_num, name_soundex, country]
    tfidf       : fitted TFIDFBundle (cosine cols = 0.0 if None or not fitted)
    chunksize   : NOTE – declared for future chunked processing but currently
                  unused. The entire pairs_df is processed in one pass. For
                  production use with millions of pairs, implement chunking here
                  to avoid OOM errors.

    Returns
    -------
    DataFrame with same row order as pairs_df, plus all feature columns.
    Feature column names:
        name_lev, name_tsort, name_tset, name_jacc_word, name_jacc_c3,
        name_exact, name_first_tok, name_len_diff, name_len_ratio,
        name_tok_count_diff, name_soundex_match,
        addr_lev, addr_tsort, addr_jacc_word, addr_jacc_c3,
        addr_house_match, addr_empty, addr_len_diff, addr_num_jacc,
        country_match,
        name_cos_char, name_cos_word, addr_cos_char
    """
    keep = ["entity_id", "name_norm", "addr_norm",
            "house_num", "name_soundex", "country"]
    rec_map: Dict[str, dict] = (
        all_records[keep].set_index("entity_id").to_dict(orient="index")
    )

    id1_col = pairs_df["source1_entity_id"].tolist()
    id2_col = pairs_df["candidate_entity_id"].tolist()
    n       = len(pairs_df)
    rows    = []

    for i in range(n):
        id1 = id1_col[i]
        id2 = id2_col[i]
        r1  = rec_map.get(id1, {})
        r2  = rec_map.get(id2, {})

        n1 = r1.get("name_norm", "")
        n2 = r2.get("name_norm", "")
        a1 = r1.get("addr_norm", "")
        a2 = r2.get("addr_norm", "")
        c1 = r1.get("country", "")
        c2 = r2.get("country", "")
        h1 = r1.get("house_num", "")
        h2 = r2.get("house_num", "")
        sx1 = r1.get("name_soundex", "")
        sx2 = r2.get("name_soundex", "")

        # ── NAME features ──────────────────────────────────────────────────
        n_lev   = levenshtein(n1, n2)
        n_tsort = token_sort_ratio(n1, n2)
        n_tset  = token_set_ratio(n1, n2)
        n_jw    = jaccard(n1, n2, n_gram=1)
        n_jc    = jaccard(n1, n2, n_gram=3)
        n_exact = 1.0 if (n1 and n1 == n2) else 0.0

        t1, t2  = n1.split() if n1 else [], n2.split() if n2 else []
        n_ftok  = 1.0 if (t1 and t2 and t1[0] == t2[0]) else 0.0
        n_ldiff = abs(len(n1) - len(n2))
        tc1, tc2 = len(t1), len(t2)
        n_lratio = (min(tc1, tc2) / max(tc1, tc2)) if max(tc1, tc2) > 0 else 1.0
        n_tcdiff = abs(tc1 - tc2)    # NEW: word count difference

        # Soundex phonetic match on first token
        n_sdx   = 1.0 if (sx1 and sx1 == sx2) else 0.0   # NEW

        # ── ADDRESS features ───────────────────────────────────────────────
        a_lev   = levenshtein(a1, a2)
        a_tsort = token_sort_ratio(a1, a2)
        a_jw    = jaccard(a1, a2, n_gram=1)
        a_jc    = jaccard(a1, a2, n_gram=3)
        a_empty = 1.0 if (not a1 or not a2) else 0.0
        a_ldiff = abs(len(a1) - len(a2))

        if h1 and h2:
            a_house = 1.0 if h1 == h2 else 0.0
        else:
            a_house = 0.5  # uncertain

        # Number token Jaccard (NEW) — e.g. "1795 … 570" vs "1795 … 570"
        num1, num2 = _number_tokens(a1), _number_tokens(a2)
        if num1 or num2:
            inter = len(num1 & num2)
            union = len(num1 | num2)
            a_num_jacc = inter / union if union else 0.0
        else:
            a_num_jacc = 1.0  # both have no numbers → equally empty

        # ── MISC ────────────────────────────────────────────────────────────
        c_match = 1.0 if (c1 and c1 == c2) else 0.0

        rows.append([
            id1, id2,
            n_lev, n_tsort, n_tset, n_jw, n_jc,
            n_exact, n_ftok, n_ldiff, n_lratio, n_tcdiff, n_sdx,
            a_lev, a_tsort, a_jw, a_jc,
            a_house, a_empty, a_ldiff, a_num_jacc,
            c_match,
        ])

    COLS = [
        "source1_entity_id", "candidate_entity_id",
        # name
        "name_lev", "name_tsort", "name_tset",
        "name_jacc_word", "name_jacc_c3",
        "name_exact", "name_first_tok",
        "name_len_diff", "name_len_ratio",
        "name_tok_count_diff", "name_soundex_match",    # NEW
        # addr
        "addr_lev", "addr_tsort",
        "addr_jacc_word", "addr_jacc_c3",
        "addr_house_match", "addr_empty",
        "addr_len_diff", "addr_num_jacc",               # NEW
        # misc
        "country_match",
    ]
    feat_df = pd.DataFrame(rows, columns=COLS)

    # ── TF-IDF cosine (batch) ──────────────────────────────────────────────
    if tfidf and tfidf.fitted:
        na = [rec_map.get(i, {}).get("name_norm", "") for i in id1_col]
        nb = [rec_map.get(i, {}).get("name_norm", "") for i in id2_col]
        aa = [rec_map.get(i, {}).get("addr_norm", "") for i in id1_col]
        ab = [rec_map.get(i, {}).get("addr_norm", "") for i in id2_col]
        cos = tfidf.cosine_batch(na, nb, aa, ab)
        feat_df["name_cos_char"] = cos[:, 0].astype(np.float32)
        feat_df["name_cos_word"] = cos[:, 1].astype(np.float32)
        feat_df["addr_cos_char"] = cos[:, 2].astype(np.float32)
    else:
        feat_df["name_cos_char"] = np.float32(0.0)
        feat_df["name_cos_word"] = np.float32(0.0)
        feat_df["addr_cos_char"] = np.float32(0.0)

    if "label" in pairs_df.columns:
        feat_df["label"] = pairs_df["label"].values

    return feat_df


# ─── Feature column list (for Person 3 to know what to use) ──────────────────

FEATURE_COLS = [
    "name_lev", "name_tsort", "name_tset",
    "name_jacc_word", "name_jacc_c3",
    "name_exact", "name_first_tok",
    "name_len_diff", "name_len_ratio",
    "name_tok_count_diff", "name_soundex_match",
    "addr_lev", "addr_tsort",
    "addr_jacc_word", "addr_jacc_c3",
    "addr_house_match", "addr_empty",
    "addr_len_diff", "addr_num_jacc",
    "country_match",
    "name_cos_char", "name_cos_word", "addr_cos_char",
]
