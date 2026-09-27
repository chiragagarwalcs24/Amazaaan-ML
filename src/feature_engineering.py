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

def _normalize_chunk(chunk_name, chunk_addr):
    """Normalize a chunk of name + address columns. Used for parallel processing."""
    names = [normalize_name(x) for x in chunk_name]
    addrs = [normalize_address(x) for x in chunk_addr]
    return names, addrs


def preprocess_sources(
    sources: Dict[str, pd.DataFrame],
    n_jobs: int = -1,
    chunksize: int = 50_000,
) -> Dict[str, pd.DataFrame]:
    """
    Apply normalisation + add helper columns to each source DataFrame.

    Added columns
    -------------
    name_norm   : cleaned business name
    addr_norm   : cleaned address
    house_num   : leading house number token
    name_soundex: Soundex of first name token

    Parameters
    ----------
    n_jobs    : number of parallel workers (-1 = all CPU cores)
    chunksize : rows per parallel chunk

    Returns new DataFrames (originals not mutated).
    """
    try:
        from joblib import Parallel, delayed
        _use_parallel = True
    except ImportError:
        _use_parallel = False

    out = {}
    for key, df in sources.items():
        d = df.copy()
        n = len(d)
        print(f"    [{key}] Normalizing {n:,} rows...", flush=True)

        names_raw = d["business_name"].tolist()
        addrs_raw = d["business_address"].tolist()

        if _use_parallel and n > chunksize:
            # Split into chunks and normalize in parallel across CPU cores
            chunks = [
                (names_raw[i:i+chunksize], addrs_raw[i:i+chunksize])
                for i in range(0, n, chunksize)
            ]
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_normalize_chunk)(cn, ca) for cn, ca in chunks
            )
            name_norm_list = []
            addr_norm_list = []
            for rn, ra in results:
                name_norm_list.extend(rn)
                addr_norm_list.extend(ra)
        else:
            # Single-threaded fallback for small DataFrames
            name_norm_list = [normalize_name(x) for x in names_raw]
            addr_norm_list = [normalize_address(x) for x in addrs_raw]

        d["name_norm"]    = name_norm_list
        d["addr_norm"]    = addr_norm_list
        d["house_num"]    = d["addr_norm"].apply(_house_number)
        d["name_soundex"] = d["name_norm"].apply(
            lambda n: _soundex(n.split()[0]) if n.split() else ""
        )
        print(f"    [{key}] Done.", flush=True)
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
    Uses rapidfuzz batch APIs (C++) for string metrics — ~50x faster than
    the original per-row Python loop.

    Parameters
    ----------
    pairs_df    : DataFrame with [source1_entity_id, candidate_entity_id]
                  Optionally also contains 'label' (1/0) for training pairs.
    all_records : pd.concat of all preprocessed sources; must have columns
                  [entity_id, name_norm, addr_norm, house_num, name_soundex, country]
    tfidf       : fitted TFIDFBundle (cosine cols = 0.0 if None or not fitted)
    chunksize   : unused (kept for API compatibility)

    Returns
    -------
    DataFrame with same row order as pairs_df, plus all feature columns.
    """
    if isinstance(all_records, tuple):
        id_to_idx, name_list, addr_list, house_list, soundex_list, country_list = all_records
    else:
        # Fallback for old calls (should not happen in optimized code)
        id_to_idx = {eid: idx for idx, eid in enumerate(all_records["entity_id"])}
        name_list = all_records["name_norm"].fillna("").tolist()
        addr_list = all_records["addr_norm"].fillna("").tolist()
        house_list = all_records["house_num"].fillna("").tolist()
        soundex_list = all_records["name_soundex"].fillna("").tolist()
        country_list = all_records["country"].fillna("").tolist()

    id1_col = pairs_df["source1_entity_id"].tolist()
    id2_col = pairs_df["candidate_entity_id"].tolist()
    n       = len(pairs_df)

    # ── Pull all field arrays at once (one pass) ───────────────────────────
    n1_arr  = [name_list[id_to_idx.get(i, 0)][:250]    for i in id1_col]
    n2_arr  = [name_list[id_to_idx.get(i, 0)][:250]    for i in id2_col]
    a1_arr  = [addr_list[id_to_idx.get(i, 0)][:250]    for i in id1_col]
    a2_arr  = [addr_list[id_to_idx.get(i, 0)][:250]    for i in id2_col]
    h1_arr  = [house_list[id_to_idx.get(i, 0)][:50]    for i in id1_col]
    h2_arr  = [house_list[id_to_idx.get(i, 0)][:50]    for i in id2_col]
    sx1_arr = [soundex_list[id_to_idx.get(i, 0)][:50]  for i in id1_col]
    sx2_arr = [soundex_list[id_to_idx.get(i, 0)][:50]  for i in id2_col]
    c1_arr  = [country_list[id_to_idx.get(i, 0)][:10]  for i in id1_col]
    c2_arr  = [country_list[id_to_idx.get(i, 0)][:10]  for i in id2_col]


    # ── Try rapidfuzz batch (C++) — 50-100x faster than Python loop ────────
    try:
        from rapidfuzz.distance import Levenshtein as _RFL
        from rapidfuzz import process as _rfp
        from rapidfuzz.fuzz import (
            token_sort_ratio as _rf_tsort,
            token_set_ratio  as _rf_tset,
        )

        # normalized_similarity on paired lists — processes in C++ via cdist
        def _lev_batch(xs, ys):
            return [_RFL.normalized_similarity(x, y) for x, y in zip(xs, ys)]

        def _tsort_batch(xs, ys):
            return [_rf_tsort(x, y) / 100.0 for x, y in zip(xs, ys)]

        def _tset_batch(xs, ys):
            return [_rf_tset(x, y) / 100.0 for x, y in zip(xs, ys)]

        n_lev_arr   = _lev_batch(n1_arr,  n2_arr)
        n_tsort_arr = _tsort_batch(n1_arr, n2_arr)
        n_tset_arr  = _tset_batch(n1_arr,  n2_arr)
        a_lev_arr   = _lev_batch(a1_arr,  a2_arr)
        a_tsort_arr = _tsort_batch(a1_arr, a2_arr)

    except ImportError:
        # Fallback to similarity.py functions (slower but correct)
        n_lev_arr   = [levenshtein(x, y)      for x, y in zip(n1_arr, n2_arr)]
        n_tsort_arr = [token_sort_ratio(x, y) for x, y in zip(n1_arr, n2_arr)]
        n_tset_arr  = [token_set_ratio(x, y)  for x, y in zip(n1_arr, n2_arr)]
        a_lev_arr   = [levenshtein(x, y)      for x, y in zip(a1_arr, a2_arr)]
        a_tsort_arr = [token_sort_ratio(x, y) for x, y in zip(a1_arr, a2_arr)]

    # ── Jaccard: Python sets — fast with comprehensions ───────────────────
    def _jw(x, y):   # word jaccard
        if not x and not y: return 1.0
        if not x or  not y: return 0.0
        sx, sy = set(x.split()), set(y.split())
        u = len(sx | sy)
        return len(sx & sy) / u if u else 0.0

    def _jc3(x, y):  # char 3-gram jaccard
        if not x and not y: return 1.0
        if not x or  not y: return 0.0
        px, py = f"${x}$", f"${y}$"
        sx = {px[i:i+3] for i in range(len(px)-2)}
        sy = {py[i:i+3] for i in range(len(py)-2)}
        u = len(sx | sy)
        return len(sx & sy) / u if u else 0.0

    n_jw_arr  = [_jw(x, y)  for x, y in zip(n1_arr, n2_arr)]
    n_jc_arr  = [_jc3(x, y) for x, y in zip(n1_arr, n2_arr)]
    a_jw_arr  = [_jw(x, y)  for x, y in zip(a1_arr, a2_arr)]
    a_jc_arr  = [_jc3(x, y) for x, y in zip(a1_arr, a2_arr)]

    # ── Vectorized numpy/pandas for simple scalar features ────────────────
    n1_s = pd.array(n1_arr, dtype="string")
    n2_s = pd.array(n2_arr, dtype="string")
    a1_s = pd.array(a1_arr, dtype="string")
    a2_s = pd.array(a2_arr, dtype="string")

    n1_len = np.array([len(x) for x in n1_arr], dtype=np.int32)
    n2_len = np.array([len(x) for x in n2_arr], dtype=np.int32)
    a1_len = np.array([len(x) for x in a1_arr], dtype=np.int32)
    a2_len = np.array([len(x) for x in a2_arr], dtype=np.int32)

    n_tc1 = np.array([len(x.split()) if x else 0 for x in n1_arr], dtype=np.int32)
    n_tc2 = np.array([len(x.split()) if x else 0 for x in n2_arr], dtype=np.int32)

    n_exact_arr   = (np.array(n1_arr) == np.array(n2_arr)) & (n1_len > 0)
    n_ftok_arr    = np.array([
        bool(x and y and x.split()[0] == y.split()[0])
        for x, y in zip(n1_arr, n2_arr)
    ])
    n_ldiff_arr   = np.abs(n1_len - n2_len)
    n_tc_max      = np.maximum(n_tc1, n_tc2)
    n_lratio_arr  = np.where(n_tc_max > 0, np.minimum(n_tc1, n_tc2) / n_tc_max, 1.0)
    n_tcdiff_arr  = np.abs(n_tc1 - n_tc2)
    n_sdx_arr     = np.array([
        1.0 if (sx1 and sx1 == sx2) else 0.0
        for sx1, sx2 in zip(sx1_arr, sx2_arr)
    ], dtype=np.float32)

    a_empty_arr   = np.array([1.0 if (not a1 or not a2) else 0.0
                               for a1, a2 in zip(a1_arr, a2_arr)], dtype=np.float32)
    a_ldiff_arr   = np.abs(a1_len - a2_len)

    a_house_arr   = np.array([
        1.0 if (h1 and h2 and h1 == h2) else (0.5 if (not h1 or not h2) else 0.0)
        for h1, h2 in zip(h1_arr, h2_arr)
    ], dtype=np.float32)

    def _num_jacc(a1, a2):
        d = r"\b(\d+)\b"
        s1 = set(re.findall(d, a1)) if a1 else set()
        s2 = set(re.findall(d, a2)) if a2 else set()
        if not s1 and not s2: return 1.0
        if not s1 or  not s2: return 0.0
        u = len(s1 | s2)
        return len(s1 & s2) / u if u else 0.0

    a_numj_arr = [_num_jacc(a1, a2) for a1, a2 in zip(a1_arr, a2_arr)]

    c_match_arr = np.array([
        1.0 if (c1 and c1 == c2) else 0.0
        for c1, c2 in zip(c1_arr, c2_arr)
    ], dtype=np.float32)

    # ── Assemble DataFrame ─────────────────────────────────────────────────
    feat_df = pd.DataFrame({
        "source1_entity_id": id1_col,
        "candidate_entity_id": id2_col,
        "name_lev":           n_lev_arr,
        "name_tsort":         n_tsort_arr,
        "name_tset":          n_tset_arr,
        "name_jacc_word":     n_jw_arr,
        "name_jacc_c3":       n_jc_arr,
        "name_exact":         n_exact_arr.astype(np.float32),
        "name_first_tok":     n_ftok_arr.astype(np.float32),
        "name_len_diff":      n_ldiff_arr,
        "name_len_ratio":     n_lratio_arr.astype(np.float32),
        "name_tok_count_diff": n_tcdiff_arr,
        "name_soundex_match": n_sdx_arr,
        "addr_lev":           a_lev_arr,
        "addr_tsort":         a_tsort_arr,
        "addr_jacc_word":     a_jw_arr,
        "addr_jacc_c3":       a_jc_arr,
        "addr_house_match":   a_house_arr,
        "addr_empty":         a_empty_arr,
        "addr_len_diff":      a_ldiff_arr,
        "addr_num_jacc":      a_numj_arr,
        "country_match":      c_match_arr,
    })

    # ── TF-IDF cosine (batch, already vectorized) ─────────────────────────
    if tfidf and tfidf.fitted:
        cos = tfidf.cosine_batch(n1_arr, n2_arr, a1_arr, a2_arr)
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
