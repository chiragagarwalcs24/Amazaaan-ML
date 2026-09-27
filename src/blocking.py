from __future__ import annotations
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Set

_STOPWORDS = {
    "pvt", "ltd", "inc", "llc", "llp", "corp", "co", "and", "the",
    "of", "for", "at", "in", "on", "by", "de", "du", "la", "le",
    "sa", "sarl", "sas", "sci", "eurl", "group", "services", "solutions",
    "enterprises", "industries", "associates", "consultants",
}

PREFIX_LEN = 5
ADDR_MIN_LEN = 4
# Max hits a single token can contribute — skip hot tokens at query time
MAX_TOKEN_HITS = 500


def _build_index(df: pd.DataFrame) -> dict:
    """Build inverted index: token -> list of entity_ids. Fast zip-based loop."""
    index = defaultdict(list)
    for eid, name_norm, addr_norm in zip(
        df["entity_id"],
        df["name_norm"].fillna(""),
        df["addr_norm"].fillna(""),
    ):
        for tok in name_norm.split():
            if len(tok) >= 2 and tok not in _STOPWORDS:
                index[tok].append(eid)
        if len(name_norm) >= PREFIX_LEN:
            index["pfx:" + name_norm[:PREFIX_LEN]].append(eid)
        for tok in addr_norm.split():
            if len(tok) >= ADDR_MIN_LEN and not tok.isdigit() and tok not in _STOPWORDS:
                index["addr:" + tok].append(eid)

    # Filter out overly common tokens — they're useless for matching
    max_freq = min(MAX_TOKEN_HITS, max(100, int(len(df) * 0.01)))
    return {k: v for k, v in index.items() if len(v) <= max_freq}


def _build_country_index(df: pd.DataFrame) -> dict:
    """country -> set of entity_ids. Only real country codes, NO catch-all."""
    idx = defaultdict(set)
    for eid, country in zip(df["entity_id"], df["country"].fillna("")):
        if country:  # Only index known countries — no catch-all "" key
            idx[country].add(eid)
    return dict(idx)


def _candidates_for_one(
    name_norm: str,
    addr_norm: str,
    country: str,
    index_s2: dict,
    index_s3: dict,
    country_s2: dict,
    country_s3: dict,
    max_candidates: int,
) -> List[str]:
    """Get candidate IDs from S2+S3 for a single S1 record."""
    cands_count: Dict[str, int] = {}

    def _add_hits(hit_list):
        # Skip if token is too common (already filtered in index, but safety check)
        if len(hit_list) > MAX_TOKEN_HITS:
            return
        for cid in hit_list:
            cands_count[cid] = cands_count.get(cid, 0) + 1

    for tok in name_norm.split():
        if len(tok) >= 2 and tok not in _STOPWORDS:
            hits2 = index_s2.get(tok)
            if hits2:
                _add_hits(hits2)
            hits3 = index_s3.get(tok)
            if hits3:
                _add_hits(hits3)

    if len(name_norm) >= PREFIX_LEN:
        pfx = "pfx:" + name_norm[:PREFIX_LEN]
        hits2 = index_s2.get(pfx)
        if hits2:
            _add_hits(hits2)
        hits3 = index_s3.get(pfx)
        if hits3:
            _add_hits(hits3)

    for tok in addr_norm.split():
        if len(tok) >= ADDR_MIN_LEN and not tok.isdigit() and tok not in _STOPWORDS:
            key = "addr:" + tok
            hits2 = index_s2.get(key)
            if hits2:
                _add_hits(hits2)
            hits3 = index_s3.get(key)
            if hits3:
                _add_hits(hits3)

    # Country filter: only restrict if we know the country AND have data for it
    # Do NOT use a catch-all set — that was the performance killer
    if country and cands_count:
        c2_set = country_s2.get(country)
        c3_set = country_s3.get(country)
        if c2_set or c3_set:
            c2_s = c2_set or set()
            c3_s = c3_set or set()
            cands_count = {c: v for c, v in cands_count.items() if (c in c2_s) or (c in c3_s)}
    # Rank by token overlap count
    sorted_cands = sorted(cands_count, key=lambda k: cands_count[k], reverse=True)
    return sorted_cands[:max_candidates]


def _parse_gt_fast(gt_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Fast GT parsing — NO iterrows."""
    gt_lookup: Dict[str, Set[str]] = {}
    has_match = gt_df["matched_entity_ids"].fillna("").str.strip() != ""
    gt_filtered = gt_df[has_match]
    for s1_id, m_str in zip(gt_filtered["source1_entity_id"], gt_filtered["matched_entity_ids"]):
        ids = {m.strip() for m in str(m_str).split(",") if m.strip()}
        gt_lookup[s1_id] = ids
    return gt_lookup


def build_candidate_pairs(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    gt_df: Optional[pd.DataFrame] = None,
    max_candidates: int = 100,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate candidate pairs for all S1 entities.
    Uses itertuples() + filtered index for maximum speed.
    """
    if verbose:
        print(f"  [Blocking] Building index on {len(s2_df)} S2 + {len(s3_df)} S3 records...")

    index_s2 = _build_index(s2_df)
    index_s3 = _build_index(s3_df)
    country_s2 = _build_country_index(s2_df)
    country_s3 = _build_country_index(s3_df)

    if verbose:
        print(f"  [Blocking] Index ready  S2 keys: {len(index_s2)}  S3 keys: {len(index_s3)}")

    gt_lookup: Dict[str, Set[str]] = {}
    if gt_df is not None:
        if verbose:
            print(f"  [Blocking] Parsing ground truth ({len(gt_df)} rows)...")
        gt_lookup = _parse_gt_fast(gt_df)
        if verbose:
            print(f"  [Blocking] GT parsed: {len(gt_lookup)} S1 entities have matches")

    valid_all = set(s2_df["entity_id"]) | set(s3_df["entity_id"])
    total = len(s1_df)

    cols = list(s1_df.columns)
    eid_idx     = cols.index("entity_id")
    name_idx    = cols.index("name_norm")   if "name_norm"  in cols else -1
    addr_idx    = cols.index("addr_norm")   if "addr_norm"  in cols else -1
    country_idx = cols.index("country")     if "country"    in cols else -1

    s1_ids_out   = []
    cand_ids_out = []
    labels_out   = [] if gt_df is not None else None

    print(f"  [Blocking] Processing {total} S1 entities...", flush=True)
    for i, row in enumerate(s1_df.itertuples(index=False, name=None)):
        if verbose and i > 0 and i % 1000 == 0:
            print(f"  [Blocking] {i}/{total} processed...", flush=True)

        s1_id   = row[eid_idx]
        name_n  = row[name_idx]    if name_idx    >= 0 else ""
        addr_n  = row[addr_idx]    if addr_idx    >= 0 else ""
        country = row[country_idx] if country_idx >= 0 else ""

        name_n  = name_n  if isinstance(name_n,  str) else ""
        addr_n  = addr_n  if isinstance(addr_n,  str) else ""
        country = country if isinstance(country, str) else ""

        cands = _candidates_for_one(
            name_n, addr_n, country,
            index_s2, index_s3, country_s2, country_s3, max_candidates,
        )

        true_matches = gt_lookup.get(s1_id, set()) if gt_df is not None else set()

        # Inject true matches missed by blocker (training only)
        if true_matches:
            cands_set = set(cands)
            for tm in true_matches:
                if tm in valid_all and tm not in cands_set:
                    cands.append(tm)
                    cands_set.add(tm)

        for c_id in cands:
            if c_id not in valid_all:
                continue
            s1_ids_out.append(s1_id)
            cand_ids_out.append(c_id)
            if labels_out is not None:
                labels_out.append(1 if (true_matches and c_id in true_matches) else 0)

    if labels_out is not None:
        pairs_df = pd.DataFrame({
            "source1_entity_id":  s1_ids_out,
            "candidate_entity_id": cand_ids_out,
            "label": labels_out,
        })
    else:
        pairs_df = pd.DataFrame({
            "source1_entity_id":  s1_ids_out,
            "candidate_entity_id": cand_ids_out,
        })

    pairs_df = (
        pairs_df
        .drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"])
        .reset_index(drop=True)
    )

    if verbose:
        n_s1 = pairs_df["source1_entity_id"].nunique()
        n_pairs = len(pairs_df)
        if gt_df is not None:
            n_pos = (pairs_df["label"] == 1).sum()
            n_neg = (pairs_df["label"] == 0).sum()
            print(f"  [Blocking] {n_s1} S1 -> {n_pairs} pairs ({n_pos} pos / {n_neg} neg)")
        else:
            print(f"  [Blocking] {n_s1} S1 -> {n_pairs} candidate pairs")

    return pairs_df


def evaluate_blocking_recall(pairs_df, gt_df, s1_ids):
    """Recall: fraction of true matches that appear in candidates."""
    gt_lookup = _parse_gt_fast(gt_df)
    gt_lookup = {k: v for k, v in gt_lookup.items() if k in s1_ids}

    cand_lookup = (
        pairs_df.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(set).to_dict()
    )

    found = total = 0
    for s1_id, true_set in gt_lookup.items():
        found += len(true_set & cand_lookup.get(s1_id, set()))
        total += len(true_set)

    recall = found / total if total > 0 else 1.0
    print(f"  [Blocking Recall] {found}/{total} matches found -> {recall:.4f} ({recall*100:.1f}%)")
    return recall
