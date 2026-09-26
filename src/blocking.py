from __future__ import annotations
import pandas as pd
import numpy as np
from collections import defaultdict  # kept for _build_index, _build_country_index
from typing import Dict, List, Optional, Set

_STOPWORDS = {
    "pvt", "ltd", "inc", "llc", "llp", "corp", "co", "and", "the",
    "of", "for", "at", "in", "on", "by", "de", "du", "la", "le",
    "sa", "sarl", "sas", "sci", "eurl", "group", "services", "solutions",
    "enterprises", "industries", "associates", "consultants",
}

PREFIX_LEN = 5
ADDR_MIN_LEN = 4


def _build_index(df: pd.DataFrame) -> dict:
    """Build inverted index: token -> set of entity_ids. Fast zip-based loop."""
    index = defaultdict(set)
    for eid, name_norm, addr_norm in zip(
        df["entity_id"],
        df["name_norm"].fillna(""),
        df["addr_norm"].fillna(""),
    ):
        for tok in name_norm.split():
            if len(tok) > 2 and tok not in _STOPWORDS:
                index[tok].add(eid)
        if len(name_norm) >= PREFIX_LEN:
            index["pfx:" + name_norm[:PREFIX_LEN]].add(eid)
        for tok in addr_norm.split():
            if len(tok) >= ADDR_MIN_LEN and not tok.isdigit() and tok not in _STOPWORDS:
                index["addr:" + tok].add(eid)
                
    # Filter out overly common tokens (dynamic stop-words)
    # If a token matches > 2000 records, it's not a useful blocking key
    for k in list(index.keys()):
        if len(index[k]) > 2000:
            del index[k]
            
    return dict(index)


def _build_country_index(df: pd.DataFrame) -> dict:
    """country -> set of entity_ids."""
    idx = defaultdict(set)
    for eid, country in zip(df["entity_id"], df["country"].fillna("")):
        if country:
            idx[country].add(eid)
        idx[""].add(eid)
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
    cands = set()
    for tok in name_norm.split():
        if len(tok) > 2 and tok not in _STOPWORDS:
            cands.update(index_s2.get(tok, set()))
            cands.update(index_s3.get(tok, set()))
    if len(name_norm) >= PREFIX_LEN:
        pfx = "pfx:" + name_norm[:PREFIX_LEN]
        cands.update(index_s2.get(pfx, set()))
        cands.update(index_s3.get(pfx, set()))
    for tok in addr_norm.split():
        if len(tok) >= ADDR_MIN_LEN and not tok.isdigit() and tok not in _STOPWORDS:
            key = "addr:" + tok
            cands.update(index_s2.get(key, set()))
            cands.update(index_s3.get(key, set()))
    if country:
        c2_c = country_s2.get(country, set())
        c2_e = country_s2.get("", set())
        c3_c = country_s3.get(country, set())
        c3_e = country_s3.get("", set())
        
        cands = {c for c in cands if (
            c in c2_c or c in c2_e or c in c3_c or c in c3_e
        )}
        
    return list(cands)[:max_candidates]


def _parse_gt_fast(gt_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Fast GT parsing using vectorized pandas — NO iterrows.
    Handles 2.2M row ground truth in seconds instead of minutes.
    """
    gt_lookup: Dict[str, Set[str]] = {}
    # Only process rows that have actual matches
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
    max_candidates: int = 300,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate candidate pairs for all S1 entities.
    If gt_df is given (training mode), adds label column and guarantees
    all true matches appear in candidates.
    """
    if verbose:
        print(f"  [Blocking] Building index on {len(s2_df)} S2 + {len(s3_df)} S3 records...")

    index_s2 = _build_index(s2_df)
    index_s3 = _build_index(s3_df)
    country_s2 = _build_country_index(s2_df)
    country_s3 = _build_country_index(s3_df)

    if verbose:
        print(f"  [Blocking] Index ready  S2 keys: {len(index_s2)}  S3 keys: {len(index_s3)}")

    # Fast GT parsing (vectorized, not iterrows)
    gt_lookup: Dict[str, Set[str]] = {}
    if gt_df is not None:
        if verbose:
            print(f"  [Blocking] Parsing ground truth ({len(gt_df)} rows)...")
        gt_lookup = _parse_gt_fast(gt_df)
        if verbose:
            print(f"  [Blocking] GT parsed: {len(gt_lookup)} S1 entities have matches")

    valid_all = set(s2_df["entity_id"]) | set(s3_df["entity_id"])
    total = len(s1_df)
    rows = []

    for i, (_, s1_row) in enumerate(s1_df.iterrows()):
        if verbose and i > 0 and i % 5000 == 0:
            print(f"  [Blocking] {i}/{total} processed...")

        s1_id   = s1_row["entity_id"]
        name_n  = s1_row.get("name_norm", "")
        addr_n  = s1_row.get("addr_norm", "")
        country = s1_row.get("country", "")

        cands = _candidates_for_one(
            name_n, addr_n, country,
            index_s2, index_s3, country_s2, country_s3, max_candidates,
        )

        # Training mode: guarantee true matches are included
        true_matches = gt_lookup.get(s1_id, set())
        if true_matches:
            cands_set = set(cands)
            for tm in true_matches:
                if tm in valid_all and tm not in cands_set:
                    cands.append(tm)
                    cands_set.add(tm)

        for c_id in cands:
            if c_id not in valid_all:
                continue
            rec = {"source1_entity_id": s1_id, "candidate_entity_id": c_id}
            if gt_df is not None:
                rec["label"] = 1 if (true_matches and c_id in true_matches) else 0
            rows.append(rec)

    pairs_df = (
        pd.DataFrame(rows)
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
