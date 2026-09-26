"""
evaluate.py
-----------
Person 2 – Local F_0.5 Evaluation Utility

Implements the exact scoring formula from the competition README:
    F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)

Computed as MACRO-AVERAGE across all Source 1 entities (including singletons).

Usage (from feature matrix + simple threshold)
----------------------------------------------
    python src/evaluate.py

Or import and call from Person 3's model script:
    from src.evaluate import f05_score, evaluate_predictions
"""

from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from typing import Dict, List, Set


# ─── Core metric ──────────────────────────────────────────────────────────────

def f05_score(precision: float, recall: float) -> float:
    """
    F_beta score with beta=0.5 (precision-heavy).

    F_0.5 = (1 + 0.5²) × P × R / (0.5² × P + R)
           = 1.25 × P × R / (0.25 × P + R)

    Returns 0.0 if both P and R are 0.
    Returns 1.0 if both are 1.
    """
    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.25 * precision * recall) / denom


def entity_f05(
    pred_set: Set[str],
    true_set: Set[str],
) -> float:
    """
    Compute F_0.5 for a single Source 1 entity.

    Parameters
    ----------
    pred_set : set of predicted match IDs (may be empty)
    true_set : set of ground truth match IDs (may be empty)

    Special cases (from README):
    - both empty → 1.0  (correctly predicted singleton)
    - pred empty, true non-empty → 0.0  (missed all matches)
    - pred non-empty, true empty → 0.0  (false merge on singleton)
    """
    if not pred_set and not true_set:
        return 1.0
    if not pred_set or not true_set:
        return 0.0
    tp        = len(pred_set & true_set)
    precision = tp / len(pred_set)
    recall    = tp / len(true_set)
    return f05_score(precision, recall)


def macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> float:
    """
    Macro-averaged F_0.5 over all Source 1 entities.

    Parameters
    ----------
    predictions  : {s1_id → set of predicted match IDs}
    ground_truth : {s1_id → set of true match IDs}
                   Every S1 entity must be a key here.

    Returns float in [0, 1].
    """
    scores = []
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        scores.append(entity_f05(pred_set, true_set))
    return float(np.mean(scores)) if scores else 0.0


# ─── Parsing helpers ──────────────────────────────────────────────────────────

def parse_gt(gt_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Parse ground truth DataFrame into {s1_id → set of match IDs}."""
    result = {}
    for _, row in gt_df.iterrows():
        s1_id  = row["source1_entity_id"]
        m_str  = row.get("matched_entity_ids", "")
        ids    = {m.strip() for m in str(m_str).split(",") if m.strip()}
        result[s1_id] = ids
    return result


def parse_predictions(matching_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Parse a matching_results DataFrame into {s1_id → set of match IDs}."""
    result = {}
    for _, row in matching_df.iterrows():
        s1_id  = row["source1_entity_id"]
        m_str  = row.get("matched_entity_ids", "")
        ids    = {m.strip() for m in str(m_str).split(",") if m.strip()}
        result[s1_id] = ids
    return result


# ─── Threshold-based evaluation on feature matrix ────────────────────────────

def evaluate_features_at_threshold(
    feat_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    score_col: str = "name_lev",
    thresholds: List[float] = None,
) -> pd.DataFrame:
    """
    Sweep a range of score thresholds on a single feature column and
    report precision, recall, and F_0.5 at each threshold.

    This is useful to judge which feature is most discriminative and to
    pick a default threshold for Person 3's simple baseline.

    Parameters
    ----------
    feat_df    : output of compute_features() with 'label' column
    gt_df      : raw ground truth DataFrame
    score_col  : name of the feature to sweep (higher = more similar)
    thresholds : list of threshold values; defaults to np.arange(0.1, 1.0, 0.05)

    Returns
    -------
    DataFrame with columns [threshold, precision, recall, f05]
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.05, 0.05).tolist()

    gt_dict = parse_gt(gt_df)
    all_s1  = set(gt_dict.keys())

    records = []
    for thr in thresholds:
        # Predict match if score ≥ threshold
        matched = feat_df[feat_df[score_col] >= thr]

        # Build predictions dict
        preds: Dict[str, Set[str]] = {s1: set() for s1 in all_s1}
        for _, row in matched.iterrows():
            s1_id = row["source1_entity_id"]
            c_id  = row["candidate_entity_id"]
            if s1_id in preds:
                preds[s1_id].add(c_id)

        score = macro_f05(preds, gt_dict)

        # Micro precision / recall from labels
        pos_pred = (feat_df[score_col] >= thr)
        tp = (pos_pred & (feat_df["label"] == 1)).sum()
        fp = (pos_pred & (feat_df["label"] == 0)).sum()
        fn = (~pos_pred & (feat_df["label"] == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        records.append({
            "threshold": round(thr, 2),
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f05":       round(score, 4),
        })

    return pd.DataFrame(records)


# ─── Demo ─────────────────────────────────────────────────────────────────────

def main():
    from src.data_loader import load_source, load_ground_truth
    from src.feature_engineering import preprocess_sources, TFIDFBundle, compute_features, FEATURE_COLS
    import warnings
    warnings.filterwarnings("ignore")

    SEP = "=" * 72

    print(f"\n{SEP}")
    print("  PERSON 2 · LOCAL F_0.5 EVALUATION")
    print(f"{SEP}\n", flush=True)

    # Load sample
    S1_N, S2_N, S3_N = 5_000, 15_000, 15_000
    print(f"[1] Loading {S1_N:,} S1 / {S2_N:,} S2 / {S3_N:,} S3 rows …", flush=True)
    s1 = load_source("train", 1).head(S1_N)
    s2 = load_source("train", 2).head(S2_N)
    s3 = load_source("train", 3).head(S3_N)
    gt = load_ground_truth()

    # Preprocess
    print("[2] Normalising …", flush=True)
    proc = preprocess_sources({"source1": s1, "source2": s2, "source3": s3})
    all_records = pd.concat(list(proc.values()), ignore_index=True)

    # Build ALL positive pairs where both S1 and candidate are in our pool
    s1_ids   = set(s1["entity_id"])
    s2_s3_ids = set(s2["entity_id"]) | set(s3["entity_id"])
    gt_s = gt[gt["source1_entity_id"].isin(s1_ids)]

    pairs = []
    for _, row in gt_s.iterrows():
        s1_id = row["source1_entity_id"]
        for m_id in row["matched_entity_ids"].split(","):
            m_id = m_id.strip()
            if m_id and m_id in s2_s3_ids:
                pairs.append({"source1_entity_id": s1_id,
                              "candidate_entity_id": m_id, "label": 1})

    # Add random negatives (3× positive count for balance)
    rng     = np.random.default_rng(0)
    neg_n   = min(len(pairs) * 3, 30_000)
    s2_list = s2["entity_id"].tolist()
    s3_list = s3["entity_id"].tolist()
    pos_map: Dict[str, set] = {}
    for p in pairs:
        pos_map.setdefault(p["source1_entity_id"], set()).add(p["candidate_entity_id"])
    for s1_id in rng.choice(list(s1_ids), size=min(neg_n // 5, len(s1_ids)), replace=False):
        for c_id in rng.choice(s2_list + s3_list, size=5, replace=False):
            if c_id not in pos_map.get(s1_id, set()):
                pairs.append({"source1_entity_id": s1_id,
                              "candidate_entity_id": c_id, "label": 0})

    pairs_df = pd.DataFrame(pairs).drop_duplicates(
        subset=["source1_entity_id", "candidate_entity_id"]
    ).reset_index(drop=True)

    n_pos = (pairs_df["label"] == 1).sum()
    n_neg = (pairs_df["label"] == 0).sum()
    print(f"   Pairs: {n_pos:,} positive / {n_neg:,} negative = {len(pairs_df):,} total")

    # Fit TF-IDF
    print("[3] Fitting TF-IDF …", flush=True)
    tfidf = TFIDFBundle(max_features=30_000).fit(
        all_records["name_norm"].tolist(),
        all_records["addr_norm"].tolist(),
    )

    # Extract features
    print("[4] Computing features …", flush=True)
    feat_df = compute_features(pairs_df, all_records, tfidf=tfidf)

    # Evaluate each feature at various thresholds — pick best single feature
    print(f"\n[5] Threshold sweep for key features:\n", flush=True)
    gt_dict = parse_gt(gt_s)

    best_results = {}
    for col in ["name_lev", "name_tsort", "name_tset",
                "name_jacc_word", "name_cos_char", "name_cos_word",
                "addr_lev", "name_soundex_match", "addr_num_jacc"]:
        if col not in feat_df.columns:
            continue
        sweep = evaluate_features_at_threshold(feat_df, gt_s, score_col=col)
        best_row = sweep.loc[sweep["f05"].idxmax()]
        best_results[col] = best_row
        print(f"  {col:<22}  best_thr={best_row.threshold:.2f}  "
              f"P={best_row.precision:.3f}  R={best_row.recall:.3f}  "
              f"F0.5={best_row.f05:.4f}")

    best_col = max(best_results, key=lambda c: best_results[c]["f05"])
    best_row = best_results[best_col]
    print(f"\n  ★ Best single feature: '{best_col}' at threshold={best_row.threshold:.2f}")
    print(f"    F_0.5 = {best_row.f05:.4f}  Precision = {best_row.precision:.3f}  "
          f"Recall = {best_row.recall:.3f}")

    # Save feature matrix
    out_path = "output/features_sample_train.csv"
    os.makedirs("output", exist_ok=True)
    feat_df.to_csv(out_path, index=False)
    print(f"\n[6] Feature matrix saved → {os.path.abspath(out_path)}")

    print(f"\n{SEP}")
    print("  EVALUATION COMPLETE")
    print(f"  Feature columns to pass to Person 3: {FEATURE_COLS}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
