"""
run_person2_pipeline.py
-----------------------
Person 2 – End-to-end pipeline demonstration.

What this script does
---------------------
1. Load a sample of the official dataset from /dataset/…
2. Demonstrate normalize_name() and normalize_address() on real rows
3. Demonstrate all similarity metrics with live examples
4. Preprocess sources → normalised DataFrames
5. Build TF-IDF vectorisers on the corpus
6. Extract a feature matrix for a representative set of candidate pairs
   (positive pairs from ground truth + random negative pairs)
7. Show feature stats and save output to  output/features_sample_train.csv

Run from the project root:
    python run_person2_pipeline.py
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure src/ is importable when run from project root
sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data_loader import load_source, load_ground_truth, DATASET_ROOT
from src.normalization import normalize_name, normalize_address
from src.similarity import levenshtein, jaccard, token_sort_ratio, token_set_ratio
from src.feature_engineering import preprocess_sources, TFIDFBundle, compute_features


# ─── Config ───────────────────────────────────────────────────────────────────
# How many S1 rows to sample for this demo run
S1_SAMPLE   = 3_000
# Extra random S2/S3 rows to add on top of the GT-matched rows (for negatives)
S2_EXTRA    = 5_000
S3_EXTRA    = 5_000
OUTPUT_DIR  = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEP = "=" * 78


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP, flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    section("PERSON 2 · DATA PREPROCESSING, NORMALIZATION & FEATURE ENGINEERING")

    # ── 1. Load official dataset ───────────────────────────────────────────────
    print(f"\n[1] Loading official dataset from:\n    {DATASET_ROOT}\n")

    s1 = load_source("train", 1).head(S1_SAMPLE)
    gt = load_ground_truth()

    # ── Smart S2/S3 loading ─────────────────────────────────────────────────
    # CRITICAL FIX: the ground truth matches may reference rows anywhere in
    # S2/S3 (e.g. row #200,000). If we blindly take .head(8000), nearly ALL
    # positive pairs will have no record in our pool → all features = 0.0.
    # Solution: collect the specific IDs we need from GT, load ALL of S2/S3
    # into a lookup, grab those rows, then add random extra rows for negatives.
    s1_ids_sample = set(s1["entity_id"])
    gt_sample     = gt[gt["source1_entity_id"].isin(s1_ids_sample)]

    needed_ids: set = set()
    for _, row in gt_sample.iterrows():
        m_str = row.get("matched_entity_ids", "")
        if m_str:
            for m_id in str(m_str).split(","):
                m_id = m_id.strip()
                if m_id:
                    needed_ids.add(m_id)

    print(f"    Source 1 sample        : {len(s1):>6,} rows")
    print(f"    GT-referenced S2/S3 IDs: {len(needed_ids):>6,} (needed for positive pairs)")

    # Load full S2 and S3 — needed to find the specific matching rows
    print("    Loading full Source 2 (this may take a moment)...")
    s2_full = load_source("train", 2)
    print("    Loading full Source 3...")
    s3_full = load_source("train", 3)

    # Extract the needed rows + random extras for negatives
    rng_load = np.random.default_rng(0)
    s2_needed = s2_full[s2_full["entity_id"].isin(needed_ids)]
    s3_needed = s3_full[s3_full["entity_id"].isin(needed_ids)]

    s2_extra_idx = rng_load.choice(len(s2_full), size=min(S2_EXTRA, len(s2_full)), replace=False)
    s3_extra_idx = rng_load.choice(len(s3_full), size=min(S3_EXTRA, len(s3_full)), replace=False)
    s2_extra = s2_full.iloc[s2_extra_idx]
    s3_extra = s3_full.iloc[s3_extra_idx]

    s2 = pd.concat([s2_needed, s2_extra]).drop_duplicates("entity_id").reset_index(drop=True)
    s3 = pd.concat([s3_needed, s3_extra]).drop_duplicates("entity_id").reset_index(drop=True)

    # Free memory
    del s2_full, s3_full

    print(f"    Source 2 final pool    : {len(s2):>6,} rows ({len(s2_needed):,} GT-matched + extras)")
    print(f"    Source 3 final pool    : {len(s3):>6,} rows ({len(s3_needed):,} GT-matched + extras)")
    print(f"    Ground truth           : {len(gt):>6,} rows  (full file)")

    # ── 2. Normalisation demo ──────────────────────────────────────────────────
    section("2 · Normalisation Demonstration")

    name_examples = [
        "Orelee's Barbershop",
        "-- Holloway Peak Inc Seafood",
        "B+ Retail Inc",
        "International South Consultants Private Ltd",
        "LLC Moncada Léarning Center",
        "ZNB Club SARL",                     # French entity from test set
        "<< Team Ecole",                     # noisy prefix from test set
        "Pvt. EFS Print Ventures Ltd.",
        "Delta Tetlecommunication Inc",
        "GREENSBORO SUMMIT CORPORATION",
    ]

    print("\n  ── Business Name Normalisation ──")
    print(f"  {'Raw Input':<45} → Normalised")
    print("  " + "-" * 70)
    for name in name_examples:
        print(f"  {name!r:<45} → {normalize_name(name)!r}")

    addr_examples = [
        "1795 Westchester Drive, High Point, NC",
        "105 ELM ST, MORGANTON, NC",
        "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
        "1 Ivanhoe Ave, PO Box 6009, Cincinnati, Ohio",
        "Mack Rd, Haltom City, Texas",
        "175 Boulevard du Président Franklin Roosevelt, Bordeaux",  # French
        "GREENSBORO, NC, 19 1/2 STARDUST TRAIL",
        "Door No 183, 41St Cross, 22Nd Main 9Th Block Jayanagar, Bengaluru",
    ]

    print("\n  ── Address Normalisation ──")
    print(f"  {'Raw Input':<55} → Normalised")
    print("  " + "-" * 90)
    for addr in addr_examples:
        print(f"  {addr!r:<55} → {normalize_address(addr)!r}")

    # ── 3. Similarity metrics demo ─────────────────────────────────────────────
    section("3 · Similarity Metrics Demonstration")

    pairs_demo = [
        ("holloway peak inc seafood", "holloway peak seafood inc",   "word-order transposition"),
        ("ram marketing pvt ltd",     "ram marketing private limited","legal suffix variation"),
        ("delta telecomm inc",        "delta tetlecommunication inc", "typo / abbrev"),
        ("summit inc",                "summit incorporated",          "suffix expansion"),
        ("1795 westchester drive",    "1795 westchester dr",          "address abbrev"),
        ("abc corp",                  "xyz llc",                      "completely different"),
    ]

    print(f"\n  {'String A':<32} {'String B':<32}  LEV  TSORT TSET  JACW  JACC3  Scenario")
    print("  " + "-" * 110)
    for a, b, scenario in pairs_demo:
        lev   = levenshtein(a, b)
        tsort = token_sort_ratio(a, b)
        tset  = token_set_ratio(a, b)
        jw    = jaccard(a, b, n_gram=1)
        jc    = jaccard(a, b, n_gram=3)
        print(f"  {a:<32} {b:<32}  {lev:.2f}  {tsort:.2f}  {tset:.2f}  {jw:.2f}  {jc:.2f}   {scenario}")

    # ── 4. Preprocess sources ──────────────────────────────────────────────────
    section("4 · Preprocessing (Normalising) Sources")

    sources = preprocess_sources({"source1": s1, "source2": s2, "source3": s3})
    all_records = pd.concat(list(sources.values()), ignore_index=True)
    print(f"\n    Total normalised records in pool : {len(all_records):,}")
    print(f"\n    Sample of normalised Source 1 rows:")
    cols_show = ["entity_id", "business_name", "name_norm", "business_address", "addr_norm"]
    print(sources["source1"][cols_show].head(5).to_string(index=False))

    # ── 5. TF-IDF fit ─────────────────────────────────────────────────────────
    section("5 · Fitting TF-IDF Vectorisers")

    tfidf = TFIDFBundle(max_features=30_000)
    tfidf.fit(
        names=all_records["name_norm"].tolist(),
        addrs=all_records["addr_norm"].tolist(),
    )
    print(f"\n    name_char vocab : {len(tfidf.name_char.vocabulary_):,} features")
    print(f"    name_word vocab : {len(tfidf.name_word.vocabulary_):,} features")
    print(f"    addr_char vocab : {len(tfidf.addr_char.vocabulary_):,} features")

    # ── 6. Build candidate pairs ───────────────────────────────────────────────
    section("6 · Building Candidate Pairs (Positive + Negative samples)")

    s1_ids = set(s1["entity_id"])
    gt_sample = gt[gt["source1_entity_id"].isin(s1_ids)]

    # Positive pairs from ground truth
    pairs = []
    for _, row in gt_sample.iterrows():
        s1_id  = row["source1_entity_id"]
        m_str  = row["matched_entity_ids"]
        if m_str:
            for m_id in m_str.split(","):
                m_id = m_id.strip()
                if m_id:
                    pairs.append({"source1_entity_id": s1_id,
                                  "candidate_entity_id": m_id,
                                  "label": 1})

    # Negative pairs (random S2/S3 IDs not in truth for that S1 entity)
    # BUG FIX: use the FULL ground truth to build pos_per_s1, not just
    # gt_sample.  If we only used gt_sample, a true match that happens to fall
    # outside our S2/S3 sample window could be picked as a "negative" pair,
    # silently injecting a wrong label=0 for a real match and hurting training.
    rng    = np.random.default_rng(42)
    s2_ids = s2["entity_id"].tolist()
    s3_ids = s3["entity_id"].tolist()

    # Build positive-set from FULL ground truth so we never mislabel true matches
    all_positives: dict = {}
    for _, grow in gt.iterrows():
        s1_id = grow["source1_entity_id"]
        m_str = grow.get("matched_entity_ids", "")
        if m_str:
            for m_id in str(m_str).split(","):
                m_id = m_id.strip()
                if m_id:
                    all_positives.setdefault(s1_id, set()).add(m_id)

    s1_list = list(s1_ids)[:200]           # use 200 S1 entities for negatives
    for s1_id in s1_list:
        pos_set = all_positives.get(s1_id, set())
        cands   = [i for i in rng.choice(s2_ids, 5, replace=False).tolist() if i not in pos_set]
        cands  += [i for i in rng.choice(s3_ids, 5, replace=False).tolist() if i not in pos_set]
        for c in cands:
            pairs.append({"source1_entity_id": s1_id,
                          "candidate_entity_id": c,
                          "label": 0})

    pairs_df = (
        pd.DataFrame(pairs)
        .drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"])
        .reset_index(drop=True)
    )
    n_pos = (pairs_df["label"] == 1).sum()
    n_neg = (pairs_df["label"] == 0).sum()
    print(f"\n    Positive pairs : {n_pos:,}")
    print(f"    Negative pairs : {n_neg:,}")
    print(f"    Total pairs    : {len(pairs_df):,}")

    # ── 7. Extract features ────────────────────────────────────────────────────
    section("7 · Extracting Feature Matrix")

    feat_df = compute_features(pairs_df, all_records, tfidf=tfidf)
    print(f"\n    Shape of feature matrix : {feat_df.shape}")

    print("\n    First 5 rows (selected columns):")
    preview_cols = [
        "source1_entity_id", "candidate_entity_id",
        "name_lev", "name_tsort", "name_tset",
        "addr_lev", "addr_tsort", "country_match",
        "name_cos_char", "label",
    ]
    print(feat_df[preview_cols].head().to_string(index=False))

    feat_cols = [c for c in feat_df.columns
                 if c not in ("source1_entity_id", "candidate_entity_id", "label")]
    print("\n    Summary statistics (numeric features):")
    stats = feat_df[feat_cols].describe().T[["mean", "std", "min", "max"]]
    print(stats.to_string())

    # ── 8. Save outputs ────────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, "features_sample_train.csv")
    feat_df.to_csv(out_path, index=False)
    print(f"\n    ✓ Feature matrix saved → {os.path.abspath(out_path)}")

    section("PERSON 2 PIPELINE COMPLETE ✓")
    print("\n  Outputs:")
    print(f"    • {os.path.abspath(out_path)}")
    print()
    print("  Hand-off to Person 3:")
    print("    from src.feature_engineering import preprocess_sources, TFIDFBundle, compute_features")
    print("    feat_df = compute_features(candidate_pairs_df, all_records, tfidf=tfidf_bundle)")
    print()


if __name__ == "__main__":
    main()
