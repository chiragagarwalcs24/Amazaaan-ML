from __future__ import annotations
import os, sys, json, time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib

from src.data_loader import load_source, load_ground_truth
from src.feature_engineering import preprocess_sources, TFIDFBundle, compute_features, FEATURE_COLS
from src.blocking import (build_candidate_pairs, evaluate_blocking_recall,
                          _build_index, _build_country_index, _candidates_for_one)
from src.evaluate import f05_score

try:
    import lightgbm as lgb
    USE_LGB = True
except ImportError:
    import xgboost as xgb
    USE_LGB = False

S1_TRAIN_SAMPLE = 20_000
MAX_CANDIDATES  = 50
OUTPUT_DIR      = "output"
MODEL_DIR       = "output/model"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)
SEP = "=" * 72

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}", flush=True)


def _fast_needed_ids(gt_df: pd.DataFrame, s1_ids_set: set) -> set:
    """Extract all S2/S3 IDs needed from GT for given S1 IDs — fast, no iterrows."""
    gt_sub = gt_df[
        gt_df["source1_entity_id"].isin(s1_ids_set) &
        gt_df["matched_entity_ids"].fillna("").str.strip().ne("")
    ]["matched_entity_ids"]
    # Split all comma-separated IDs at once
    all_ids = gt_sub.str.split(",").explode().str.strip()
    return set(all_ids[all_ids != ""])


# ── TRAIN PHASE ──────────────────────────────────────────────────────────────
def run_train():
    section("TRAIN PHASE: Building and saving model")

    print("\n[1] Loading S1 + Ground Truth...", flush=True)
    s1_full = load_source("train", 1)
    print(f"   S1 loaded: {len(s1_full)} rows", flush=True)
    gt_full = load_ground_truth()
    print(f"   GT loaded: {len(gt_full)} rows", flush=True)

    # Sample S1
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(s1_full), size=min(S1_TRAIN_SAMPLE, len(s1_full)), replace=False)
    s1 = s1_full.iloc[sample_idx].reset_index(drop=True)
    del s1_full
    print(f"   S1 sample : {len(s1)} rows", flush=True)

    # Fast: find which S2/S3 IDs are needed (no iterrows!)
    s1_ids_set = set(s1["entity_id"])
    print("[2] Finding needed S2/S3 IDs from GT (fast)...", flush=True)
    needed_ids = _fast_needed_ids(gt_full, s1_ids_set)
    print(f"   GT-referenced IDs needed: {len(needed_ids)}", flush=True)

    print("[3] Loading S2 and S3...", flush=True)
    s2_full = load_source("train", 2)
    print(f"   S2 loaded: {len(s2_full)} rows", flush=True)
    s3_full = load_source("train", 3)
    print(f"   S3 loaded: {len(s3_full)} rows", flush=True)

    s2_needed = s2_full[s2_full["entity_id"].isin(needed_ids)]
    s3_needed = s3_full[s3_full["entity_id"].isin(needed_ids)]
    s2_extra_idx = rng.choice(len(s2_full), size=min(15000, len(s2_full)), replace=False)
    s3_extra_idx = rng.choice(len(s3_full), size=min(15000, len(s3_full)), replace=False)
    s2 = pd.concat([s2_needed, s2_full.iloc[s2_extra_idx]]).drop_duplicates("entity_id").reset_index(drop=True)
    s3 = pd.concat([s3_needed, s3_full.iloc[s3_extra_idx]]).drop_duplicates("entity_id").reset_index(drop=True)
    del s2_full, s3_full
    print(f"   S2 pool: {len(s2)}  |  S3 pool: {len(s3)}", flush=True)

    print("[4] Normalizing text...", flush=True)
    sources = preprocess_sources({"s1": s1, "s2": s2, "s3": s3})
    s1_proc = sources["s1"]
    s2_proc = sources["s2"]
    s3_proc = sources["s3"]
    all_records = pd.concat(list(sources.values()), ignore_index=True)
    print(f"   Normalized {len(all_records)} records total", flush=True)

    print("[5] Fitting TF-IDF vectorizers...", flush=True)
    tfidf = TFIDFBundle(max_features=50_000)
    tfidf.fit(all_records["name_norm"].tolist(), all_records["addr_norm"].tolist())
    print(f"   name_char: {len(tfidf.name_char.vocabulary_)} | name_word: {len(tfidf.name_word.vocabulary_)} | addr_char: {len(tfidf.addr_char.vocabulary_)}", flush=True)

    print("[6] Running blocking...", flush=True)
    pairs_df = build_candidate_pairs(
        s1_proc, s2_proc, s3_proc,
        gt_df=gt_full, max_candidates=MAX_CANDIDATES, verbose=True,
    )
    evaluate_blocking_recall(pairs_df, gt_full, s1_ids_set)

    print("[7] Computing features in chunks to save memory...", flush=True)
    chunk_size = 50_000
    feat_chunks = []
    for start in range(0, len(pairs_df), chunk_size):
        end = min(start + chunk_size, len(pairs_df))
        print(f"   Chunk {start:,}/{len(pairs_df):,}...", flush=True)
        chunk_feat = compute_features(pairs_df.iloc[start:end].copy(), all_records, tfidf=tfidf)
        feat_chunks.append(chunk_feat)
    feat_df = pd.concat(feat_chunks, ignore_index=True)
    print(f"   Feature matrix: {feat_df.shape}", flush=True)

    if "label" not in feat_df.columns:
        print("ERROR: No label column.")
        return

    n_pos = (feat_df["label"] == 1).sum()
    n_neg = (feat_df["label"] == 0).sum()
    print(f"   Positive: {n_pos}  |  Negative: {n_neg}", flush=True)

    print("[8] Training model...", flush=True)
    from sklearn.model_selection import train_test_split
    s1_unique = feat_df["source1_entity_id"].unique()
    train_s1, val_s1 = train_test_split(s1_unique, test_size=0.2, random_state=42)
    train_df = feat_df[feat_df["source1_entity_id"].isin(train_s1)]
    val_df   = feat_df[feat_df["source1_entity_id"].isin(val_s1)]
    X_train, y_train = train_df[FEATURE_COLS], train_df["label"]
    X_val,   y_val   = val_df[FEATURE_COLS],   val_df["label"]

    pos_cnt = (y_train == 1).sum()
    neg_cnt = (y_train == 0).sum()
    scale_w = neg_cnt / pos_cnt if pos_cnt > 0 else 1.0
    print(f"   scale_pos_weight = {scale_w:.1f}", flush=True)

    if USE_LGB:
        model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            scale_pos_weight=scale_w, random_state=42, n_jobs=-1, verbosity=-1,
        )
    else:
        model = xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            scale_pos_weight=scale_w, eval_metric="logloss", random_state=42, n_jobs=-1,
        )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    print("   Training complete!", flush=True)

    print("[9] Finding best threshold...", flush=True)
    val_df = val_df.copy()
    val_df["prob"] = model.predict_proba(X_val)[:, 1]
    best_f05, best_thr, best_p, best_r = 0.0, 0.5, 0.0, 0.0
    for thr in np.arange(0.05, 0.95, 0.025):
        preds = val_df["prob"] >= thr
        tp = (preds & (val_df["label"] == 1)).sum()
        fp = (preds & (val_df["label"] == 0)).sum()
        fn = (~preds & (val_df["label"] == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sc   = f05_score(prec, rec)
        if sc > best_f05:
            best_f05, best_thr, best_p, best_r = sc, round(float(thr), 3), prec, rec

    print(f"   Best Threshold : {best_thr}", flush=True)
    print(f"   F0.5 Score     : {best_f05:.4f}", flush=True)
    print(f"   Precision      : {best_p:.4f}", flush=True)
    print(f"   Recall         : {best_r:.4f}", flush=True)

    print("[10] Saving model, TF-IDF, threshold...", flush=True)
    joblib.dump(model, f"{MODEL_DIR}/model.pkl")
    joblib.dump(tfidf, f"{MODEL_DIR}/tfidf.pkl")
    with open(f"{MODEL_DIR}/threshold.json", "w") as f:
        json.dump({"threshold": best_thr, "f05": best_f05}, f)
    feat_df.to_csv(f"{OUTPUT_DIR}/features_sample_train.csv", index=False)
    print(f"   Saved to {MODEL_DIR}/", flush=True)
    section("TRAIN PHASE COMPLETE")
    return model, tfidf, best_thr


# ── TEST / INFERENCE PHASE ───────────────────────────────────────────────────
def run_test(chunk_size: int = 10_000):
    section("TEST PHASE: Generating submission files")

    print("[1] Loading saved model and TF-IDF...", flush=True)
    if not os.path.exists(f"{MODEL_DIR}/model.pkl"):
        print("ERROR: No saved model. Run train phase first.")
        return
    model   = joblib.load(f"{MODEL_DIR}/model.pkl")
    tfidf   = joblib.load(f"{MODEL_DIR}/tfidf.pkl")
    with open(f"{MODEL_DIR}/threshold.json") as f:
        meta = json.load(f)
    threshold = meta["threshold"]
    print(f"   Threshold: {threshold}  |  Train F0.5: {meta['f05']:.4f}", flush=True)

    print("[2] Loading test data...", flush=True)
    t1 = load_source("test", 1)
    t2 = load_source("test", 2)
    t3 = load_source("test", 3)
    print(f"   Test S1: {len(t1)}  |  S2: {len(t2)}  |  S3: {len(t3)}", flush=True)

    print("[3] Normalizing test data (takes a few minutes)...", flush=True)
    cache_path = os.path.join(OUTPUT_DIR, "test_normalized.pkl")
    if os.path.exists(cache_path):
        print(f"   Loading cached normalized data from {cache_path}...", flush=True)
        sources = joblib.load(cache_path)
    else:
        sources = preprocess_sources({"s1": t1, "s2": t2, "s3": t3}, n_jobs=-1)
        print(f"   Saving normalized data to {cache_path} for faster restarts...", flush=True)
        joblib.dump(sources, cache_path)
        
    t1_proc = sources["s1"]
    t2_proc = sources["s2"]
    t3_proc = sources["s3"]
    all_records_test = pd.concat(list(sources.values()), ignore_index=True)
    print(f"   Normalized {len(all_records_test)} records", flush=True)

    all_s1_ids = list(t1_proc["entity_id"])
    print(f"[4] Building blocking index on test S2 ({len(t2_proc)}) + S3 ({len(t3_proc)})...", flush=True)
    idx_s2   = _build_index(t2_proc)
    idx_s3   = _build_index(t3_proc)
    ctry_s2  = _build_country_index(t2_proc)
    ctry_s3  = _build_country_index(t3_proc)
    valid_all = set(t2_proc["entity_id"]) | set(t3_proc["entity_id"])
    print(f"   Index built: S2 keys={len(idx_s2)}  S3 keys={len(idx_s3)}", flush=True)

    all_match_results  = {}
    all_candidate_sets = {}
    t1_records = t1_proc.to_dict("records")
    total_s1   = len(t1_records)
    t0 = time.time()

    checkpoint_file = f"{OUTPUT_DIR}/inference_checkpoint.csv"
    processed_chunks = set()
    if os.path.exists(checkpoint_file):
        print(f"   Resuming from checkpoint {checkpoint_file}...", flush=True)
        try:
            chk_df = pd.read_csv(checkpoint_file, dtype=str)
            if "chunk_start" in chk_df.columns:
                processed_chunks = set(chk_df["chunk_start"].astype(int).unique())
            for _, row in chk_df.iterrows():
                s1_id = row["source1_entity_id"]
                c_id = row["candidate_entity_id"]
                if pd.notna(s1_id) and pd.notna(c_id) and str(s1_id).strip() and str(c_id).strip():
                    all_match_results.setdefault(s1_id, set()).add(c_id)
        except Exception as e:
            print(f"   Warning: could not read checkpoint: {e}", flush=True)
    else:
        pd.DataFrame(columns=["chunk_start", "source1_entity_id", "candidate_entity_id"]).to_csv(checkpoint_file, index=False)

    print(f"[5] Running inference in chunks of {chunk_size}...", flush=True)
    for chunk_start in range(0, total_s1, chunk_size):
        if chunk_start in processed_chunks:
            continue
        chunk_end = min(chunk_start + chunk_size, total_s1)
        chunk = t1_records[chunk_start:chunk_end]
        elapsed = time.time() - t0
        pct = chunk_start / total_s1 * 100
        print(f"   Chunk {chunk_start}/{total_s1} ({pct:.0f}%) -- {elapsed:.0f}s elapsed", flush=True)

        cand_rows = []
        for s1_row in chunk:
            s1_id   = s1_row["entity_id"]
            name_n  = s1_row.get("name_norm", "")
            addr_n  = s1_row.get("addr_norm", "")
            country = s1_row.get("country", "")
            cands = _candidates_for_one(
                name_n, addr_n, country,
                idx_s2, idx_s3, ctry_s2, ctry_s3, MAX_CANDIDATES,
            )
            cands = [c for c in cands if c in valid_all]
            all_candidate_sets[s1_id] = set(cands)
            for c_id in cands:
                cand_rows.append({"source1_entity_id": s1_id, "candidate_entity_id": c_id})

        if not cand_rows:
            pd.DataFrame([{"chunk_start": chunk_start, "source1_entity_id": "", "candidate_entity_id": ""}]).to_csv(checkpoint_file, mode='a', header=False, index=False)
            continue

        chunk_pairs = pd.DataFrame(cand_rows).drop_duplicates(
            subset=["source1_entity_id", "candidate_entity_id"]
        ).reset_index(drop=True)

        feat_chunk = compute_features(chunk_pairs, all_records_test, tfidf=tfidf)
        probs = model.predict_proba(feat_chunk[FEATURE_COLS])[:, 1]
        feat_chunk = feat_chunk.copy()
        feat_chunk["prob"] = probs
        matched = feat_chunk[feat_chunk["prob"] >= threshold]

        # Fast: groupby instead of iterrows
        chunk_results = []
        for s1_id, group in matched.groupby("source1_entity_id"):
            cands = group["candidate_entity_id"].tolist()
            all_match_results.setdefault(s1_id, set()).update(cands)
            for c in cands:
                chunk_results.append({"chunk_start": chunk_start, "source1_entity_id": s1_id, "candidate_entity_id": c})
                
        if not chunk_results:
            chunk_results.append({"chunk_start": chunk_start, "source1_entity_id": "", "candidate_entity_id": ""})
            
        pd.DataFrame(chunk_results).to_csv(checkpoint_file, mode='a', header=False, index=False)

    print(f"   Inference done in {time.time()-t0:.0f}s", flush=True)

    print("[6] Writing output files...", flush=True)
    match_rows = []
    for s1_id in all_s1_ids:
        matches = all_match_results.get(s1_id, set())
        match_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(sorted(matches)),
        })
    match_df = pd.DataFrame(match_rows)
    match_path = f"{OUTPUT_DIR}/matching_results.tsv"
    match_df.to_csv(match_path, sep="\t", index=False)
    n_with = (match_df["matched_entity_ids"] != "").sum()
    print(f"   Saved: {match_path}  ({len(match_df)} rows, {n_with} with matches)", flush=True)

    cand_rows_all = []
    for s1_id in all_s1_ids:
        cands = all_candidate_sets.get(s1_id, set())
        cand_rows_all.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(sorted(cands)),
        })
    cand_df = pd.DataFrame(cand_rows_all)
    cand_path = f"{OUTPUT_DIR}/candidate_pairs.tsv"
    cand_df.to_csv(cand_path, sep="\t", index=False)
    print(f"   Saved: {cand_path}", flush=True)

    section("TEST PHASE COMPLETE -- Files ready for submission!")
    print(f"\n  Submit to Unstop: {os.path.abspath(match_path)}", flush=True)


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--test-only",  action="store_true")
    args = parser.parse_args()
    if args.test_only:
        run_test()
    elif args.train_only:
        run_train()
    else:
        run_train()
        run_test()

if __name__ == "__main__":
    main()
