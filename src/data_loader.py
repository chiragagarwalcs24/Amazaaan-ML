"""
data_loader.py
--------------
Data Loading Utilities for Business Entity Resolution.

All files in this challenge are tab-separated (.tsv).
Reading without sep='\\t' silently produces a single-column frame.
"""

import os
import pandas as pd
from typing import Dict


# ─── canonical dataset root (relative to project root) ───────────────────────
DATASET_ROOT = r"C:\Users\Lenovo\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"


def _tsv(path: str) -> pd.DataFrame:
    """Read any .tsv with tab separator, keeping all columns as str."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    return df


def _fill(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Replace NaN with '' in the given columns."""
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    return df


# ─── Public API ───────────────────────────────────────────────────────────────

def load_source(split: str, src: int, base_dir: str = DATASET_ROOT) -> pd.DataFrame:
    """
    Load one source file.

    Parameters
    ----------
    split : 'train' or 'test'
    src   : 1, 2, or 3
    base_dir : override the default dataset root if needed.

    Returns
    -------
    DataFrame with columns: entity_id, business_name, business_address, country
    """
    fname = f"{split}_source{src}.tsv"
    path = os.path.join(base_dir, split, fname)
    df = _tsv(path)
    df = _fill(df, ["business_name", "business_address", "country"])
    return df


def load_ground_truth(base_dir: str = DATASET_ROOT) -> pd.DataFrame:
    """
    Load train_ground_truth.tsv.

    Returns
    -------
    DataFrame with columns: source1_entity_id, matched_entity_ids
    """
    path = os.path.join(base_dir, "train", "train_ground_truth.tsv")
    df = _tsv(path)
    df = _fill(df, ["matched_entity_ids"])
    return df


def load_all_sources(split: str, base_dir: str = DATASET_ROOT) -> Dict[str, pd.DataFrame]:
    """
    Load all three source files for a split.

    Returns
    -------
    {'source1': df1, 'source2': df2, 'source3': df3}
    """
    return {f"source{i}": load_source(split, i, base_dir) for i in range(1, 4)}
