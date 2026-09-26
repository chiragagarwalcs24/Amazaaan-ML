"""
eda.py
------
Person 2 – Exploratory Data Analysis

Analyses the training data to understand:
- Record counts and null rates per source
- Country distribution
- Name / address length distributions
- Most common noise patterns in names (leading symbols, ALL CAPS, etc.)
- Ground truth match distribution (how many S2/S3 per S1 entity)
- Blank address rates per country

Run from project root:
    python src/eda.py
"""

import os
import sys
import re

sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from src.data_loader import load_source, load_ground_truth

SEP = "=" * 72

def section(t):
    print(f"\n{SEP}\n  {t}\n{SEP}", flush=True)


def main():
    section("PERSON 2 · EXPLORATORY DATA ANALYSIS")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[Loading training sources …]", flush=True)
    s1 = load_source("train", 1)
    s2 = load_source("train", 2)
    s3 = load_source("train", 3)
    gt = load_ground_truth()

    # ── 1. Basic counts & nulls ────────────────────────────────────────────────
    section("1 · Record Counts & Missing Values")
    for name, df in [("Source 1", s1), ("Source 2", s2), ("Source 3", s3)]:
        n = len(df)
        nn = (df["business_name"] == "").sum()
        na = (df["business_address"] == "").sum()
        print(f"  {name} : {n:>9,} rows | "
              f"blank name={nn:>6,} ({nn/n*100:.1f}%) | "
              f"blank addr={na:>7,} ({na/n*100:.1f}%)")

    # ── 2. Country distribution ────────────────────────────────────────────────
    section("2 · Country Distribution (Train)")
    for name, df in [("Source 1", s1), ("Source 2", s2), ("Source 3", s3)]:
        print(f"\n  {name}:")
        vc = df["country"].value_counts()
        for country, cnt in vc.items():
            print(f"    {country:<12} {cnt:>9,}  ({cnt/len(df)*100:.1f}%)")

    # ── 3. Name length distribution ────────────────────────────────────────────
    section("3 · Business Name Length (chars) Distribution")
    for name, df in [("Source 1", s1), ("Source 2", s2), ("Source 3", s3)]:
        lens = df["business_name"].str.len()
        p = np.percentile(lens, [25, 50, 75, 95, 99])
        print(f"  {name} : min={lens.min()} "
              f"p25={p[0]:.0f} median={p[1]:.0f} "
              f"p75={p[2]:.0f} p95={p[3]:.0f} p99={p[4]:.0f} "
              f"max={lens.max()}")

    # ── 4. Address length distribution ────────────────────────────────────────
    section("4 · Business Address Length (chars) Distribution")
    for name, df in [("Source 1", s1), ("Source 2", s2), ("Source 3", s3)]:
        non_empty = df.loc[df["business_address"] != "", "business_address"]
        lens = non_empty.str.len()
        if len(lens) == 0:
            print(f"  {name} : all addresses empty")
            continue
        p = np.percentile(lens, [25, 50, 75, 95])
        print(f"  {name} : non-empty={len(lens):,} "
              f"p25={p[0]:.0f} median={p[1]:.0f} "
              f"p75={p[2]:.0f} p95={p[3]:.0f} max={lens.max()}")

    # ── 5. Noise patterns in names ─────────────────────────────────────────────
    section("5 · Name Noise Patterns (Source 2 + 3 samples of 200k each)")
    patterns = {
        "Leading symbol/dash (e.g. '-- name')": r"^\s*[^a-zA-Z0-9\u0900-\u097F\u0C00-\u0C7F]",
        "ALL CAPS name":                         lambda s: s == s.upper() and s.strip() != "",
        "Contains '&'":                          r"&",
        "Contains digit in name":                r"\d",
        "Ends with legal suffix (Inc/Ltd/LLC)":  r"\b(inc|ltd|llc|corp|pvt|llp|limited|sarl|sas)\.?\s*$",
        "Has trailing dot/comma":                r"[.,]\s*$",
        "Non-Latin characters":                  lambda s: bool(re.search(r"[^\x00-\x7F]", s)),
    }

    for src_name, df in [("Source 2", s2.head(200_000)),
                          ("Source 3", s3.head(200_000))]:
        print(f"\n  {src_name} (first 200k):")
        names = df["business_name"]
        n = len(names)
        for label, pat in patterns.items():
            if callable(pat):
                count = names.apply(pat).sum()
            else:
                count = names.str.contains(pat, regex=True, na=False, flags=re.IGNORECASE).sum()
            print(f"    {label:<48} {count:>7,}  ({count/n*100:.1f}%)")

    # ── 6. Ground truth match distribution ────────────────────────────────────
    section("6 · Ground Truth — Matches per Source 1 Entity")
    gt["n_matches"] = gt["matched_entity_ids"].apply(
        lambda x: len([m for m in x.split(",") if m.strip()]) if x.strip() else 0
    )
    vc = gt["n_matches"].value_counts().sort_index()
    total = len(gt)
    print(f"\n  Total S1 entities : {total:,}")
    for k, v in vc.items():
        bar = "█" * min(int(v / total * 60), 60)
        print(f"  matches={k:>3}  {v:>9,}  ({v/total*100:5.1f}%)  {bar}")

    # ── 7. Source prefix breakdown in GT ──────────────────────────────────────
    section("7 · Source Breakdown in Ground Truth Matches")
    s2_count = s3_count = 0
    for ids_str in gt["matched_entity_ids"]:
        if ids_str.strip():
            for m in ids_str.split(","):
                m = m.strip()
                if m.startswith("S2-"):
                    s2_count += 1
                elif m.startswith("S3-"):
                    s3_count += 1
    print(f"\n  Total S2 match references : {s2_count:,}")
    print(f"  Total S3 match references : {s3_count:,}")
    print(f"  S2 / S3 ratio             : {s2_count / max(s3_count,1):.2f}")

    # ── 8. Blank address by country ────────────────────────────────────────────
    section("8 · Blank Address Rate by Country")
    for src_name, df in [("Source 2", s2), ("Source 3", s3)]:
        print(f"\n  {src_name}:")
        g = df.groupby("country")["business_address"].apply(
            lambda x: (x == "").sum() / len(x) * 100
        ).round(1)
        for country, pct in g.items():
            print(f"    {country:<12} blank_addr={pct:.1f}%")

    section("EDA COMPLETE")
    print("  Use these insights to refine normalization rules and feature weights.\n")


if __name__ == "__main__":
    main()
