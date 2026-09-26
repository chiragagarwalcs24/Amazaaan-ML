"""
package_submission.py
---------------------
Person 4 – Submission Packaging Script

Creates the final submission zip package with the correct structure:

    AmazaanML_Team_submission.zip
    ├── output/
    │   ├── matching_results.tsv
    │   └── candidate_pairs.tsv
    ├── code/
    │   └── business_entity_resolution/
    │       ├── src/
    │       ├── utils/
    │       ├── README.md
    │       └── requirements.txt
    └── Documentation_template.md

Usage:
    python package_submission.py
"""

import os
import sys
import zipfile
import shutil

TEAM_NAME   = "AmazaanML_Team"
OUTPUT_DIR  = "output"
ZIP_NAME    = f"{TEAM_NAME}_submission.zip"

# ── Files that MUST exist before we package ──────────────────────────────────
REQUIRED_OUTPUT_FILES = [
    os.path.join(OUTPUT_DIR, "matching_results.tsv"),
    os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"),
]

SOURCE_FILES = [
    # (local path, path inside zip)
    ("output/matching_results.tsv",   "output/matching_results.tsv"),
    ("output/candidate_pairs.tsv",    "output/candidate_pairs.tsv"),
    ("Documentation_template.md",     "Documentation_template.md"),
    ("README.md",                     "code/business_entity_resolution/README.md"),
    ("requirements.txt",              "code/business_entity_resolution/requirements.txt"),
]

SRC_DIR   = "src"
UTILS_DIR = "utils"
CODE_BASE = "code/business_entity_resolution"


def check_prerequisites():
    """Verify required output files exist."""
    missing = [f for f in REQUIRED_OUTPUT_FILES if not os.path.isfile(f)]
    if missing:
        print("ERROR: Required output files missing:")
        for f in missing:
            print(f"  ✗  {f}")
        print()
        print("Run the inference pipeline first:")
        print("  python src/pipeline.py --test-only")
        sys.exit(1)
    print("✓  Output files found.")


def check_validation():
    """Run the submission validator before packaging."""
    test_dir = os.path.join(
        "dataset", "6ab10eb3b23ba_student_resource",
        "student_resource", "dataset", "test"
    )
    if not os.path.isdir(test_dir):
        print(f"WARNING: test dir not found at {test_dir} — skipping validation.")
        return

    print("Running submission validator...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "utils/validate_submission.py",
         "--matching", "output/matching_results.tsv",
         "--candidate", "output/candidate_pairs.tsv",
         "--test-dir", test_dir],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR: Validation failed. Fix the issues above before packaging.")
        print(result.stderr)
        sys.exit(1)
    print("✓  Validation passed.")


def build_zip():
    """Create the submission zip archive."""
    if os.path.exists(ZIP_NAME):
        os.remove(ZIP_NAME)
        print(f"  Removed old {ZIP_NAME}")

    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
        # Individual files
        for local_path, zip_path in SOURCE_FILES:
            if os.path.isfile(local_path):
                zf.write(local_path, zip_path)
                print(f"  + {zip_path}")
            else:
                print(f"  WARNING: {local_path} not found, skipping.")

        # src/ directory → code/business_entity_resolution/src/
        for fname in os.listdir(SRC_DIR):
            if fname.endswith(".py") or fname == "__init__.py":
                local = os.path.join(SRC_DIR, fname)
                zname = f"{CODE_BASE}/src/{fname}"
                zf.write(local, zname)
                print(f"  + {zname}")

        # utils/ directory → code/business_entity_resolution/utils/
        for fname in os.listdir(UTILS_DIR):
            if fname.endswith(".py"):
                local = os.path.join(UTILS_DIR, fname)
                zname = f"{CODE_BASE}/utils/{fname}"
                zf.write(local, zname)
                print(f"  + {zname}")

    size_mb = os.path.getsize(ZIP_NAME) / (1024 * 1024)
    print(f"\n✓  Created: {ZIP_NAME}  ({size_mb:.1f} MB)")


def print_zip_contents():
    """Print the full zip tree for verification."""
    print(f"\nContents of {ZIP_NAME}:")
    with zipfile.ZipFile(ZIP_NAME, "r") as zf:
        for info in zf.infolist():
            size_kb = info.file_size / 1024
            print(f"  {info.filename:<70}  ({size_kb:,.0f} KB)")


def main():
    print("=" * 60)
    print(f"  {TEAM_NAME} — Submission Packager")
    print("=" * 60)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("\n[1] Checking prerequisites...")
    check_prerequisites()

    print("\n[2] Validating submission format...")
    check_validation()

    print("\n[3] Building zip archive...")
    build_zip()

    print("\n[4] Verifying zip contents...")
    print_zip_contents()

    print("\n" + "=" * 60)
    print(f"  DONE — upload {ZIP_NAME} to Unstop portal")
    print("=" * 60)


if __name__ == "__main__":
    main()
