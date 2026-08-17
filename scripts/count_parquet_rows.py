"""
Analyze and save unique combinations of columns in a Parquet dataset.

Usage:
    python scripts/count_parquet_rows.py data/external/bacdive_growth.parquet --save

Options:
    --save: Save the unique combinations of columns to a Parquet file.
       - Specified by SAVE_COMBO1_NAME and SAVE_COMBO2_NAME
    --out-dir: Save the unique combinations of columns to a directory.
       - Specified by OUT_DIR
"""

import argparse
import os
import pandas as pd

OUT_DIR = "data/external"
SAVE_COMBO1_NAME = "bacdive_growth_150k.parquet"
SAVE_COMBO2_NAME = "bacdive_growth_98k.parquet"

def analyze_parquet(file_path: str, save: bool = False, out_dir: str | None = None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read the parquet file
    print(f"Reading: {file_path}...")
    df = pd.read_parquet(file_path)

    # 1. Total number of rows
    total_rows = len(df)
    print(f"\n--- Row Counts ---")
    print(f"Total rows: {total_rows:,}")

    # Define column combinations
    combo_1 = [
        "strain_name",
        "media_name",
        "temperature",
        "oxygen_tolerance",
        "growth_class",
    ]
    combo_2 = ["strain_name",
               "media_name",
               "growth_class"]

    combos = [
        ("Condition 1 (strain, media, temp, O2, growth_class)", combo_1, SAVE_COMBO1_NAME),
        ("Condition 2 (strain, media, growth_class)", combo_2, SAVE_COMBO2_NAME),
    ]

    dest_dir = out_dir or OUT_DIR

    print("\n--- Unique Combination Counts ---")
    for label, cols, save_name in combos:
        # Check if columns exist in DataFrame
        missing_cols = [c for c in cols if c not in df.columns]
        if missing_cols:
            print(f"{label}: Error - Missing columns {missing_cols}")
            continue

        unique_df = df.drop_duplicates(subset=cols)
        unique_count = len(unique_df)
        print(f"{label}: {unique_count:,} unique rows")
        print(f"  Growth class 1 count: {len(df[df['growth_class'] == 1].drop_duplicates(subset=cols))}")
        print(f"  Growth class 0 count: {len(df[df['growth_class'] == 0].drop_duplicates(subset=cols))}")

        if save:
            os.makedirs(dest_dir, exist_ok=True)
            out_path = os.path.join(dest_dir, save_name)
            unique_df.to_parquet(out_path, index=False)
            print(f"  Saved: {out_path}")

        print("--------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count total and unique rows in a Parquet dataset."
    )
    parser.add_argument(
        "parquet_path",
        type=str,
        help="Path to the input .parquet file (or folder)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help=f"Write unique-combo parquet files ({SAVE_COMBO1_NAME}, {SAVE_COMBO2_NAME}).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=OUT_DIR,
        help=f"Directory for saved files (default: {OUT_DIR}).",
    )

    args = parser.parse_args()
    analyze_parquet(args.parquet_path, save=args.save, out_dir=args.out_dir)