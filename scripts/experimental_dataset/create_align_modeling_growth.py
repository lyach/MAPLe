#!/usr/bin/env python3
"""Filter tidy growth curves and attach strain / media metadata.

Pipeline:
  1. Drop water controls and other/empty strains.
  2. Drop rows whose (exp_id, date, strain, media) is marked
     confidence == 'low' in the curation CSV.
  3. Compare growth_class (1/0) to curation growth (yes/no); overwrite
     mismatches with the curation label.
  4. Keep identifier columns + growth_class.
  5. Left-merge strain metadata and media features.

Usage:
    python scripts/experimental_dataset/filter_noisy_curves.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "aero_anaero_tidy_curves.csv"
FLAGS = ROOT / "data" / "metadata" / "align_dataset" / "align_growth_curve_curation.csv"
STRAIN_META = ROOT / "data" / "metadata" / "align_dataset" / "align_strains_metadata.csv"
MEDIA_META = ROOT / "data" / "metadata" / "modeling" / "media_features.csv"
OUTPUT = ROOT / "data" / "processed" / "aero_anaero_tidy_curves_filtered.csv"

# curation key → tidy CSV column
FIELD_MAP = {
    "exp_id": "exp_id",
    "date": "date",
    "strain": "strain_name",
    "media": "media_name",
}

GROWTH_MAP = {1: "yes", 0: "no"}
FLAGS_TO_CLASS = {"yes": 1, "no": 0}

KEEP_COLS = [
    "exp_id",
    "date",
    "well_id",
    "row",
    "col",
    "well_index",
    "strain_name",
    "media_name",
    "atmosphere",
    "temperature",
    "growth_class",
]


def drop_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Remove water controls and other/empty strains (first cleanup steps)."""
    if "media_name" in df.columns:
        df = df.loc[df["media_name"] != "water"]
        print(f"removed water controls, shape: {df.shape[0]} rows x {df.shape[1]} cols")

    if "strain_name" in df.columns:
        df = df.loc[~df["strain_name"].str.contains("other")]
        print(f"removed `other` strains, shape: {df.shape[0]} rows x {df.shape[1]} cols")

        df = df.loc[df["strain_name"] != "empty"]
        print(f"removed `empty` strains, shape: {df.shape[0]} rows x {df.shape[1]} cols")

    return df


def load_low_flags(flags: pd.DataFrame) -> pd.DataFrame:
    """Unique (exp_id, date, strain, media) rows with confidence == low."""
    low = flags[flags.confidence.astype(str).str.strip().str.lower() == "low"]
    return low[list(FIELD_MAP)].drop_duplicates().reset_index(drop=True)


def build_drop_mask(df: pd.DataFrame, flags: pd.DataFrame) -> pd.Series:
    """True for tidy rows that match any flag on all four fields."""
    mask = pd.Series(False, index=df.index)
    for i, flag in flags.iterrows():
        flag_mask = pd.Series(True, index=df.index)
        for key, col in FIELD_MAP.items():
            value = int(flag[key]) if col == "date" else flag[key]
            flag_mask &= df[col] == value
        n = int(flag_mask.sum())
        label = (
            f"exp_id={flag.exp_id!r}, date={flag.date!r}, "
            f"strain={flag.strain!r}, media={flag.media!r}"
        )
        if n == 0:
            print(f"  Warning: flag {i + 1} matched no rows ({label})")
        mask |= flag_mask
    return mask


def _flags_growth_table(flags: pd.DataFrame) -> pd.DataFrame:
    """FLAGS growth labels keyed by tidy join columns."""
    right = flags[list(FIELD_MAP) + ["growth"]].copy()
    right["date"] = right.date.astype(int)
    right["growth"] = right.growth.astype(str).str.strip().str.lower()
    right = right.rename(columns=FIELD_MAP).drop_duplicates()
    right["growth_from_flags"] = right.growth.map(FLAGS_TO_CLASS)
    return right


def compare_growth(df: pd.DataFrame, flags: pd.DataFrame) -> None:
    """Compare filtered growth_class (1/0) to FLAGS growth (yes/no) on join keys."""
    left = (
        df[list(FIELD_MAP.values()) + ["growth_class"]]
        .assign(date=lambda d: d.date.astype(int))
        .drop_duplicates()
    )
    left["growth_from_input"] = left.growth_class.map(GROWTH_MAP)

    right = _flags_growth_table(flags)
    merged = left.merge(
        right[list(FIELD_MAP.values()) + ["growth"]],
        on=list(FIELD_MAP.values()),
        how="inner",
    ).rename(columns={"growth": "growth_from_flags"})

    agree = merged.growth_from_input == merged.growth_from_flags
    n_match = int(agree.sum())
    n_mismatch = int((~agree).sum())

    print("\n=== growth_class (filtered) vs growth (FLAGS) ===")
    print(f"  joined rows (unique keys) : {len(merged):,}")
    print(f"  matches                   : {n_match:,}")
    print(f"  mismatches                : {n_mismatch:,}")

    print("\n-- crosstab (filtered × flags) --")
    print(
        pd.crosstab(
            merged.growth_from_input,
            merged.growth_from_flags,
            rownames=["filtered"],
            colnames=["flags"],
        ).to_string()
    )

    if n_mismatch:
        cols = list(FIELD_MAP.values()) + ["growth_from_input", "growth_from_flags"]
        print("\n-- mismatches --")
        print(merged.loc[~agree, cols].to_string(index=False))


def replace_growth(df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Overwrite growth_class with FLAGS growth only where labels disagree."""
    right = _flags_growth_table(flags)
    keys = list(FIELD_MAP.values())
    out = df.copy()
    out["date"] = out.date.astype(int)
    out = out.merge(
        right[keys + ["growth_from_flags"]],
        on=keys,
        how="left",
    )
    mismatch = (
        out.growth_from_flags.notna()
        & (out.growth_class != out.growth_from_flags)
    )
    n = int(mismatch.sum())
    out.loc[mismatch, "growth_class"] = out.loc[mismatch, "growth_from_flags"].astype(int)
    out = out.drop(columns=["growth_from_flags"])
    print(f"replaced growth_class on {n:,} mismatch rows / {len(out):,} total")
    return out


def keep_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep identifier columns plus growth_class."""
    cols = [c for c in KEEP_COLS if c in df.columns]
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        print(f"  Warning: keep_columns missing: {missing}")
    out = df[cols].copy()
    print(f"kept {len(cols)} columns, shape: {out.shape[0]} rows x {out.shape[1]} cols")
    return out


def merge_strain_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Left-merge align strain metadata on lowercase strain / strain_name."""
    meta = pd.read_csv(STRAIN_META)
    # skip last 4 cols: sequence_hash, date_retrieved, resolution_method, requires_verification
    meta = meta.iloc[:, :-4]
    meta_cols = [c for c in meta.columns if c != "strain"]

    left = df.copy()
    left["_join"] = left.strain_name.astype(str).str.strip().str.lower()
    right = meta.copy()
    right["_join"] = right.strain.astype(str).str.strip().str.lower()

    out = left.merge(right[["_join"] + meta_cols], on="_join", how="left")
    out = out.drop(columns=["_join"])

    n_unmatched = int(out[meta_cols[0]].isna().sum()) if meta_cols else 0
    if n_unmatched:
        unmatched = out.loc[out[meta_cols[0]].isna(), "strain_name"].unique().tolist()
        print(f"  Warning: {n_unmatched} rows unmatched strain metadata — {unmatched}")
    print(f"merged strain metadata, shape: {out.shape[0]} rows x {out.shape[1]} cols")
    return out


def merge_media_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Left-merge media features on media_name."""
    meta = pd.read_csv(MEDIA_META)
    left = df.copy()
    left["_join"] = left.media_name.astype(str).str.strip().str.lower()
    right = meta.copy()
    right["_join"] = right.media_name.astype(str).str.strip().str.lower()
    right = right.drop(columns=["media_name"])

    out = left.merge(right, on="_join", how="left")
    out = out.drop(columns=["_join"])

    feat_cols = [c for c in meta.columns if c != "media_name"]
    n_unmatched = int(out[feat_cols[0]].isna().sum()) if feat_cols else 0
    if n_unmatched:
        unmatched = out.loc[out[feat_cols[0]].isna(), "media_name"].unique().tolist()
        print(f"  Warning: {n_unmatched} rows unmatched media metadata — {unmatched}")
    print(f"merged media metadata, shape: {out.shape[0]} rows x {out.shape[1]} cols")
    return out


def main() -> None:
    flags_all = pd.read_csv(FLAGS)
    flags = load_low_flags(flags_all)
    df = pd.read_csv(INPUT)

    print(f"Input : {INPUT}")
    print(f"Flags : {FLAGS}")
    print(f"Rows before filtering: {len(df):,}")

    print("\n=== drop controls ===")
    df = drop_controls(df)

    print(f"\nApplying {len(flags)} low-confidence flag(s):")
    drop_mask = build_drop_mask(df, flags)
    out = df[~drop_mask].copy()

    print(f"\nRows after filtering : {len(out):,}")
    print(f"Rows removed         : {len(df) - len(out):,}")

    compare_growth(out, flags_all)

    print("\n=== replace growth / keep cols / merge metadata ===")
    out = replace_growth(out, flags_all)
    out = keep_columns(out)
    out = merge_strain_metadata(out)
    out = merge_media_metadata(out)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)
    print(f"\nOutput: {OUTPUT}")


if __name__ == "__main__":
    main()
