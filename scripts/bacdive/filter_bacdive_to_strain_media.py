#!/usr/bin/env python3
"""Collapse the BacDive growth table to one row per (strain, medium).

Pipeline:
  1. Sort by (strain_name, media_name, medium_growth_class) with
     medium_growth_class descending, so growth (1) sorts before no-growth (0).
  2. Drop duplicates on (strain_name, media_name), keeping the first row —
     i.e. the positive label wins whenever a pair carries both 1 and 0.
  3. Drop the temperature-aware growth columns and rename
     medium_growth_class -> growth_class.
  4. Report stats along the way.

Usage:
    python scripts/bacdive/filter_bacdive_to_strain_media.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "bacdive" / "bd_temp.parquet"
OUTPUT = ROOT / "data" / "processed" / "bacdive" / "bd_med.parquet"

KEYS = ["strain_name", "media_name"]
LABEL = "medium_growth_class"
OUT_LABEL = "growth_class"

# temperature-aware growth columns, dropped so LABEL can take the growth_class
# name -- the surviving label is medium-only
DROP_COLS = [
    "growth_class",
    "growth_source",
    "growth_override",
    "temp_growth_class",
    "temp_join",
]


def describe(df: pd.DataFrame, label: str) -> None:
    """Print shape, class balance and key cardinalities for one table."""
    counts = df[LABEL].value_counts()
    n_pos = int(counts.get(1, 0))
    n_neg = int(counts.get(0, 0))
    pct = 100 * n_pos / len(df) if len(df) else 0.0
    print(f"{label}: {len(df):,} rows x {df.shape[1]} cols")
    print(f"  {LABEL}=1 : {n_pos:,} ({pct:.1f}%)")
    print(f"  {LABEL}=0 : {n_neg:,}")
    print(f"  strains   : {df.strain_name.nunique():,}")
    print(f"  media     : {df.media_name.nunique():,}")
    print(f"  unique {tuple(KEYS)}: {len(df.drop_duplicates(KEYS)):,}")

    other = sorted(set(df[LABEL].unique()) - {0, 1})
    if other:
        print(f"  Warning: unexpected {LABEL} values — {other}")


def report_duplicates(df: pd.DataFrame) -> None:
    """Break down duplicate (strain, medium) pairs into agreeing vs conflicting."""
    agg = df.groupby(KEYS, sort=False)[LABEL].agg(["size", "min", "max"])
    dup = agg[agg["size"] > 1]
    conflict = dup[dup["min"] != dup["max"]]

    n_extra = int(agg["size"].sum() - len(agg))
    print(f"duplicated pairs      : {len(dup):,} / {len(agg):,}")
    print(f"  rows to be dropped  : {n_extra:,}")
    print(f"  conflicting pairs   : {len(conflict):,} (both 1 and 0)")
    print(f"  agreeing pairs      : {len(dup) - len(conflict):,}")

    if len(conflict):
        # negatives discarded in favour of the positive label
        n_neg_dropped = int(
            df.merge(conflict.reset_index()[KEYS], on=KEYS, how="inner")[LABEL]
            .eq(0)
            .sum()
        )
        print(f"  {LABEL}=0 rows dropped on conflicts: {n_neg_dropped:,}")
        print("\n-- conflicting pairs (first 20) --")
        print(conflict.head(20).reset_index().to_string(index=False))


def dedup_to_strain_media(df: pd.DataFrame) -> pd.DataFrame:
    """Sort positives first, then keep one row per (strain_name, media_name)."""
    ordered = df.sort_values(by=KEYS + [LABEL], ascending=[True, True, False])
    out = ordered.drop_duplicates(subset=KEYS).reset_index(drop=True)
    print(f"dropped duplicates on {KEYS}: {len(df):,} -> {len(out):,} rows")
    return out


def finalize_label(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the temperature-aware growth columns, then LABEL -> OUT_LABEL."""
    present = [c for c in DROP_COLS if c in df.columns]
    missing = [c for c in DROP_COLS if c not in df.columns]
    if missing:
        print(f"  Warning: nothing to drop for: {missing}")

    if "growth_class" in present:
        agree = int((df[LABEL] == df["growth_class"]).sum())
        print(f"  {LABEL} vs growth_class: {agree:,} agree, "
              f"{len(df) - agree:,} differ (growth_class discarded)")

    out = df.drop(columns=present).rename(columns={LABEL: OUT_LABEL})
    print(f"dropped {present}, renamed {LABEL} -> {OUT_LABEL}")
    print(f"  {len(out):,} rows x {out.shape[1]} cols")
    return out


def main() -> None:
    print(f"Input : {INPUT}")
    df = pd.read_parquet(INPUT)

    missing = [c for c in KEYS + [LABEL] if c not in df.columns]
    if missing:
        raise SystemExit(f"missing required columns: {missing}")

    print("\n=== before ===")
    describe(df, "input")

    print("\n=== duplicates ===")
    report_duplicates(df)

    print("\n=== dedup ===")
    out = dedup_to_strain_media(df)

    print("\n=== after ===")
    describe(out, "output")

    n_pos_in = int((df[LABEL] == 1).sum())
    n_neg_in = int((df[LABEL] == 0).sum())
    n_pos_out = int((out[LABEL] == 1).sum())
    n_neg_out = int((out[LABEL] == 0).sum())
    print("\n=== delta ===")
    print(f"  rows      : {len(df):,} -> {len(out):,} ({len(out) - len(df):+,})")
    print(f"  {LABEL}=1 : {n_pos_in:,} -> {n_pos_out:,} ({n_pos_out - n_pos_in:+,})")
    print(f"  {LABEL}=0 : {n_neg_in:,} -> {n_neg_out:,} ({n_neg_out - n_neg_in:+,})")
    print(f"  media per strain (mean): {len(out) / out.strain_name.nunique():.2f}")

    print("\n=== columns ===")
    out = finalize_label(out)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT, index=False)
    print(f"\nOutput: {OUTPUT}")


if __name__ == "__main__":
    main()
