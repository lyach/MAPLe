#!/usr/bin/env python3
"""Split BacDive rows into culture-medium vs culture-temp provenance.

``growth_source`` alone is biased as a count of "datapoints from culture temp":

* A matched culture-temp *flag* always overwrites ``growth_class``, even when
  it agrees with the medium (almost all positives). Those rows would exist
  from ``culture medium`` anyway; only the label attribution changed.
* ``growth_override==1`` is the unbiased count: the temp flag was applied
  *and* differs from the medium (typically CIP no-growth temperatures).
* Matching is skipped when the isolate has only one distinct temperature
  (``temp_join=skipped_single_temp``). Those rows stay ``culture medium``
  even if a temp entry shares the medium's ``@ref``.
* Range expansion and same-label dedup (which prefers ``culture temp``)
  inflate ``growth_source==culture temp`` further.

Requires a table built after the provenance columns were added
(``growth_source``, ``growth_override``, ``medium_growth_class``,
``temp_growth_class``, ``temp_join``).

USAGE:
    python scripts/utils/diagnose_growth_source.py
    python scripts/utils/diagnose_growth_source.py data/external/bacdive_growth_full.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = [
    ROOT / "data/external/bacdive_growth_full.parquet",
    ROOT / "data/external/bacdive_growth_full_texpand.parquet",
]

REQUIRED = [
    "growth_class",
    "growth_source",
    "growth_override",
    "medium_growth_class",
    "temp_join",
]


def _path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    for p in DEFAULTS:
        if p.exists():
            return p
    raise SystemExit(
        "no parquet given and none of the defaults exist:\n  "
        + "\n  ".join(str(p) for p in DEFAULTS)
    )


def main() -> None:
    path = _path()
    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{path} is missing {missing}. Rebuild with the current extractor."
        )

    n = len(df)
    src = df["growth_source"].fillna("").astype(str)
    ov = pd.to_numeric(df["growth_override"], errors="coerce").fillna(0).astype(int)
    y = pd.to_numeric(df["growth_class"], errors="coerce")
    med = pd.to_numeric(df["medium_growth_class"], errors="coerce")
    join = df["temp_join"].fillna("").astype(str)

    print(f"file: {path}")
    print(f"rows: {n:,}\n")

    print("=== do not use growth_source alone ===")
    print(
        "  'culture temp' = a temp flag was *applied*, including when it "
        "agrees with the medium."
    )
    print("  use growth_override==1 for labels culture temp actually changed.\n")

    print("=== growth_source x growth_class ===")
    print(
        pd.crosstab(src, y, margins=True, rownames=["growth_source"],
                    colnames=["growth_class"]).to_string()
    )

    print("\n=== culture-temp rows: agreed vs override ===")
    is_temp = src.eq("culture temp")
    n_temp = int(is_temp.sum())
    n_ov = int((is_temp & ov.eq(1)).sum())
    n_agreed = n_temp - n_ov
    print(f"  growth_source == culture temp : {n_temp:,}  ({n_temp / n:.1%} of rows)")
    print(f"    agreed with medium          : {n_agreed:,}  "
          f"(would exist from culture medium anyway)")
    print(f"    override (temp ≠ medium)    : {n_ov:,}  "
          f"(this is the culture-temp *label* count)")

    print("\n=== growth_override x growth_class (among overrides) ===")
    if n_ov:
        print(
            pd.crosstab(ov.eq(1), y, rownames=["override"],
                        colnames=["growth_class"]).to_string()
        )
        print("\n  override direction (medium_growth_class -> growth_class):")
        sub = df.loc[ov.eq(1), ["medium_growth_class", "growth_class"]]
        print(
            pd.crosstab(sub["medium_growth_class"], sub["growth_class"],
                        rownames=["medium"], colnames=["emitted"]).to_string()
        )
    else:
        print("  none")

    print("\n=== temp_join (was @ref matching applied?) ===")
    print(
        pd.crosstab(join, src, margins=True, rownames=["temp_join"],
                    colnames=["growth_source"]).to_string()
    )
    n_skip = int(join.eq("skipped_single_temp").sum())
    n_skip_had_temp = 0
    if "temp_growth_class" in df.columns and n_skip:
        tgc = df.loc[join.eq("skipped_single_temp"), "temp_growth_class"]
        n_skip_had_temp = int(
            tgc.fillna("").astype(str).str.strip().ne("").sum()
        )
    print(
        f"\n  skipped_single_temp rows whose medium @ref still has a "
        f"culture-temp flag: {n_skip_had_temp:,} / {n_skip:,}"
    )
    print(
        "  (these are counted as culture medium only because matching is "
        "gated on >1 distinct temperature)"
    )

    print("\n=== recommended headline counts ===")
    n_medium = int(src.eq("culture medium").sum())
    print(f"  rows                          : {n:,}")
    print(f"  label from culture medium     : {n_medium:,}")
    print(f"  label from culture temp, agree: {n_agreed:,}  "
          f"(medium already had this label)")
    print(f"  label from culture temp, flip : {n_ov:,}  "
          f"(report this as culture-temp-driven labels)")
    print(f"  positives                     : {int((y == 1).sum()):,}")
    print(f"  negatives                     : {int((y == 0).sum()):,}")
    n_neg_ov = int((ov.eq(1) & (y == 0)).sum())
    n_neg = int((y == 0).sum())
    if n_neg:
        print(
            f"  negatives that are temp flips : {n_neg_ov:,} / {n_neg:,} "
            f"({n_neg_ov / n_neg:.1%})"
        )


if __name__ == "__main__":
    main()
