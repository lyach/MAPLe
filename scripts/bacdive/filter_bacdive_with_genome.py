#!/usr/bin/env python3
"""Keep BacDive growth rows whose strain has a genome accession.

Drops rows from the growth parquet whose genome_accession is an empty
string ''.

Usage:
   python scripts/bacdive/filter_bacdive_with_genome.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRO = ROOT / "data" / "processed"
DEFAULT_PARQUET = PRO / "bacdive" / "bd_temp.parquet"


def blank(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().replace({"None": "", "nan": ""})


def default_out(parquet: Path) -> Path:
    """ Add `_w_genome` suffix if not present.
    """
    if parquet.stem.endswith("_w_genome"):
        return parquet
    return parquet.with_name(f"{parquet.stem}_w_genome{parquet.suffix}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET,
                   help="Growth parquet to filter (default: %(default)s)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output parquet (default: <input stem>_w_genome.parquet)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or default_out(args.parquet)
    if out == args.parquet:
        raise SystemExit(
            f"refusing to overwrite the input table in place: {out}. "
            "Pass an explicit --out."
        )

    df = pd.read_parquet(args.parquet)
    ids = blank(df["bacdive_id"])

    # empty string, not NA — drop rows with no recorded accession
    has_acc = df["genome_accession"] != ""
    n_empty = int((~has_acc).sum())
    filtered = df.loc[has_acc].copy()

    print(f"parquet in : {len(df):,} rows, {ids.nunique():,} strains")
    print(f"dropped empty genome_accession: {n_empty:,}")
    print(f"parquet out: {len(filtered):,} rows, "
          f"{blank(filtered['bacdive_id']).nunique():,} strains")

    out.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
