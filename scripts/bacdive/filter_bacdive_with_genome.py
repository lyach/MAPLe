#!/usr/bin/env python3
"""Keep BacDive growth rows whose strain has a CDS FASTA on disk.

Reads bacdive_genome_map.csv, takes bacdive_id values with cds_on_disk == 1,
filters bacdive_growth_x.parquet to those IDs, then drops rows whose
genome_accession is an empty string ''.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "data" / "external"
DEFAULT_MAP = EXT / "bacdive_genome_map.csv"

# 132k BacDive (with temperature range expansion)
# DEFAULT_PARQUET = EXT / "bacdive_growth_132k.parquet"
# DEFAULT_OUT = EXT / "bacdive_growth_52k.parquet"

# 107k BacDive (without temperature range expansion)
# DEFAULT_PARQUET = EXT / "bacdive_growth_107k.parquet"
# DEFAULT_OUT = EXT / "bacdive_growth_40k.parquet"

# 132k BacDive (with temperature range expansion and keep negative labels)
DEFAULT_PARQUET = EXT / "bacdive_growth_132k_dedup_w_negs.parquet"
DEFAULT_OUT = EXT / "bacdive_growth_52k_dedup_w_negs.parquet"


def blank(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().replace({"None": "", "nan": ""})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", type=Path, default=DEFAULT_MAP,
                   help="Genome map CSV (default: %(default)s)")
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET,
                   help="Growth parquet to filter (default: %(default)s)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Filtered parquet output (default: %(default)s)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    gmap = pd.read_csv(args.map, dtype=str).fillna("")
    cds_ids = set(gmap.loc[gmap.cds_on_disk == "1", "bacdive_id"])
    print(f"genome_map: {len(gmap)} rows, {len(cds_ids)} bacdive_id with cds_on_disk==1")

    df = pd.read_parquet(args.parquet)
    ids = blank(df["bacdive_id"])
    keep = ids.isin(cds_ids)
    filtered = df.loc[keep].copy()
    n_cds = len(filtered)

    # empty string, not NA — drop rows with no recorded accession
    has_acc = filtered["genome_accession"] != ""
    n_empty = int((~has_acc).sum())
    filtered = filtered.loc[has_acc].copy()

    print(f"parquet in : {len(df):,} rows, {ids.nunique():,} strains")
    print(f"after cds  : {n_cds:,} rows "
          f"({blank(df.loc[keep, 'bacdive_id']).nunique():,} strains)")
    print(f"dropped empty genome_accession: {n_empty:,}")
    print(f"parquet out: {len(filtered):,} rows, "
          f"{blank(filtered['bacdive_id']).nunique():,} strains")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
