#!/usr/bin/env python3
"""Compare in-house experimental strains with BacDive by genome accession.

Matches ncbi_assembly_accession (version suffix '.N' stripped, e.g.
GCF_000193495.2 -> GCF_000193495) to genome_accession. GCF_ and GCA_ are
treated as the same assembly number: BacDive stores GCA stems, the in-house
table stores GCF versions of the same genomes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "data" / "external"
DEFAULT_CSV = EXT / "strains_metadata.csv"
DEFAULT_PARQUET = EXT / "bacdive_growth_40k.parquet"

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.max_rows", 200)


def blank(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().replace({"None": "", "nan": ""})


def strip_version(value: str) -> str:
    """GCF_000193495.2 -> GCF_000193495. No-op if there is no '.N' suffix."""
    s = str(value).strip()
    if s.lower() in {"", "none", "nan"}:
        return ""
    if "." in s:
        body, ver = s.rsplit(".", 1)
        if ver.isdigit():
            return body
    return s


def acc_key(value: str) -> str:
    """Version-stripped, GCF/GCA-equivalent: GCF_000193495.2 -> 000193495."""
    stem = strip_version(value)
    return stem[4:] if stem[:4] in ("GCF_", "GCA_") else stem


def species_key(name: str) -> str:
    parts = str(name).split()
    return " ".join(parts[:2]).lower() if len(parts) >= 2 else str(name).lower()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                   help="In-house strains CSV (default: %(default)s)")
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET,
                   help="BacDive growth parquet (default: %(default)s)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ih = pd.read_csv(args.csv, dtype=str).fillna("")
    ih["acc_stem"] = blank(ih["ncbi_assembly_accession"]).map(strip_version)
    ih["acc_key"] = blank(ih["ncbi_assembly_accession"]).map(acc_key)
    ih["species"] = blank(ih["strain"]).map(species_key)
    n_ih_blank = int((ih.acc_key == "").sum())
    ih_keyed = ih.loc[ih.acc_key != ""].copy()

    bd = pd.read_parquet(args.parquet, columns=[
        "bacdive_id", "strain_name_raw", "genome_accession",
    ])
    bd["acc_stem"] = blank(bd["genome_accession"]).map(strip_version)
    bd["acc_key"] = blank(bd["genome_accession"]).map(acc_key)
    n_bd_blank = int((bd.acc_key == "").sum())

    n_rows_per_strain = (
        bd.groupby("bacdive_id").size().rename("n_growth_rows")
    )
    bd_strains = (
        bd.drop_duplicates("bacdive_id")
        [["bacdive_id", "strain_name_raw", "genome_accession", "acc_stem", "acc_key"]]
        .merge(n_rows_per_strain, on="bacdive_id", how="left")
    )
    bd_strains["species"] = blank(bd_strains["strain_name_raw"]).map(species_key)
    bd_keyed = bd_strains.loc[bd_strains.acc_key != ""].copy()

    ih_keys = set(ih_keyed.acc_key)
    bd_keys = set(bd_keyed.acc_key)
    common_keys = ih_keys & bd_keys
    ih_only_keys = ih_keys - bd_keys

    # prefix-sensitive stem (GCF_000193495 vs GCA_000193495 stay distinct)
    prefix_overlap = set(ih_keyed.acc_stem) & set(bd_keyed.acc_stem)

    overlap = (
        ih_keyed.loc[ih_keyed.acc_key.isin(common_keys),
                     ["strain", "ncbi_assembly_accession", "acc_key", "species"]]
        .merge(
            bd_keyed.loc[bd_keyed.acc_key.isin(common_keys),
                         ["bacdive_id", "strain_name_raw", "genome_accession",
                          "acc_key", "n_growth_rows"]],
            on="acc_key",
            how="left",
        )
        .sort_values(["strain", "bacdive_id"])
    )
    ih_only = (
        ih_keyed.loc[ih_keyed.acc_key.isin(ih_only_keys),
                     ["strain", "ncbi_assembly_accession", "species"]]
        .sort_values("strain")
        .copy()
    )
    ih_only["n_bacdive_same_species"] = ih_only.species.map(
        bd_keyed.species.value_counts()
    ).fillna(0).astype(int)

    n_ih = ih["strain"].nunique()
    n_bd = bd_strains["bacdive_id"].nunique()
    n_common_ih = overlap["strain"].nunique()
    n_ih_only = ih_only["strain"].nunique()
    n_bd_only = n_bd - bd_keyed.loc[bd_keyed.acc_key.isin(common_keys),
                                    "bacdive_id"].nunique()
    n_overlap_rows = int(overlap.drop_duplicates("bacdive_id").n_growth_rows.sum())

    ih_species = set(ih.species) - {""}
    bd_species = set(bd_strains.species) - {""}
    species_overlap = ih_species & bd_species

    print(f"in-house CSV : {len(ih):,} rows, {n_ih:,} strains, "
          f"{ih_keyed.acc_key.nunique():,} accessions "
          f"({n_ih_blank} blank)")
    print(f"BacDive parquet: {len(bd):,} rows, {n_bd:,} strains, "
          f"{bd_keyed.acc_key.nunique():,} accessions "
          f"({n_bd_blank} blank rows)")
    print("match key    : version suffix stripped; GCF_/GCA_ treated as equivalent")
    print()
    print(f"overlap      : {n_common_ih:,} in-house strains  "
          f"({100 * n_common_ih / n_ih:.1f}% of in-house; "
          f"{100 * n_common_ih / n_bd:.2f}% of BacDive)")
    print(f"in-house only: {n_ih_only:,} strains")
    print(f"BacDive only : {n_bd_only:,} strains")
    print(f"prefix-sensitive stem overlap (GCF vs GCA kept distinct): "
          f"{len(prefix_overlap)}")
    print(f"overlap BacDive growth rows: {n_overlap_rows:,}")
    print(f"species-name overlap (independent of genome): "
          f"{len(species_overlap)}/{len(ih_species)} in-house species")

    print(f"\n=== overlapping strains ({n_common_ih}) ===")
    if overlap.empty:
        print("(none)")
    else:
        show = overlap.rename(columns={
            "strain": "inhouse_strain",
            "ncbi_assembly_accession": "inhouse_accession",
            "strain_name_raw": "bacdive_strain",
            "genome_accession": "bacdive_accession",
        })[[
            "inhouse_strain", "bacdive_strain",
            "inhouse_accession", "bacdive_accession",
            "bacdive_id", "n_growth_rows",
        ]]
        print(show.to_string(index=False))

    print(f"\n=== in-house strains not in BacDive ({n_ih_only}) ===")
    if ih_only.empty:
        print("(none)")
    else:
        show = ih_only.rename(columns={
            "ncbi_assembly_accession": "inhouse_accession",
        })[["strain", "inhouse_accession", "n_bacdive_same_species"]]
        print(show.to_string(index=False))
        n_same_sp = int((ih_only.n_bacdive_same_species > 0).sum())
        print(f"\n  of these, {n_same_sp} have >=1 BacDive strain of the same species "
              f"(different assembly)")


if __name__ == "__main__":
    main()
