"""Create the AlignBio report dataframe.

Dataframe to be used in the AlignBio report.
Takes as input the tidy parquet file produced by the dataset pipeline.
Returns a dataframe with columns:
[RunIdx, Plate, Well, Organism, Media, MaxOD, Atmosphere, Temperature, Site]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MEDIA_MAP: dict[str, str] = {
    "lb": "LB",
    "mb": "Marine Medium",
    "nb": "Nutrient Broth",
    "bhi": "BHI",
    "mrs": "MRS Broth",
    "thy": "THY",
    "tsb": "Tryptic Soy Broth",
    "gifu": "GAM",
    "mrc": "MRC",
    "cdm": "CDM",
    "m9": "M9 Minimal Medium",
}

ORGANISM_MAP: dict[str, str] = {
    "bacillus subtilis": "Bacillus subtilis",
    "streptococcus thermophilus": "Streptococcus thermophilus",
    "priestia megaterium": "Priestia megaterium",
    "cupriavidus necator": "Cupriavidus necator",
    "kocuria rhizophila": "Kocuria rhizophila",
    "escherichia coli bl21": "Escherichia coli  BL21",
    "arthrobacter globiformis": "Arthrobacter globiformis",
    "azotobacter vinelandii": "Azotobacter vinelandii",
    "acetobacter aceti": "Acetobacter aceti",
    "caulobacter vibrioides": "Caulobacter vibrioides",
    "escherichia coli k12": "Escherichia coli  K12",
    "acetobacter pasteurianus": "Acetobacter pasteurianus",
    "gluconobacter oxydans": "Gluconobacter oxydans",
    "paracoccus denitrificans": "Paracoccus denitrificans",
    "micrococcus luteus (schroeter) cohn": "Micrococcus luteus (Schroeter) Cohn",
    "pseudomonas putida": "Pseudomonas putida  KT2440",
    "bifidobacterium animalis": "Bifidobacterium animalis  subsp. lactis",
    "corynebacterium glutamicum": "Corynebacterium glutamicum  534",
    "deinococcus radiodurans": "Deinococcus radiodurans",
    "acinetobacter baylyi": "Acinetobacter baylyi",
}


def create_align_df(
    parquet_path: str | Path,
    drop_controls: bool = True,
) -> pd.DataFrame:
    """Build the AlignBio report DataFrame from a tidy parquet file.

    Parameters
    ----------
    parquet_path:
        Path to the tidy long-format parquet produced by the dataset pipeline.
    drop_controls:
        If ``True`` (default), rows where ``media_name == "water"`` or
        ``strain_name == "empty"`` are excluded before building the report.

    Returns
    -------
    pd.DataFrame
        Wide-format report with one row per (exp_id, well_id) and columns:
        RunIdx, Plate, Well, Organism, Media, MaxOD, Atmosphere, Temperature, Site.
    """
    df = pd.read_parquet(parquet_path)

    # Keep only sample wells (exclude blanks)
    df = df.loc[~df["is_blank"]].copy()

    if drop_controls:
        mask = (df["media_name"].str.lower() == "water") | (
            df["strain_name"].str.lower() == "empty"
        )
        df = df.loc[~mask]

    # Per-well MaxOD
    max_od = (
        df.groupby(["exp_id", "well_id"])["od600"]
        .max()
        .rename("MaxOD")
        .reset_index()
    )

    # One metadata row per well (first timepoint is sufficient)
    meta_cols = [
        "exp_id",
        "well_id",
        "row",
        "col",
        "well_index",
        "strain_name",
        "media_name",
        "atmosphere",
        "temperature",
    ]
    meta = (
        df.sort_values("timepoint")
        .groupby(["exp_id", "well_id"], sort=False)[meta_cols]
        .first()
        .reset_index(drop=True)
    )

    wide = meta.merge(max_od, on=["exp_id", "well_id"], how="left")

    # RunIdx: 1-based well index within each plate
    wide = wide.sort_values(["exp_id", "well_index"])
    wide["RunIdx"] = wide.groupby("exp_id").cumcount() + 1

    # Well label: capitalised row + col (e.g. "A1")
    wide["Well"] = wide["row"].str.upper() + wide["col"].astype(str)

    # Apply name mappings (unmapped values fall back to the original)
    wide["Organism"] = (
        wide["strain_name"].str.lower().map(ORGANISM_MAP).fillna(wide["strain_name"])
    )
    wide["Media"] = (
        wide["media_name"].str.lower().map(MEDIA_MAP).fillna(wide["media_name"])
    )

    wide["Site"] = "UT"

    report = wide.rename(
        columns={
            "exp_id": "Plate",
            "atmosphere": "Atmosphere",
            "temperature": "Temperature",
        }
    )[
        [
            "RunIdx",
            "Plate",
            "Well",
            "Organism",
            "Media",
            "MaxOD",
            "Atmosphere",
            "Temperature",
            "Site",
        ]
    ]

    return report.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the AlignBio report DataFrame.")
    parser.add_argument("input", type=Path, help="Tidy parquet file from the dataset pipeline.")
    parser.add_argument("output", type=Path, help="Output CSV path.")
    parser.add_argument(
        "--keep-controls",
        action="store_true",
        default=False,
        help="Keep water-media and empty-organism rows (dropped by default).",
    )
    args = parser.parse_args()

    df = create_align_df(args.input, drop_controls=not args.keep_controls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
