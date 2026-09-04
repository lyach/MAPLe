"""
bacdive_loader.py
=================
Read a BacDive growth/metabolite parquet, collapse it to one row per
isolate (`bacdive_id`), and emit StrainInput records for the NCBI
resolution pipeline.

The growth table is (strain, medium) grain -- NCBI metadata is per isolate,
so we resolve once and left-join the result back onto every growth row.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .align_loader import StrainInput
from .config import PARQUET_MERGE_FIELDNAMES

log = logging.getLogger("bacdive_loader")

# Columns used as NCBI / isolate handles. `bacdive_id` is required; the rest
# are used when present.
IDENTITY_COLUMNS = [
    "bacdive_id",
    "strain_name",
    "strain_name_raw",
    "genome_accession",
    "genome_accession_source",
    "genome_link",
    "n16s_accession",
    "strain_designation",
    "is_type_strain",
    "dsm_number",
    "culture_collection_ids",
    "ncbi_taxid",
    "ncbi_taxid_level",
]


def _parquet_column_names(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq
        return list(pq.read_schema(path).names)
    except Exception:
        return list(pd.read_parquet(path).columns)


def _cell(row: pd.Series, col: str) -> str:
    if col not in row.index:
        return ""
    val = row[col]
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    text = str(val).strip()
    if text.lower() in ("nan", "none"):
        return ""
    return text


def load_bacdive_strains(path: Path) -> list[StrainInput]:
    """Unique isolates from a BacDive parquet, one StrainInput each.

    Raises FileNotFoundError if `path` is missing, ValueError if there is no
    `bacdive_id` column to join on.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find BacDive parquet: {path}\n"
            "Expected e.g. data/external/bacdive_growth_full.parquet"
        )

    available = _parquet_column_names(path)
    if "bacdive_id" not in available:
        raise ValueError(
            f"{path} has no bacdive_id column (found: {available[:20]}...). "
            "Cannot deduplicate isolates or join metadata back."
        )

    cols = [c for c in IDENTITY_COLUMNS if c in available]
    df = pd.read_parquet(path, columns=cols)
    df["_bid"] = df["bacdive_id"].fillna("").astype(str).str.strip()
    df = df.loc[df["_bid"] != ""].drop_duplicates(subset=["_bid"], keep="first")

    strains: list[StrainInput] = []
    for _, row in df.iterrows():
        bid = _cell(row, "bacdive_id")
        raw = _cell(row, "strain_name_raw")
        name = _cell(row, "strain_name") or (f"{bid}_{raw}" if raw else bid)
        strains.append(StrainInput(
            strain=name,
            strain_id=bid,
            organism_name=raw,
            genome_accession=_cell(row, "genome_accession"),
            genome_accession_source=_cell(row, "genome_accession_source"),
            genome_link=_cell(row, "genome_link"),
            n16s_accession=_cell(row, "n16s_accession"),
            strain_designation=_cell(row, "strain_designation"),
            is_type_strain=_cell(row, "is_type_strain"),
            dsm_number=_cell(row, "dsm_number"),
            culture_collection_ids=_cell(row, "culture_collection_ids"),
            ncbi_taxid=_cell(row, "ncbi_taxid"),
            ncbi_taxid_level=_cell(row, "ncbi_taxid_level"),
        ))

    log.info("Loaded %d unique isolate(s) from %s (%d identity column(s))",
             len(strains), path, len(cols))
    return strains


def merge_metadata_onto_parquet(
    parquet_path: Path,
    metadata_csv: Path,
    output_path: Path,
) -> int:
    """Left-join NCBI sidecar columns onto the original parquet on bacdive_id.

    Does not overwrite existing BacDive columns (genus/family/order/phylum/
    gc_content). Returns the number of output rows.
    """
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata sidecar not found: {metadata_csv}")

    src = pd.read_parquet(parquet_path)
    meta = pd.read_csv(metadata_csv, dtype=str).fillna("")
    if "bacdive_id" not in meta.columns:
        raise ValueError(
            f"{metadata_csv} has no bacdive_id column -- cannot join onto parquet"
        )

    add_cols = [c for c in PARQUET_MERGE_FIELDNAMES if c not in src.columns]
    keep = ["bacdive_id"] + [c for c in add_cols if c in meta.columns]
    meta_small = (
        meta[keep]
        .copy()
        .drop_duplicates(subset=["bacdive_id"], keep="first")
    )
    meta_small["_join"] = meta_small["bacdive_id"].astype(str).str.strip()
    src = src.copy()
    src["_join"] = src["bacdive_id"].fillna("").astype(str).str.strip()

    merged = src.merge(
        meta_small.drop(columns=["bacdive_id"]),
        on="_join",
        how="left",
    )
    merged = merged.drop(columns=["_join"])
    for col in add_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    n_matched = int((merged.get("ncbi_assembly_accession", pd.Series(dtype=str))
                     .fillna("").astype(str).str.strip() != "").sum())
    log.info(
        "Wrote %s (%d rows, %d with ncbi_assembly_accession) from %d unique metadata row(s)",
        output_path, len(merged), n_matched, len(meta_small),
    )
    return len(merged)
