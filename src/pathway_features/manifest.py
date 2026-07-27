"""
manifest.py
===========
Builds the strain <-> accession <-> FASTA-path table that gets shipped to
DRAC for annotation. Cross-validates strains_metadata.csv against the
FASTA files actually on disk so mismatches are caught locally, before an
expensive remote run, rather than surfacing as a silent gap in the
annotation output.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("pathway_features.manifest")


def build_manifest(strains_metadata_csv: Path, fasta_dir: Path) -> pd.DataFrame:
    """Return a DataFrame with one row per strain that has both a metadata
    row and a FASTA file on disk: columns [strain, ncbi_assembly_accession,
    fasta_filename]. Logs (but does not raise on) any mismatches -- callers
    that want a hard failure should check the return value's length against
    the input metadata's.
    """
    if not strains_metadata_csv.exists():
        raise FileNotFoundError(
            f"strains_metadata.csv not found at {strains_metadata_csv} "
            "-- run scripts/get_strain_metadata.py first."
        )
    if not fasta_dir.exists():
        raise FileNotFoundError(f"FASTA directory not found at {fasta_dir}")

    meta = pd.read_csv(strains_metadata_csv)
    required_cols = {"strain", "ncbi_assembly_accession"}
    missing_cols = required_cols - set(meta.columns)
    if missing_cols:
        raise ValueError(
            f"{strains_metadata_csv} is missing required column(s): {sorted(missing_cols)}"
        )

    fasta_accessions = {p.stem: p.name for p in fasta_dir.glob("*.fna")}

    meta_accessions = set(meta["ncbi_assembly_accession"])
    unmatched_fasta = set(fasta_accessions) - meta_accessions
    if unmatched_fasta:
        log.warning(
            "%d FASTA file(s) have no matching row in strains_metadata.csv "
            "(excluded from manifest): %s",
            len(unmatched_fasta), sorted(unmatched_fasta),
        )

    missing_fasta = meta_accessions - set(fasta_accessions)
    if missing_fasta:
        log.warning(
            "%d strain(s) in strains_metadata.csv have no FASTA file "
            "(excluded from manifest): %s",
            len(missing_fasta), sorted(missing_fasta),
        )

    rows = []
    for _, row in meta.iterrows():
        accession = row["ncbi_assembly_accession"]
        if accession not in fasta_accessions:
            continue
        rows.append({
            "strain": row["strain"],
            "ncbi_assembly_accession": accession,
            "fasta_filename": fasta_accessions[accession],
        })

    manifest = pd.DataFrame(rows, columns=["strain", "ncbi_assembly_accession", "fasta_filename"])
    log.info(
        "Built manifest: %d/%d strains matched to a FASTA file",
        len(manifest), len(meta),
    )
    return manifest


def write_manifest(manifest: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_csv, index=False)
    log.info("Wrote manifest -> %s (%d rows)", output_csv, len(manifest))
