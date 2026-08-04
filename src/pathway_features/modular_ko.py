"""Loads PyKofamSearch's KO hits and Pyrodigal's per-genome gene universe."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("pathway_features.modular_ko")


def load_ko_hits(ko_hits_tsv: Path) -> pd.DataFrame:
    """Load id_protein/id_ko columns from PyKofamSearch's ko_hits.tsv."""
    if not ko_hits_tsv.exists():
        raise FileNotFoundError(
            f"PyKofamSearch output not found at {ko_hits_tsv} -- run the "
            "pipeline on DRAC first and copy "
            "$MODULAR_ROOT/output/ko_annotation/ back into "
            "data/external/modular_pathway/ko_annotation/. See "
            "docs/modular_pathway_features.md."
        )
    df = pd.read_csv(ko_hits_tsv, sep="\t", usecols=["id_protein", "id_ko"])
    log.info("Loaded ko_hits.tsv: %d hit(s) across %d protein(s)", len(df), df["id_protein"].nunique())
    return df


def kos_for_accession(ko_hits: pd.DataFrame, accession: str) -> pd.Series:
    """gene_id -> ko_id for one genome's hits."""
    subset = ko_hits.loc[ko_hits["id_protein"].str.startswith(accession + "_")]
    return subset.set_index("id_protein")["id_ko"]


def load_gene_universe(genes_dir: Path, accession: str) -> list[str]:
    """All gene IDs Pyrodigal predicted for one genome, accession-prefixed
    to match ko_hits.tsv.
    """
    faa_path = genes_dir / f"{accession}.faa"
    if not faa_path.exists():
        raise ValueError(
            f"No protein FASTA found for accession {accession!r} at {faa_path} -- "
            "Pyrodigal may have failed on this genome, or the accession is wrong."
        )
    gene_ids = [
        f"{accession}_{line[1:].split(None, 1)[0]}"
        for line in faa_path.read_text(encoding="utf-8").splitlines()
        if line.startswith(">")
    ]
    if not gene_ids:
        raise ValueError(f"{faa_path} has zero predicted genes -- Pyrodigal may have failed on this genome.")
    return gene_ids
