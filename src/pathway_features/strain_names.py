"""
strain_names.py
=================
Joins any accession-indexed DataFrame (a completeness matrix, a confounder
feature table, ...) to canonical strain names via strains_metadata.csv.
Annotation-source-independent -- used by the modular pipeline's
build_modular_pathway_features.py for both its completeness matrix and its
CheckM2 confounder output.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("pathway_features.strain_names")


def join_strain_names(df: pd.DataFrame, strains_metadata_csv: Path) -> pd.DataFrame:
    """Join a DataFrame indexed by NCBI assembly accession to strain names,
    returning a copy indexed by strain instead. Logs (does not raise on)
    any accessions present in one side but not the other.
    """
    meta = pd.read_csv(strains_metadata_csv)
    accession_to_strain = dict(zip(meta["ncbi_assembly_accession"], meta["strain"]))

    unmatched = set(df.index) - set(accession_to_strain)
    if unmatched:
        log.warning(
            "%d genome(s) have no matching strain in strains_metadata.csv (dropped): %s",
            len(unmatched), sorted(unmatched),
        )

    missing = set(accession_to_strain) - set(df.index)
    if missing:
        log.warning(
            "%d strain(s) in strains_metadata.csv are missing from the input "
            "(the pipeline may have failed on them): %s",
            len(missing), sorted(accession_to_strain[a] for a in missing),
        )

    matched = df.loc[df.index.intersection(accession_to_strain)].copy()
    matched.index = matched.index.map(accession_to_strain)
    matched.index.name = "strain"

    log.info("Joined %d/%d strain(s)", len(matched), len(meta))
    return matched
