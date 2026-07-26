"""
distillate.py
==============
Parses DRAM's distill-step output (product.tsv) -- a genome x module table
of percent-completeness values -- into a tidy strain x pathway completeness
matrix, joined back to canonical strain names via strains_metadata.csv.

DRAM's product.tsv indexes rows by genome ID, which (given how
run_dram_annotate.sbatch invokes `DRAM.py annotate` on data/external/fasta/
*.fna) is the FASTA filename stem, i.e. the NCBI assembly accession
(e.g. "GCF_000193495.2"). Column values are completeness fractions in
[0, 1] (DRAM reports some columns as booleans/labels rather than fractions --
see load_product_tsv's coercion note).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("pathway_features.distillate")


def load_product_tsv(product_tsv: Path) -> pd.DataFrame:
    """Load DRAM's product.tsv, indexed by genome ID (accession). Non-numeric
    columns (DRAM includes a few text/label columns alongside the
    completeness fractions) are dropped with a warning, since they can't
    feed the completeness matrix or PCA.
    """
    if not product_tsv.exists():
        raise FileNotFoundError(
            f"DRAM distillate not found at {product_tsv} -- run DRAM on DRAC "
            "first and copy its distillate/ output back into "
            "data/external/dram/distillate/. See docs/pathway_features.md."
        )

    df = pd.read_csv(product_tsv, sep="\t", index_col=0)
    df.index.name = "ncbi_assembly_accession"

    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    all_nan_cols = [c for c in numeric_df.columns if numeric_df[c].isna().all()]
    if all_nan_cols:
        log.warning(
            "Dropping %d non-numeric product.tsv column(s): %s",
            len(all_nan_cols), all_nan_cols,
        )
        numeric_df = numeric_df.drop(columns=all_nan_cols)

    log.info("Loaded product.tsv: %d genome(s) x %d module(s)", *numeric_df.shape)
    return numeric_df


def join_strain_names(completeness: pd.DataFrame, strains_metadata_csv: Path) -> pd.DataFrame:
    """Join a completeness matrix indexed by accession to strain names,
    returning a copy indexed by strain instead. Logs (does not raise on)
    any accessions present in one side but not the other.
    """
    meta = pd.read_csv(strains_metadata_csv)
    accession_to_strain = dict(zip(meta["ncbi_assembly_accession"], meta["strain"]))

    unmatched = set(completeness.index) - set(accession_to_strain)
    if unmatched:
        log.warning(
            "%d genome(s) in the distillate have no matching strain in "
            "strains_metadata.csv (dropped): %s",
            len(unmatched), sorted(unmatched),
        )

    missing = set(accession_to_strain) - set(completeness.index)
    if missing:
        log.warning(
            "%d strain(s) in strains_metadata.csv are missing from the "
            "distillate (DRAM may have failed on them): %s",
            len(missing), sorted(accession_to_strain[a] for a in missing),
        )

    matched = completeness.loc[completeness.index.intersection(accession_to_strain)].copy()
    matched.index = matched.index.map(accession_to_strain)
    matched.index.name = "strain"

    log.info("Joined %d/%d strain(s) to the distillate", len(matched), len(meta))
    return matched
