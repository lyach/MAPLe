"""
modular_gene_pathway_membership.py
=====================================
Builds a per-strain, gene x KEGG-module BINARY MEMBERSHIP matrix from this
pipeline's PyKofamSearch output (ko_hits.tsv). Rows are every gene Pyrodigal
predicted for that genome (including zero-hit genes, kept as explicit
all-zero rows -- not dropped), columns are KEGG module IDs, values are
binary, MULTI-LABEL (a gene's KO can belong to more than one module at
once).

Column scope is this pipeline's own kegg_pathway_completeness.csv's module
set (see modular_completeness.py / modular_kegg_modules.py) -- give_
completeness computes real module IDs directly for a broad catalog, and
this keeps the two matrices joinable on the same module-ID axis.

The matrix-builder here reindexes the full gene universe (always unique,
from Pyrodigal's .faa headers) SEPARATELY from the hits Series, because
ko_hits.tsv is genuinely long/multi-row per gene -- a gene can hit more
than one distinct KO (confirmed in PyKofamSearch's own README example: one
protein with two separate KO hits), so the hits Series' gene_id index can
be duplicated. Reindexing directly against a duplicated index would emit
duplicate rows; passing the full gene universe in as its own unique list
avoids that.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger("pathway_features.modular_gene_pathway_membership")


def build_gene_module_matrix(
    gene_universe: list[str],
    hits: pd.Series,
    ko_to_modules: dict[str, list[str]],
    module_ids: list[str],
) -> pd.DataFrame:
    """Binary gene_id x module_id DataFrame, multi-label.

    gene_universe -- every gene Pyrodigal predicted for this strain (from
                      modular_ko.load_gene_universe), always unique -- the
                      full row set.
    hits          -- this strain's KO hits (from modular_ko.kos_for_
                      accession): gene_id index (duplicated if a gene hit
                      multiple KOs), ko_id values. Every entry here IS a
                      real hit -- ko_hits.tsv never has blank/no-hit rows --
                      so no truthiness filter is needed before expanding to
                      modules.

    Genes absent from `hits` entirely, or whose KO(s) aren't part of any of
    `module_ids`, get an explicit all-zero row -- never dropped.
    """
    pairs = [
        (gene_id, module_id)
        for gene_id, ko in hits.items()
        for module_id in ko_to_modules.get(ko, ())
    ]

    if pairs:
        pair_df = pd.DataFrame(pairs, columns=["gene_id", "module_id"]).drop_duplicates()
        matrix = pd.crosstab(pair_df["gene_id"], pair_df["module_id"])
    else:
        matrix = pd.DataFrame(index=pd.Index([], name="gene_id"))

    matrix = matrix.reindex(index=gene_universe, columns=module_ids, fill_value=0)
    matrix = matrix.clip(upper=1).astype(int)
    matrix.index.name = "gene_id"
    return matrix
