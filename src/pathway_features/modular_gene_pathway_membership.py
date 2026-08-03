"""Builds the per-strain gene x KEGG-module binary membership matrix."""

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
    """Binary gene_id x module_id DataFrame, multi-label, all genes included."""
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
