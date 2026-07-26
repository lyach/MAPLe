"""
validation.py
==============
Sanity checks run before PCA/matrix construction, so a bad DRAM run or
malformed product.tsv fails loudly with an actionable message instead of
silently producing a degenerate matrix.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger("pathway_features.validation")


def validate_completeness_matrix(completeness: pd.DataFrame, expected_strains: list[str]) -> None:
    """Raise ValueError on conditions that would make the downstream
    matrix/PCA output meaningless or misleading.
    """
    if completeness.empty:
        raise ValueError("Completeness matrix is empty -- check product.tsv parsed correctly.")

    missing_strains = sorted(set(expected_strains) - set(completeness.index))
    if missing_strains:
        raise ValueError(
            f"{len(missing_strains)} strain(s) missing from the distillate after "
            f"joining: {missing_strains}. DRAM may have failed on these genomes -- "
            "check the annotation logs on DRAC before proceeding."
        )

    constant_cols = [
        c for c in completeness.columns
        if completeness[c].nunique(dropna=False) <= 1
    ]
    if constant_cols:
        log.warning(
            "%d pathway(s) are constant across all strains (no signal, dropped "
            "before PCA): %s",
            len(constant_cols), constant_cols[:10] if len(constant_cols) > 10 else constant_cols,
        )
