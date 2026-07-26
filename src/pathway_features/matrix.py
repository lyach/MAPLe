"""
matrix.py
=========
Applies a completeness threshold to turn the strain x pathway completeness
matrix into a binary presence/absence matrix (kegg_pathway_presence.csv).

DIAGNOSTIC OUTPUT ONLY -- not a modeling input. The canonical output is the
continuous completeness matrix itself (written directly by
build_pathway_features.py, no module needed); thresholding here throws away
signal a model could otherwise use. This is useful for manual inspection
("which strains have pathway X") and is only written when
build_pathway_features.py is run with --emit-diagnostics.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger("pathway_features.matrix")


def build_presence_matrix(completeness: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return a same-shaped 0/1 DataFrame: 1 where completeness >= threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError(f"completeness_threshold must be in [0, 1], got {threshold}")

    presence = (completeness >= threshold).astype(int)
    n_present = int(presence.values.sum())
    n_total = presence.size
    log.info(
        "Presence matrix at threshold=%.2f: %d/%d (strain, pathway) cells present",
        threshold, n_present, n_total,
    )
    return presence
