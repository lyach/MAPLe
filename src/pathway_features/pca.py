"""
pca.py
======
Reduces the strain x pathway completeness matrix to a small set of principal
components. Runs on raw completeness percentages (not the thresholded binary
matrix) to preserve the gradient between e.g. 40%- and 90%-complete pathways.

EDA/VISUALIZATION ONLY -- this fits PCA once across ALL strains, which is
NOT valid for modeling: it leaks test-strain information into the transform
before cross-validation ever splits the data. Its output
(eda_pca_exploratory.csv) must never be joined into training data as-is. If
the modeling pipeline wants pathway features dimensionality-reduced, it must
fit its own PCA inside the CV loop, on training strains only, each fold.
Only written when build_pathway_features.py is run with --emit-diagnostics.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("pathway_features.pca")


def run_eda_pca(
    completeness: pd.DataFrame,
    n_components: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize and PCA-reduce the completeness matrix.

    Returns (scores, loadings):
        scores   -- strain x component DataFrame (PC1, PC2, ...)
        loadings -- component x pathway DataFrame, for projecting new
                    strains later without refitting.
    """
    n_components = min(n_components, completeness.shape[0], completeness.shape[1])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(completeness.values)

    pca = PCA(n_components=n_components, random_state=random_state)
    transformed = pca.fit_transform(scaled)

    component_names = [f"PC{i + 1}" for i in range(n_components)]
    scores = pd.DataFrame(transformed, index=completeness.index, columns=component_names)
    loadings = pd.DataFrame(
        pca.components_, index=component_names, columns=completeness.columns,
    )

    explained = pca.explained_variance_ratio_.sum()
    log.info(
        "PCA: %d component(s) explain %.1f%% of variance across %d pathway(s)",
        n_components, explained * 100, completeness.shape[1],
    )
    return scores, loadings
