"""Feature engineering pipeline.

Builds an sklearn ``ColumnTransformer`` from the dataset config's feature.  
Each column maps to an encoding:

- ``"OH"``          -> OneHotEncoder
- ``"ordinal"``     -> OrdinalEncoder
- ``"numeric"``     -> StandardScaler
- ``"path:<file>"`` -> precomputed embeddings merged by name
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from src.modeling.config import DatasetConfig

# Maps feature columns to semantic groups
FEATURE_GROUPS: dict[str, str] = {
    "strain_name": "strain",
    "media_name": "media",
    "atmosphere": "condition",
    "temperature": "condition",
}


def _load_embeddings(path: str | Path) -> pd.DataFrame:
    """Load a precomputed embedding CSV (first column = key, rest = floats)."""
    df = pd.read_csv(path)
    key_col = df.columns[0]
    df[key_col] = df[key_col].astype(str).str.strip().str.lower()
    return df


def build_feature_pipeline(
    cfg: DatasetConfig,
    df: pd.DataFrame,
    root: Path | None = None,
) -> tuple[ColumnTransformer, list[str], dict[str, str]]:
    """Build an sklearn ``ColumnTransformer`` from the dataset config.

    Parameters
    ----------
    cfg:
        Dataset config whose ``features`` dict maps column names to
        encoding specifications.
    df:
        The modeling DataFrame (used only to resolve embedding merges
        and validate column presence -- **not** fitted here).
    root:
        Project root for resolving relative embedding paths.

    Returns
    -------
    transformer:
        Unfitted ``ColumnTransformer``.
    feature_columns:
        Ordered list of raw columns consumed by the transformer.
    feature_group_map:
        Maps each raw column to a semantic group (``"strain"``,
        ``"media"``, ``"condition"``, ``"genome"``, ``"embedding"``).
    """
    if root is None:
        root = Path.cwd()

    transformers: list[tuple[str, Any, list[str]]] = []
    feature_columns: list[str] = []
    feature_group_map: dict[str, str] = {}

    for col, spec in cfg.features.items():
        if col not in df.columns:
            logger.warning("feature column '{}' not in DataFrame — skipping", col)
            continue

        spec_lower = spec.strip().upper() if not spec.startswith("path:") else spec

        if spec_lower == "OH":
            transformers.append((
                f"oh_{col}",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                [col],
            ))
            feature_columns.append(col)
            feature_group_map[col] = FEATURE_GROUPS.get(col, "other")

        elif spec_lower == "ORDINAL":
            transformers.append((
                f"ord_{col}",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                [col],
            ))
            feature_columns.append(col)
            feature_group_map[col] = FEATURE_GROUPS.get(col, "other")

        elif spec_lower == "NUMERIC":
            transformers.append((
                f"num_{col}",
                StandardScaler(),
                [col],
            ))
            feature_columns.append(col)
            feature_group_map[col] = FEATURE_GROUPS.get(col, "genome")

        elif spec.startswith("path:"):
            emb_path = root / spec[5:].strip()
            logger.info("loading embeddings from {}", emb_path)
            emb_df = _load_embeddings(emb_path)
            emb_key = emb_df.columns[0]
            emb_cols = [c for c in emb_df.columns if c != emb_key]
            group_name = "strain_emb" if "strain" in col else "media_emb"
            for ec in emb_cols:
                feature_group_map[ec] = group_name
            # passthrough the numeric embedding columns
            transformers.append((
                f"emb_{col}",
                "passthrough",
                emb_cols,
            ))
            feature_columns.extend(emb_cols)

        else:
            logger.warning("unknown feature spec '{}' for column '{}' — skipping", spec, col)

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return ct, feature_columns, feature_group_map


def warn_loso_onehot(cfg: DatasetConfig, cv_strategy: str) -> None:
    """Warn if identity one-hot is the only strain/media feature under LOSO/LOMO."""
    if cv_strategy == "loso":
        strain_specs = {
            col: spec
            for col, spec in cfg.features.items()
            if FEATURE_GROUPS.get(col) == "strain"
        }
        if strain_specs and all(s.upper() == "OH" for s in strain_specs.values()):
            logger.warning(
                "LOSO CV with only one-hot strain_name: the held-out strain "
                "will have an all-zero feature vector. Add genome/taxonomy "
                "features for meaningful generalization."
            )

    if cv_strategy == "lomo":
        media_specs = {
            col: spec
            for col, spec in cfg.features.items()
            if FEATURE_GROUPS.get(col) == "media"
        }
        if media_specs and all(s.upper() == "OH" for s in media_specs.values()):
            logger.warning(
                "LOMO CV with only one-hot media_name: the held-out medium "
                "will have an all-zero feature vector. Add ECFP embeddings "
                "for meaningful generalization."
            )


def get_feature_names(
    ct: ColumnTransformer,
) -> list[str]:
    """Extract feature names from a *fitted* ColumnTransformer."""
    return list(ct.get_feature_names_out())


def build_feature_group_index(
    feature_names: list[str],
    feature_group_map: dict[str, str],
) -> dict[str, list[int]]:
    """Map each semantic group to feature-index positions in X.

    Used by ``interpret.py`` for grouped SHAP importance.
    """
    group_idx: dict[str, list[int]] = {}
    for i, fname in enumerate(feature_names):
        matched = False
        for raw_col, group in feature_group_map.items():
            if raw_col in fname:
                group_idx.setdefault(group, []).append(i)
                matched = True
                break
        if not matched:
            group_idx.setdefault("other", []).append(i)
    return group_idx


def featurize(
    df: pd.DataFrame,
    cfg: DatasetConfig,
    target: str = "growth_class",
    root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, dict[str, str]]:
    """ Wrapper or EDA/notebook use only: build pipeline, fit on *all* data, transform.

    In CV, the transformer must be fit per fold via ``build_feature_pipeline``.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    feature_names : list[str]
    groups : ndarray of strain names (for group-aware CV)
    feature_group_map : dict mapping raw columns to semantic groups
    """
    ct, _, feature_group_map = build_feature_pipeline(cfg, df, root=root)
    ct.fit(df)
    X = ct.transform(df)
    y = df[target].values
    feature_names = get_feature_names(ct)
    groups = df["strain_name"].values if "strain_name" in df.columns else np.arange(len(df))
    return X, y, feature_names, groups, feature_group_map
