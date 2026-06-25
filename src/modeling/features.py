"""Feature engineering pipeline.

Builds an sklearn ``ColumnTransformer`` from the dataset config's ``features``
dict.  Each column maps to a transformer call string, or a list of steps applied
in order (e.g. log transform then scale):
    genome_size_bp:
      - log_transform
      - "StandardScaler()"

Embedding columns use ``"path:<file>"`` (precomputed CSV, first col = key).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

from src.modeling.config import DatasetConfig

_TRANSFORMER_REGISTRY: dict[str, type] = {
    "StandardScaler": StandardScaler,
    "RobustScaler": RobustScaler,
    "MinMaxScaler": MinMaxScaler,
    "OneHotEncoder": OneHotEncoder,
    "OrdinalEncoder": OrdinalEncoder,
}

_EVAL_NAMESPACE: dict[str, Any] = {
    "True": True,
    "False": False,
    "None": None,
    **_TRANSFORMER_REGISTRY,
}

# Maps feature columns to semantic groups
FEATURE_GROUPS: dict[str, str] = {
    "strain_name": "strain",
    "media_name": "media",
    "atmosphere": "condition",
    "temperature": "condition",
    "genome_size_bp": "strain",
    "gc_content_pct": "strain",
    "num_contigs": "strain",
    "n50": "strain",
}


_PREPROCESS_REGISTRY: dict[str, FunctionTransformer] = {
    "log_transform": FunctionTransformer(
        np.log1p,
        validate=False,
        feature_names_out="one-to-one",
    ),
}


def _parse_transformer(spec: str) -> Any:
    """Instantiate an sklearn transformer from a config call string."""
    spec = spec.strip()
    try:
        return eval(spec, {"__builtins__": {}}, _EVAL_NAMESPACE)  # noqa: S307
    except Exception as exc:
        raise ValueError(f"invalid transformer spec {spec!r}: {exc}") from exc


def _parse_step(step: str) -> Any:
    """Parse one pipeline step (preprocess keyword or sklearn call string)."""
    step = step.strip()
    if step in _PREPROCESS_REGISTRY:
        return _PREPROCESS_REGISTRY[step]
    return _parse_transformer(step)


def _build_transformer(spec: str | list[str]) -> Any:
    """Build a single transformer or ``Pipeline`` from a feature spec."""
    if isinstance(spec, str):
        return _parse_step(spec)

    if not spec:
        raise ValueError("feature spec list must be non-empty")

    steps: list[tuple[str, Any]] = []
    for i, step in enumerate(spec):
        name = step.strip()
        if name in _PREPROCESS_REGISTRY:
            steps.append((name, _PREPROCESS_REGISTRY[name]))
        else:
            steps.append((f"step_{i}", _parse_transformer(name)))

    if len(steps) == 1:
        return steps[0][1]
    return Pipeline(steps)


def _spec_steps(spec: str | list[str]) -> list[str]:
    return [spec] if isinstance(spec, str) else spec


def _is_onehot_spec(spec: str | list[str]) -> bool:
    return any(s.strip().startswith("OneHotEncoder") for s in _spec_steps(spec))


def _default_feature_group(transformer: Any) -> str:
    if isinstance(transformer, Pipeline):
        for _, step in transformer.steps:
            if step.__class__.__name__.endswith("Scaler"):
                return "genome"
        return "other"
    if transformer.__class__.__name__.endswith("Scaler"):
        return "genome"
    return "other"


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

        if isinstance(spec, str) and spec.startswith("path:"):
            emb_path = root / spec[5:].strip()
            logger.info("loading embeddings from {}", emb_path)
            emb_df = _load_embeddings(emb_path)
            emb_key = emb_df.columns[0]
            emb_cols = [c for c in emb_df.columns if c != emb_key]
            group_name = "strain_emb" if "strain" in col else "media_emb"
            for ec in emb_cols:
                feature_group_map[ec] = group_name
            transformers.append((col, "passthrough", emb_cols))
            feature_columns.extend(emb_cols)
            continue

        try:
            transformer = _build_transformer(spec)
        except ValueError as exc:
            logger.warning("{} — skipping column '{}'", exc, col)
            continue

        transformers.append((col, transformer, [col]))
        feature_columns.append(col)
        feature_group_map[col] = FEATURE_GROUPS.get(col, _default_feature_group(transformer))

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
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
        if strain_specs and all(_is_onehot_spec(s) for s in strain_specs.values()):
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
        if media_specs and all(_is_onehot_spec(s) for s in media_specs.values()):
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
