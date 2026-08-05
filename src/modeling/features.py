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


def _is_external_matrix_spec(spec: Any) -> bool:
    """True when *spec* is a list whose first ``path:`` item points to an external matrix."""
    if not isinstance(spec, list):
        return False
    return any(isinstance(item, str) and item.startswith("path:") for item in spec)


def _parse_external_matrix_spec(
    spec: list,
) -> tuple[str, int | None, str]:
    """Extract ``(csv_path, pca_components, merge_on)`` from a mixed list spec.
    """
    csv_path: str | None = None
    pca_components: int | None = None
    merge_on: str = "strain_name"

    for item in spec:
        if isinstance(item, str) and item.startswith("path:"):
            csv_path = item[5:].strip()
        elif isinstance(item, dict):
            if "pca_components" in item:
                pca_components = int(item["pca_components"])
            if "merge_on" in item:
                merge_on = item["merge_on"]

    if csv_path is None:
        raise ValueError(f"external matrix spec must contain a 'path:' entry, got {spec!r}")

    return csv_path, pca_components, merge_on


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


def merge_external_features(
    cfg: DatasetConfig,
    df: pd.DataFrame,
    root: Path | None = None,
) -> pd.DataFrame:
    """Merge external feature CSVs (``path:`` specs) into the modeling DataFrame.

    Must be called **once** on the full DataFrame before any train/test
    splitting so that both partitions contain the new columns.
    """
    if root is None:
        root = Path.cwd()

    df = df.copy()

    for col, spec in cfg.features.items():
        # --- simple path: spec (existing embeddings) ---
        if isinstance(spec, str) and spec.startswith("path:"):
            if col not in df.columns:
                logger.warning("merge key '{}' not in DataFrame — skipping", col)
                continue

            feat_path = root / spec[5:].strip()
            logger.info("merging external features from {} on key '{}'", feat_path, col)
            feat_df = _load_embeddings(feat_path)
            feat_key = feat_df.columns[0]
            feat_cols = [c for c in feat_df.columns if c != feat_key]

            already_present = [c for c in feat_cols if c in df.columns]
            if already_present:
                logger.debug("columns {} already in DataFrame — skipping merge", already_present)
                continue

            join_series = df[col].astype(str).str.strip().str.lower()
            feat_indexed = feat_df.set_index(feat_key)
            for fc in feat_cols:
                df[fc] = join_series.map(feat_indexed[fc]).values

            n_missing = df[feat_cols[0]].isna().sum()
            if n_missing:
                logger.warning(
                    "{}/{} rows have no match in {} for key '{}'",
                    n_missing, len(df), feat_path.name, col,
                )
            continue

        # --- external matrix spec (pathway matrix + PCA) ---
        if _is_external_matrix_spec(spec):
            csv_path, _pca_components, merge_on = _parse_external_matrix_spec(spec)
            feat_path = root / csv_path

            if merge_on not in df.columns:
                logger.warning(
                    "merge_on key '{}' not in DataFrame — skipping '{}'", merge_on, col,
                )
                continue

            logger.info(
                "merging external matrix from {} on key '{}' (feature '{}')",
                feat_path, merge_on, col,
            )
            feat_df = _load_embeddings(feat_path)
            feat_key = feat_df.columns[0]
            feat_cols = [c for c in feat_df.columns if c != feat_key]

            already_present = [c for c in feat_cols if c in df.columns]
            if already_present:
                logger.debug("columns {} already in DataFrame — skipping merge", already_present)
                continue

            join_keys = df[merge_on].astype(str).str.strip().str.lower()
            feat_indexed = feat_df.set_index(feat_key)[feat_cols]
            merged_block = feat_indexed.reindex(join_keys.values).reset_index(drop=True)
            merged_block.index = df.index

            n_missing = merged_block[feat_cols[0]].isna().sum()
            if n_missing:
                logger.warning(
                    "{}/{} rows have no match in {} for key '{}' — filling with 0.0",
                    n_missing, len(df), feat_path.name, merge_on,
                )
                merged_block = merged_block.fillna(0.0)

            df = pd.concat([df, merged_block], axis=1)
            continue

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
        # --- external matrix spec (pathway CSV + optional PCA) ---
        if _is_external_matrix_spec(spec):
            csv_path, pca_components, _merge_on = _parse_external_matrix_spec(spec)
            feat_path = root / csv_path
            feat_df = _load_embeddings(feat_path)
            feat_key = feat_df.columns[0]
            feat_cols = [c for c in feat_df.columns if c != feat_key]

            missing = [c for c in feat_cols if c not in df.columns]
            if missing:
                logger.error(
                    "columns {} from {} not in DataFrame — "
                    "call merge_external_features() first. Skipping '{}'.",
                    missing, feat_path.name, col,
                )
                continue

            if pca_components is not None:
                from sklearn.decomposition import PCA

                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=pca_components)),
                ])
                transformers.append((col, pipeline, feat_cols))
                feature_columns.extend(feat_cols)

                for i in range(pca_components):
                    feature_group_map[f"pca{i}"] = col
                logger.info(
                    "feature '{}': {} columns -> PCA({}) per fold",
                    col, len(feat_cols), pca_components,
                )
            else:
                transformers.append((f"{col}_num", "passthrough", feat_cols))
                feature_columns.extend(feat_cols)
                for fc in feat_cols:
                    feature_group_map[fc] = col

            continue

        if col not in df.columns:
            logger.warning("feature column '{}' not in DataFrame — skipping", col)
            continue

        # --- simple path: spec (precomputed embeddings) ---
        if isinstance(spec, str) and spec.startswith("path:"):
            feat_path = root / spec[5:].strip()
            feat_df = _load_embeddings(feat_path)
            feat_key = feat_df.columns[0]
            feat_cols = [c for c in feat_df.columns if c != feat_key]

            missing = [c for c in feat_cols if c not in df.columns]
            if missing:
                logger.error(
                    "columns {} from {} not in DataFrame — "
                    "call merge_external_features() first. Skipping.",
                    missing, feat_path.name,
                )
                continue

            group_name = "strain_emb" if "strain" in col else "media_emb"

            numeric_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(feat_df[c])]
            categorical_cols = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(feat_df[c])]

            for fc in feat_cols:
                feature_group_map[fc] = group_name

            if numeric_cols:
                transformers.append((f"{col}_num", "passthrough", numeric_cols))
                feature_columns.extend(numeric_cols)

            if categorical_cols:
                cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                transformers.append((f"{col}_cat", cat_encoder, categorical_cols))
                feature_columns.extend(categorical_cols)

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
    df = merge_external_features(cfg, df, root=root)
    ct, _, feature_group_map = build_feature_pipeline(cfg, df, root=root)
    ct.fit(df)
    X = ct.transform(df)
    y = df[target].values
    feature_names = get_feature_names(ct)
    groups = df["strain_name"].values if "strain_name" in df.columns else np.arange(len(df))
    return X, y, feature_names, groups, feature_group_map
