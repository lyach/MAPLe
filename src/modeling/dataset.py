"""Build the modeling dataset by merging curves CSV with strain metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.modeling.config import DatasetConfig

# Columns that identify a unique condition for replicate averaging
_GROUP_COLS = ["strain_name", "media_name", "atmosphere", "temperature"]

# Continuous growth parameters that get averaged across replicates
_CONTINUOUS_TARGETS = ["max_od", "auc", "growth_rate", "lag_time_h", "time_to_max_od_h"]

# Columns dropped after replicate averaging (per-well identifiers)
_PER_WELL_ID_COLS = ["exp_id", "date", "replicate"]


def _normalize_key(s: pd.Series) -> pd.Series:
    """Lowercase, strip whitespace for join-key alignment."""
    return s.astype(str).str.strip().str.lower()


def _resolve_curves_paths(cfg: DatasetConfig, root: Path) -> list[Path]:
    """Return absolute paths for one or more curves CSV files."""
    paths = cfg.curves_file if isinstance(cfg.curves_file, list) else [cfg.curves_file]
    return [root / p for p in paths]


def _load_curves(curves_paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate one or more curves CSV files."""
    dfs = [pd.read_csv(path) for path in curves_paths]
    if len(dfs) == 1:
        logger.info("loaded 1 curves file, shape: {} rows x {} cols", *dfs[0].shape)
        return dfs[0]

    df = pd.concat(dfs, ignore_index=True)
    logger.info(
        "loaded {} curves files, total shape: {} rows x {} cols",
        len(dfs),
        *df.shape,
    )
    return df


def build_modeling_dataset(
    cfg: DatasetConfig,
    root: Path | None = None,
) -> pd.DataFrame:
    """Load curves + optional strain metadata, merge, and optionally average replicates.

    Parameters
    ----------
    cfg:
        Parsed ``DatasetConfig`` from the dataset YAML.
    root:
        Project root used to resolve relative paths in *cfg*.
        Defaults to the current working directory.

    Returns
    -------
    pd.DataFrame
        Tidy table with identifier columns, raw feature columns, and targets.
    """
    if root is None:
        root = Path.cwd()

    curves_paths = _resolve_curves_paths(cfg, root)
    logger.info("loading curves: {}", curves_paths)
    df = _load_curves(curves_paths)
    
    # remove water controls (only when a media_name column is present)
    if "media_name" in df.columns:
        df = df.loc[df["media_name"] != "water"]
        logger.info("removed water controls, shape: {} rows x {} cols", *df.shape)

    if "strain_name" in df.columns:
        # remove `other` strains
        df = df.loc[~df["strain_name"].str.contains("other")]
        logger.info("removed `other` strains, shape: {} rows x {} cols", *df.shape)

        # remove `empty` strains
        df = df.loc[df["strain_name"] != "empty"]
        logger.info("removed `empty` strains, shape: {} rows x {} cols", *df.shape)

    # merge strain metadata
    if cfg.strain_metadata_file:
        meta_path = root / cfg.strain_metadata_file
        logger.info("loading strain metadata: {}", meta_path)
        meta = pd.read_csv(meta_path)
        logger.info("strain metadata shape: {} rows x {} cols", *meta.shape)

        left_key = cfg.merge.left
        right_key = cfg.merge.right

        df["_join_key"] = _normalize_key(df[left_key])
        meta["_join_key"] = _normalize_key(meta[right_key])

        n_before = len(df)
        df = df.merge(
            meta.drop(columns=[right_key]),
            on="_join_key",
            how="left",
        )
        df.drop(columns=["_join_key"], inplace=True)

        n_unmatched = df.filter(like="_join_key").isna().any(axis=1).sum() if False else 0
        meta_cols = [c for c in meta.columns if c not in {"_join_key", right_key}]
        n_unmatched = df[meta_cols[0]].isna().sum() if meta_cols else 0
        if n_unmatched:
            unmatched_strains = df.loc[df[meta_cols[0]].isna(), left_key].unique()
            logger.warning(
                "{} rows have no metadata match — strains: {}",
                n_unmatched,
                list(unmatched_strains),
            )
        logger.info("merged shape: {} rows x {} cols", *df.shape)

    # replicate averaging
    if cfg.grain == "averaged":
        logger.info("averaging replicates (group by {})", _GROUP_COLS)
        group_cols = [c for c in _GROUP_COLS if c in df.columns]

        agg: dict[str, str | tuple[str, ...]] = {}
        for col in _CONTINUOUS_TARGETS:
            if col in df.columns:
                agg[col] = "mean"
        if "growth_class" in df.columns:
            agg["growth_class"] = "mean"

        df = df.groupby(group_cols, as_index=False).agg(agg)

        if "growth_class" in df.columns:
            df["growth_class"] = (df["growth_class"] >= 0.5).astype(int)

        logger.info("averaged shape: {} rows x {} cols", *df.shape)
    else:
        # keep per-well identifiers
        logger.info("keeping per-well granularity ({} rows)", len(df))

    return df
