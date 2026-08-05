"""Cross-validation splitters.

Strategies
----------
- ``loso``  : Leave-One-Strain-Out
- ``lomo``  : Leave-One-Media-Out
- ``grouped`` : GroupKFold grouped by ``group_on`` column
- ``stratified_grouped`` : StratifiedGroupKFold (stratified by target, grouped
  by ``group_on`` column)
- ``random`` : Shuffled KFold with no grouping (rows split randomly).
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
)


def _encode_groups(series: pd.Series) -> np.ndarray:
    """Map string group labels to integer codes."""
    return series.astype("category").cat.codes.values


def make_splits(
    strategy: str,
    df: pd.DataFrame,
    y: np.ndarray | None = None,
    n_splits: int = 5,
    seed: int = 42,
    group_on: str = "strain_name",
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) arrays for the chosen CV strategy.

    Parameters
    ----------
    strategy:
        One of ``"loso"``, ``"lomo"``, ``"grouped"``,
        ``"stratified_grouped"``, ``"random"``.
    df:
        Modeling DataFrame (needs ``strain_name`` and/or ``media_name``,
        or the column named by ``group_on`` for grouped strategies).
    y:
        Target array, required for ``stratified_grouped``.
    n_splits:
        Number of folds for ``grouped`` / ``stratified_grouped`` / ``random``.
    seed:
        Random seed for shuffled strategies (``random``). Ignored otherwise.
    group_on:
        Column used as groups for ``grouped`` / ``stratified_grouped``.
    """
    strategy = strategy.lower().strip()

    if strategy == "loso":
        groups = _encode_groups(df["strain_name"])
        splitter = LeaveOneGroupOut()
        n = len(np.unique(groups))
        logger.info("LOSO: {} folds (one per strain)", n)
        yield from splitter.split(df, groups=groups)

    elif strategy == "lomo":
        groups = _encode_groups(df["media_name"])
        splitter = LeaveOneGroupOut()
        n = len(np.unique(groups))
        logger.info("LOMO: {} folds (one per medium)", n)
        yield from splitter.split(df, groups=groups)

    elif strategy == "grouped":
        if group_on not in df.columns:
            raise ValueError(
                f"grouped requires column '{group_on}' (cv.group_on) in DataFrame"
            )
        groups = _encode_groups(df[group_on])
        n_splits = min(n_splits, len(np.unique(groups)))
        splitter = GroupKFold(n_splits=n_splits)
        logger.info("GroupKFold: {} folds (grouped by {})", n_splits, group_on)
        yield from splitter.split(df, groups=groups)

    elif strategy == "stratified_grouped":
        if y is None:
            raise ValueError("stratified_grouped requires y (target array)")
        if group_on not in df.columns:
            raise ValueError(
                f"stratified_grouped requires column '{group_on}' "
                "(cv.group_on) in DataFrame"
            )
        groups = _encode_groups(df[group_on])
        n_splits = min(n_splits, len(np.unique(groups)))
        splitter = StratifiedGroupKFold(n_splits=n_splits)
        logger.info(
            "StratifiedGroupKFold: {} folds (grouped by {})", n_splits, group_on
        )
        yield from splitter.split(df, y, groups=groups)

    elif strategy == "random":
        n_splits = min(n_splits, len(df))
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        logger.info(
            "KFold (random): {} folds, shuffle=True, seed={}",
            n_splits, seed,
        )
        yield from splitter.split(df)

    else:
        raise ValueError(
            f"Unknown CV strategy '{strategy}'. "
            "Choose from: loso, lomo, grouped, stratified_grouped, random"
        )


def get_fold_group_label(
    strategy: str,
    df: pd.DataFrame,
    test_idx: np.ndarray,
) -> str:
    """Return a readable label for the held-out fold."""
    if strategy == "loso":
        return df.iloc[test_idx]["strain_name"].iloc[0]
    elif strategy == "lomo":
        return df.iloc[test_idx]["media_name"].iloc[0]
    else:
        return f"fold_{len(test_idx)}_samples"
