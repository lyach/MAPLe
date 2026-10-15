"""Precompute, save, and consume reusable cross-validation splits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from loguru import logger

from src.modeling.cv import make_splits
from src.modeling.io import read_table, write_table


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_group_fold_map(
    df: pd.DataFrame,
    split_on: str,
    strategies: list[str] | str,
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: int = 42,
    y: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a tidy group->fold map ``[strategy, repeat, fold, <split_on>]``.

    Parameters
    ----------
    df:
        Source dataset.
    split_on:
        Column whose values define the CV groups (map key).
    strategies:
        One or more group-consistent strategies (``grouped`` /
        ``stratified_grouped``, or ``loso`` / ``lomo`` when ``split_on`` is
        ``strain_name`` / ``media_name``).
    n_splits, n_repeats, seed:
        CV parameters. Repeats use ``seed + repeat`` with shuffling so folds
        differ across repeats.
    y:
        Target array, required for ``stratified_grouped``.
    """
    if isinstance(strategies, str):
        strategies = [strategies]
    if split_on not in df.columns:
        raise ValueError(f"split_on column '{split_on}' not in dataframe")

    shuffle = n_repeats > 1
    records: list[dict] = []

    for strategy in strategies:
        strategy = strategy.lower().strip()
        if strategy in {"random", "stratified"}:
            raise ValueError(
                f"'{strategy}' assigns individual rows and cannot be stored as "
                "a group->fold map; use grouped or stratified_grouped."
            )
        for repeat in range(n_repeats):
            group_to_fold: dict = {}
            splits = make_splits(
                strategy,
                df,
                y=y,
                n_splits=n_splits,
                seed=seed + repeat,
                group_on=split_on,
                shuffle=shuffle,
            )
            for fold, (_, test_idx) in enumerate(splits):
                test_groups = pd.unique(df.iloc[test_idx][split_on])
                for g in test_groups:
                    if g in group_to_fold and group_to_fold[g] != fold:
                        raise ValueError(
                            f"strategy '{strategy}' assigned group '{g}' "
                            f"(split_on='{split_on}') to folds "
                            f"{group_to_fold[g]} and {fold}; it is not "
                            "group-consistent for this column."
                        )
                    group_to_fold[g] = fold
            for g, fold in group_to_fold.items():
                records.append(
                    {"strategy": strategy, "repeat": repeat, "fold": fold, split_on: g}
                )
            logger.info(
                "{} (repeat {}): mapped {} '{}' groups across folds",
                strategy, repeat, len(group_to_fold), split_on,
            )

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _is_classification(series: pd.Series) -> bool:
    s = series.dropna()
    if pd.api.types.is_bool_dtype(s):
        return True
    if pd.api.types.is_float_dtype(s):
        return False
    return s.nunique() <= 20


def fold_distribution_stats(
    df: pd.DataFrame,
    split_map: pd.DataFrame,
    split_on: str,
    target: str,
) -> list[dict]:
    """Per (strategy, repeat, fold) train/test distribution stats.

    Reports row/group counts and the target distribution (class counts and
    fractions for classification, or summary stats for regression) on train vs
    test, and asserts train/test group sets are disjoint (no leakage).
    """
    is_clf = _is_classification(df[target])
    all_groups = set(pd.unique(df[split_on]))
    stats: list[dict] = []

    keys = split_map[["strategy", "repeat", "fold"]].drop_duplicates()
    for _, (strategy, repeat, fold) in keys.iterrows():
        test_groups = set(
            split_map[
                (split_map["strategy"] == strategy)
                & (split_map["repeat"] == repeat)
                & (split_map["fold"] == fold)
            ][split_on]
        )
        train_groups = all_groups - test_groups
        assert not (train_groups & test_groups), "train/test group leakage detected"

        test_mask = df[split_on].isin(test_groups).values
        df_test = df[test_mask]
        df_train = df[~test_mask]

        entry = {
            "strategy": strategy,
            "repeat": int(repeat),
            "fold": int(fold),
            "n_train": int(len(df_train)),
            "n_test": int(len(df_test)),
            "n_groups_train": int(len(train_groups)),
            "n_groups_test": int(len(test_groups)),
        }
        if is_clf:
            entry["target_type"] = "classification"
            entry["target_train"] = _class_dist(df_train[target])
            entry["target_test"] = _class_dist(df_test[target])
        else:
            entry["target_type"] = "regression"
            entry["target_train"] = _num_summary(df_train[target])
            entry["target_test"] = _num_summary(df_test[target])
        stats.append(entry)

    return stats


def _class_dist(series: pd.Series) -> dict:
    counts = series.value_counts(dropna=False)
    total = int(counts.sum())
    return {
        str(k): {"n": int(v), "frac": round(v / total, 4) if total else 0.0}
        for k, v in counts.items()
    }


def _num_summary(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {"n": 0}
    return {
        "n": int(s.size),
        "mean": round(float(s.mean()), 4),
        "std": round(float(s.std()), 4),
        "min": round(float(s.min()), 4),
        "max": round(float(s.max()), 4),
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_split_map(split_map: pd.DataFrame, path: str | Path) -> None:
    write_table(split_map, path)


def load_split_map(path: str | Path) -> pd.DataFrame:
    return read_table(path)


def split_stats_path(splits_file: str | Path) -> Path:
    """Return the ``*_stats.json`` sibling of a saved split map."""
    path = Path(splits_file)
    return path.with_name(f"{path.stem}_stats.json")


def load_split_stats(splits_file: str | Path) -> dict:
    """Load the stats JSON written next to a saved split map."""
    path = split_stats_path(splits_file)
    if not path.exists():
        raise FileNotFoundError(
            f"split stats not found: {path}. Re-run create_splits.py so the "
            "map is written with a companion *_stats.json."
        )
    with open(path) as f:
        return json.load(f)


def assert_split_source_length(
    n_rows: int,
    splits_file: str | Path,
    modeling_dataset_path: str | Path | None = None,
) -> None:
    """Require the modeling table and the split-source table to have the same row count.

    Compares ``n_rows`` against ``dataset_shape[0]`` in the split stats JSON.
    """
    stats = load_split_stats(splits_file)
    expected = stats.get("dataset_shape", [None])[0]
    if expected is None:
        raise ValueError(
            f"split stats {split_stats_path(splits_file)} has no dataset_shape[0]"
        )
    if int(n_rows) != int(expected):
        src = stats.get("dataset_path", "<unknown>")
        modeling = modeling_dataset_path or "<modeling_dataset_path>"
        raise ValueError(
            f"modeling dataset has {n_rows} rows ({modeling}) but the "
            f"precomputed splits were built from a table with {expected} rows "
            f"({src}). Point modeling_dataset_path at the same file used to "
            "create the splits, or regenerate the splits from the current table."
        )
    logger.info(
        "split source length matches modeling dataset: {} rows", n_rows
    )


# ---------------------------------------------------------------------------
# Consume
# ---------------------------------------------------------------------------

def iter_saved_splits(
    df: pd.DataFrame,
    split_map: pd.DataFrame,
    split_on: str,
    strategy: str,
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield ``(repeat, fold, train_idx, test_idx)`` from a saved group map.

    Positional indices into ``df`` are derived by matching ``df[split_on]``
    against the map. Raises if the strategy is absent from the map or if any
    row's group value is missing from a repeat's group set (guards against
    silent mismatch/leakage).
    """
    if split_on not in df.columns:
        raise ValueError(f"split_on column '{split_on}' not in dataframe")
    sub = split_map[split_map["strategy"] == strategy]
    if sub.empty:
        available = sorted(split_map["strategy"].unique())
        raise ValueError(
            f"strategy '{strategy}' not found in split map; available: {available}"
        )

    row_groups = df[split_on].values
    for repeat in sorted(sub["repeat"].unique()):
        rep = sub[sub["repeat"] == repeat]
        mapped_groups = set(rep[split_on])
        present = set(pd.unique(row_groups))
        missing = present - mapped_groups
        if missing:
            raise ValueError(
                f"{len(missing)} '{split_on}' value(s) in the dataframe are absent "
                f"from the saved split map (repeat {repeat}), e.g. "
                f"{sorted(map(str, missing))[:5]}. The split map is stale or was "
                "built from a different dataset."
            )
        for fold in sorted(rep["fold"].unique()):
            test_groups = set(rep[rep["fold"] == fold][split_on])
            test_mask = np.isin(row_groups, list(test_groups))
            test_idx = np.flatnonzero(test_mask)
            train_idx = np.flatnonzero(~test_mask)
            yield int(repeat), int(fold), train_idx, test_idx


def held_out_label(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    split_on: str,
    max_groups: int = 3,
) -> str:
    """Readable label for a held-out fold based on its group values."""
    groups = pd.unique(df.iloc[test_idx][split_on])
    if len(groups) == 1:
        return str(groups[0])
    head = ", ".join(map(str, groups[:max_groups]))
    if len(groups) > max_groups:
        head += f", +{len(groups) - max_groups} more"
    return head
