"""Evaluation metrics for classification and regression tasks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import pearsonr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def _safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _safe_prauc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def _safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(pearsonr(y_true, y_pred)[0])


_CLF_METRICS: dict[str, Any] = {
    "auroc": lambda yt, yp, yprob: _safe_auroc(yt, yprob),
    "pr_auc": lambda yt, yp, yprob: _safe_prauc(yt, yprob),
    "f1": lambda yt, yp, yprob: float(f1_score(yt, yp, zero_division=0)),
    "mcc": lambda yt, yp, yprob: float(matthews_corrcoef(yt, yp)),
    "balanced_accuracy": lambda yt, yp, yprob: float(balanced_accuracy_score(yt, yp)),
}

_REG_METRICS: dict[str, Any] = {
    "rmse": lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp))),
    "r2": lambda yt, yp: float(r2_score(yt, yp)) if len(yt) > 1 else float("nan"),
    "pearson_r": lambda yt, yp: _safe_pearsonr(yt, yp),
    "mae": lambda yt, yp: float(mean_absolute_error(yt, yp)),
}


# ---------------------------------------------------------------------------
# Per-fold evaluation
# ---------------------------------------------------------------------------

def evaluate_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    task: str,
    requested: list[str] | None = None,
) -> dict[str, float]:
    """Compute metrics for a single fold.

    Parameters
    ----------
    y_true, y_pred:
        Ground truth and predictions.
    y_prob:
        Predicted probabilities (classification only).
    task:
        ``"growth_class"`` for classification, else regression.
    requested:
        Metric names to compute; ``None`` = all available.
    """
    is_clf = task == "growth_class"
    pool = _CLF_METRICS if is_clf else _REG_METRICS

    if requested:
        keys = [k for k in requested if k in pool]
    else:
        keys = list(pool.keys())

    results: dict[str, float] = {}
    for k in keys:
        fn = pool[k]
        try:
            if is_clf:
                prob = y_prob if y_prob is not None else y_pred.astype(float)
                results[k] = fn(y_true, y_pred, prob)
            else:
                results[k] = fn(y_true, y_pred)
        except Exception as exc:
            logger.debug("metric '{}' failed: {}", k, exc)
            results[k] = float("nan")

    return results


# ---------------------------------------------------------------------------
# Aggregate across folds
# ---------------------------------------------------------------------------

def _is_numeric_metric(value: Any) -> bool:
    """True for real floats/ints (incl. NaN); false for str/bool metadata."""
    if isinstance(value, (bool, str)) or value is None:
        return False
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def aggregate_metrics(
    fold_metrics: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute mean +/- std of each metric across folds.

    Per-fold NaNs are dropped before aggregation.
    """
    if not fold_metrics:
        return {}

    keys = [
        k for k, v in fold_metrics[0].items()
        if _is_numeric_metric(v) and k not in {"fold", "n_train", "n_test"}
    ]
    agg: dict[str, dict[str, float]] = {}

    for k in keys:
        vals = []
        for fm in fold_metrics:
            v = fm.get(k, float("nan"))
            if not _is_numeric_metric(v):
                continue
            fv = float(v)
            if not np.isnan(fv):
                vals.append(fv)
        if vals:
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        else:
            agg[k] = {"mean": float("nan"), "std": float("nan")}

    return agg


def per_group_metrics(
    predictions_df: pd.DataFrame,
    group_col: str,
    task: str,
    requested: list[str] | None = None,
) -> pd.DataFrame:
    """Compute metrics broken down by a grouping column (e.g. strain, media).

    Parameters
    ----------
    predictions_df:
        Must have columns ``y_true``, ``y_pred``, and optionally ``y_prob``.
    group_col:
        Column in *predictions_df* to group by.
    task:
        Classification or regression task name.

    Returns
    -------
    DataFrame with one row per group and one column per metric.
    """
    rows: list[dict[str, Any]] = []
    for name, grp in predictions_df.groupby(group_col):
        y_true = grp["y_true"].values
        y_pred = grp["y_pred"].values
        y_prob = grp["y_prob"].values if "y_prob" in grp.columns else None
        m = evaluate_fold(y_true, y_pred, y_prob, task, requested)
        m[group_col] = name
        rows.append(m)

    return pd.DataFrame(rows).set_index(group_col)
