"""SHAP model interpretability and grouped feature importance."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from loguru import logger
from sklearn.base import BaseEstimator

ShapClass = Literal["positive", "negative"]


def _resolve_output_idx(
    shap_class: ShapClass | int | None,
    n_outputs: int,
) -> int:
    """Map ``shap_class`` / index to a concrete output axis index."""
    if isinstance(shap_class, int):
        if not 0 <= shap_class < n_outputs:
            raise ValueError(
                f"shap_class index {shap_class} out of range for {n_outputs} outputs"
            )
        return shap_class

    if shap_class is None:
        # Default: positive class for binary, else last output.
        return 1 if n_outputs == 2 else n_outputs - 1

    if shap_class == "positive":
        return 1 if n_outputs == 2 else n_outputs - 1
    if shap_class == "negative":
        return 0

    raise ValueError(f"Unknown shap_class: {shap_class!r}")


def _select_output(
    shap_values: shap.Explanation,
    shap_class: ShapClass | int | None = "positive",
) -> shap.Explanation:
    """Collapse multi-output SHAP to a single output for beeswarm/bar plots.

    Parameters
    ----------
    shap_class:
        ``"positive"`` / ``"negative"`` for binary classifiers, or an explicit
        class index. Ignored when SHAP is already 2-D (e.g. regression).
    """
    vals = getattr(shap_values, "values", None)
    if vals is None or getattr(vals, "ndim", 0) != 3:
        return shap_values

    n_outputs = vals.shape[-1]
    output_idx = _resolve_output_idx(shap_class, n_outputs)

    logger.info(
        "multi-output SHAP (shape {}); using {} (output index {})",
        vals.shape,
        shap_class,
        output_idx,
    )
    return shap_values[:, :, output_idx]


def compute_shap_values(
    model: BaseEstimator,
    X: np.ndarray,
    feature_names: list[str],
) -> shap.Explanation:
    """Compute SHAP values using the most appropriate explainer.

    - Tree-based models use ``TreeExplainer``.
    - Linear models use ``LinearExplainer``.
    - Everything else uses ``KernelExplainer`` on a subsample.
    """
    model_type = type(model).__name__.lower()

    if any(t in model_type for t in ("xgb", "randomforest", "gradient")):
        explainer = shap.TreeExplainer(model)
        sv = explainer(X)
    elif any(t in model_type for t in ("logistic", "ridge", "linear")):
        explainer = shap.LinearExplainer(model, X)
        sv = explainer(X)
    else:
        bg = shap.sample(X, min(100, len(X)))
        explainer = shap.KernelExplainer(model.predict, bg)
        sv = explainer(X)

    sv.feature_names = feature_names
    return sv


def plot_shap_summary(
    shap_values: shap.Explanation,
    output_dir: Path,
    max_display: int = 20,
    shap_class: ShapClass | int | None = "positive",
) -> None:
    """Save beeswarm and bar SHAP summary plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    shap_values = _select_output(shap_values, shap_class=shap_class)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    path = output_dir / "shap_beeswarm.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("saved {}", path)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.tight_layout()
    path = output_dir / "shap_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("saved {}", path)


def compute_group_importance(
    shap_values: shap.Explanation,
    group_index: dict[str, list[int]],
    shap_class: ShapClass | int | None = "positive",
) -> dict[str, float]:
    """Aggregate mean |SHAP| by feature group (strain, media, condition, ...).

    Parameters
    ----------
    shap_values:
        SHAP Explanation object.
    group_index:
        Mapping from group name to list of feature indices, as returned
        by ``features.build_feature_group_index``.
    shap_class:
        Which class output to use for multi-output explanations.

    Returns
    -------
    Dict mapping group name to mean absolute SHAP value.
    """
    shap_values = _select_output(shap_values, shap_class=shap_class)
    vals = np.abs(shap_values.values)

    importance: dict[str, float] = {}
    for group, indices in group_index.items():
        importance[group] = float(np.mean(vals[:, indices]))

    return importance


def plot_group_importance(
    group_importance: dict[str, float],
    output_dir: Path,
) -> None:
    """Bar chart of mean |SHAP| per feature group."""
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = sorted(group_importance, key=group_importance.get, reverse=True)
    values = [group_importance[g] for g in groups]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(groups, values)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Group Importance")
    ax.invert_yaxis()

    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    path = output_dir / "group_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("saved {}", path)


def run_interpretation(
    model: BaseEstimator,
    X: np.ndarray,
    feature_names: list[str],
    group_index: dict[str, list[int]],
    output_dir: Path,
    do_group_importance: bool = True,
    shap_class: ShapClass | int | None = "positive",
) -> dict[str, float]:
    """Full interpretability pipeline: SHAP values + plots + group importance.

    Returns the group importance dict.
    """
    logger.info("computing SHAP values ({} samples, {} features)", *X.shape)
    sv = compute_shap_values(model, X, feature_names)

    fig_dir = output_dir / "figures"
    plot_shap_summary(sv, fig_dir, shap_class=shap_class)

    group_imp: dict[str, float] = {}
    if do_group_importance and group_index:
        group_imp = compute_group_importance(sv, group_index, shap_class=shap_class)
        plot_group_importance(group_imp, fig_dir)
        logger.info("group importance: {}", group_imp)

    return group_imp
