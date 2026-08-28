"""Utility functions for plotting modeling results"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
import seaborn as sns

def _read_dataframe_from_csv(path: str) -> pd.DataFrame:
    """Read dataframe from CSV file"""
    return pd.read_csv(path)


def _as_dataframe(df_or_path: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Coerce a DataFrame-or-CSV-path input into a DataFrame."""
    if isinstance(df_or_path, pd.DataFrame):
        return df_or_path
    return _read_dataframe_from_csv(str(df_or_path))


def _metrics_block(data: dict) -> dict[str, dict[str, float]]:
    """Extract ``metric_name -> {mean, std}`` from a loaded metrics JSON."""

    def is_metric_map(d: dict) -> bool:
        return bool(d) and all(
            isinstance(v, dict) and "mean" in v and "std" in v for v in d.values()
        )

    if is_metric_map(data):
        return data

    for key in sorted(data):
        val = data[key]
        if isinstance(val, dict) and is_metric_map(val):
            return val

    raise ValueError("Could not parse metrics JSON structure")


def _load_metrics_blocks(
    jsons: list[dict | str | Path],
) -> list[dict[str, dict[str, float]]]:
    """Load mean/std metric blocks from dicts or JSON paths."""
    blocks: list[dict[str, dict[str, float]]] = []
    for j in jsons:
        data = _read_metrics_from_json(j)
        blocks.append(_metrics_block(data))
    return blocks


def _read_metrics_from_json(json_or_path: dict | str | Path) -> dict:
    """Read metrics from a JSON object or file path."""
    if isinstance(json_or_path, dict):
        return json_or_path
    with open(json_or_path) as f:
        return json.load(f)


def _color_cycle(n: int, palette: str | None = None) -> list:
    """Return *n* distinct colours from a seaborn palette or the default colour cycle."""
    if palette is not None:
        return list(sns.color_palette(palette, n))
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return [prop_cycle[i % len(prop_cycle)] for i in range(n)]


def _panel_subfigure(
    n_panels: int,
    *,
    col_wrap: int = 2,
    figsize: tuple[float, float] | None = None,
    panel_w: float = 4.5,
    panel_h: float = 4.0,
    min_w: float = 5.0,
    min_h: float = 4.0,
) -> tuple[Figure, np.ndarray, int, int]:
    """Create a wrapped grid of subplots (at most ``col_wrap`` columns)."""
    n_cols = min(col_wrap, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    default_w = max(panel_w * n_cols, min_w)
    default_h = max(panel_h * n_rows, min_h)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize or (default_w, default_h),
        squeeze=False,
    )
    return fig, axes, n_rows, n_cols


def _hide_unused_panel_axes(
    axes: np.ndarray,
    n_panels: int,
    n_rows: int,
    n_cols: int,
) -> None:
    for idx in range(n_panels, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)


def _add_shared_side_legend(
    fig: Figure,
    legend_handles: dict,
    *,
    title: str | None = None,
    fig_title: str | None = None,
    axes_right: float = 0.72,
) -> None:
    """Reserve space on the right and draw one figure-level legend."""
    layout_top = 0.90 if fig_title else 0.96
    layout_rect = (0, 0, axes_right if legend_handles else 1.0, layout_top)
    fig.tight_layout(rect=layout_rect)
    if legend_handles:
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            title=title,
            loc="center left",
            bbox_to_anchor=(axes_right + 0.02, 0.5),
            frameon=True,
            borderaxespad=0.5,
        )
    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=12, fontweight="bold", y=layout_top + 0.04)


def plot_scatter_predictions(
    dfs: list[pd.DataFrame | str | Path],
    color_by: str,
    *,
    y_pred_col: str = "y_pred",
    y_true_col: str = "y_true",
    titles: list[str] | None = None,
    col_wrap: int = 2,
    figsize: tuple[float, float] | None = None,
    palette: str | None = None,
    fig_title: str | None = None,
) -> Figure:
    """Plot predicted-vs-true scatter panels for one or more prediction sets.

    Parameters
    ----------
    dfs:
        List of dataframes and/or CSV paths.
    color_by:
        Column name to color points by (e.g. ``"fold_label"`` or
        ``"atmosphere"``).
    y_pred_col, y_true_col:
        Column names holding predicted and true values, respectively.
    titles:
        Optional per-panel titles. Defaults to the CSV filename stem for
        path inputs, else ``"Panel i"``.
    col_wrap:
        Maximum number of panel columns before wrapping to a new row.
    figsize:
        Overall figure size; auto-computed from the grid shape if omitted.
    palette:
        Seaborn palette name. Falls back to the default matplotlib colour cycle when ``None``.
    fig_title:
        Optional title for the whole figure (``fig.suptitle``).

    Returns
    -------
    Figure
        The assembled matplotlib figure.
    """
    n = len(dfs)
    if n == 0:
        raise ValueError("`dfs` must contain at least one dataframe or CSV path")

    dataframes = [_as_dataframe(d) for d in dfs]

    if titles is None:
        titles = [
            Path(d).stem if isinstance(d, (str, Path)) else f"Panel {i + 1}"
            for i, d in enumerate(dfs)
        ]

    n_cols = min(col_wrap, n)
    fig, axes, n_rows, n_cols = _panel_subfigure(n, col_wrap=col_wrap, figsize=figsize)

    # Shared color mapping, pooled across all panels, for a single legend.
    color_vals = sorted(
        pd.concat([df[color_by] for df in dataframes], ignore_index=True)
        .dropna()
        .unique()
    )
    colors = {v: c for v, c in zip(color_vals, _color_cycle(len(color_vals), palette=palette))}

    legend_handles: dict = {}

    for idx, (df, title) in enumerate(zip(dataframes, titles)):
        ax = axes[idx // n_cols][idx % n_cols]

        for cval, grp in df.groupby(color_by, sort=True):
            scatter = ax.scatter(
                grp[y_pred_col],
                grp[y_true_col],
                color=colors[cval],
                s=35,
                alpha=0.75,
                edgecolors="white",
                linewidths=0.4,
            )
            if cval not in legend_handles:
                legend_handles[cval] = scatter

        # y = x reference line
        lo = min(df[y_pred_col].min(), df[y_true_col].min())
        hi = max(df[y_pred_col].max(), df[y_true_col].max())
        ax.plot([lo, hi], [lo, hi], color="grey", lw=1.0, ls="--", alpha=0.7, zorder=0)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel(y_pred_col)
        ax.set_ylabel(y_true_col)

    _hide_unused_panel_axes(axes, n, n_rows, n_cols)
    _add_shared_side_legend(
        fig, legend_handles, title=color_by, fig_title=fig_title
    )

    return fig


def plot_metrics_comparison(
    jsons: list[dict | str | Path],
    metrics: list[str],
    *,
    metric_titles: list[str] | None = None,
    json_titles: list[str] | None = None,
    legend_title: str | None = "Model",
    col_wrap: int = 2,
    figsize: tuple[float, float] | None = None,
    palette: str = "Set1",
    fig_title: str | None = None,
) -> Figure:
    """Compare aggregated metrics across experiments as bar panels.

    Each panel shows one metric (mean +/- std error bars) with one bar per
    experiment.

    Parameters
    ----------
    jsons:
        List of metrics dicts and/or paths to ``metrics.json`` files.
    metrics:
        Metric names to plot (e.g. ``["rmse", "r2", "pearson_r", "mae"]``).
    metric_titles:
        Optional panel titles; defaults to metric names.
    json_titles:
        Bar labels and legend entries; defaults to the parent directory
        name for path inputs, else ``"Run i"``.
    legend_title:
        Title for the shared figure legend; ``None`` omits the title.
    col_wrap:
        Maximum number of panel columns before wrapping to a new row.
    figsize:
        Overall figure size; auto-computed from the grid shape if omitted.
    palette:
        Seaborn palette name. Defaults to ``"Set1"``.
    fig_title:
        Optional title for the whole figure (``fig.suptitle``).

    Returns
    -------
    Figure
        The assembled matplotlib figure.
    """
    if not jsons:
        raise ValueError("`jsons` must contain at least one metrics dict or path")
    if not metrics:
        raise ValueError("`metrics` must contain at least one metric name")

    blocks = _load_metrics_blocks(jsons)

    if json_titles is None:
        json_titles = [
            Path(j).parent.name
            if isinstance(j, (str, Path))
            else f"Run {i + 1}"
            for i, j in enumerate(jsons)
        ]

    if metric_titles is None:
        metric_titles = list(metrics)
    if len(metric_titles) != len(metrics):
        raise ValueError("`metric_titles` must match the length of `metrics`")

    n_metrics = len(metrics)
    fig, axes, n_rows, n_cols = _panel_subfigure(
        n_metrics, col_wrap=col_wrap, figsize=figsize
    )

    bar_colors = _color_cycle(len(jsons), palette=palette)
    x = np.arange(len(jsons))
    legend_handles: dict = {}

    for idx, (metric, panel_title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[idx // n_cols][idx % n_cols]

        means = []
        stds = []
        for block in blocks:
            if metric not in block:
                raise KeyError(f"metric '{metric}' not found in metrics JSON")
            means.append(float(block[metric]["mean"]))
            stds.append(float(block[metric]["std"]))

        for i, (label, mean, std, color) in enumerate(
            zip(json_titles, means, stds, bar_colors)
        ):
            bar = ax.bar(
                x[i],
                mean,
                yerr=std,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                capsize=3,
                width=0.7,
                error_kw={"elinewidth": 1.0, "capthick": 1.0},
            )
            if label not in legend_handles:
                legend_handles[label] = bar

        ax.set_title(panel_title, fontsize=10)
        ax.tick_params(axis="x", labelbottom=False)
        ax.axhline(0, color="grey", lw=0.6, alpha=0.5, zorder=0)

    _hide_unused_panel_axes(axes, n_metrics, n_rows, n_cols)
    _add_shared_side_legend(
        fig, legend_handles, title=legend_title, fig_title=fig_title
    )

    return fig


# ---------------------------------------------------------------------------
# Classification (imbalanced) diagnostic plots
# ---------------------------------------------------------------------------


def plot_roc_comparison(
    dfs: list[pd.DataFrame | str | Path],
    *,
    y_true_col: str = "y_true",
    y_prob_col: str = "y_prob",
    titles: list[str] | None = None,
    palette: str = "Set1",
    figsize: tuple[float, float] = (6.5, 5.5),
    fig_title: str | None = None,
) -> Figure:
    """Overlay ROC curves from one or more prediction sets.

    One ROC curve per [``dfs``] entry, with the AUROC value shown in the legend.

    Parameters
    ----------
    dfs:
        List of DataFrames and/or CSV paths with columns for true labels
        and predicted probabilities.
    y_true_col, y_prob_col:
        Column names for ground-truth binary labels and P(growth).
    titles:
        Legend labels; defaults to the CSV filename stem or ``"Run i"``.
    palette:
        Seaborn palette name.
    figsize:
        Overall figure size.
    fig_title:
        Optional suptitle.

    Returns
    -------
    Figure
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    if not dfs:
        raise ValueError("`dfs` must contain at least one DataFrame or path")

    dataframes = [_as_dataframe(d) for d in dfs]

    if titles is None:
        titles = [
            Path(d).stem if isinstance(d, (str, Path)) else f"Run {i + 1}"
            for i, d in enumerate(dfs)
        ]

    colors = _color_cycle(len(dataframes), palette=palette)
    fig, ax = plt.subplots(figsize=figsize)

    for df, label, color in zip(dataframes, titles, colors):
        y_true = df[y_true_col].values
        y_prob = df[y_prob_col].values
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{label} (AUROC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="grey", alpha=0.6)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.set_aspect("equal")
    fig.tight_layout(rect=(0, 0, 1.0, 0.94 if fig_title else 1.0))
    if fig_title:
        fig.suptitle(fig_title, fontsize=12, fontweight="bold")
    return fig


def plot_pr_curve_comparison(
    dfs: list[pd.DataFrame | str | Path],
    *,
    pos_label: int = 0,
    y_true_col: str = "y_true",
    y_prob_col: str = "y_prob",
    titles: list[str] | None = None,
    palette: str = "Set1",
    figsize: tuple[float, float] = (6.5, 5.5),
    fig_title: str | None = None,
) -> Figure:
    """Overlay Precision-Recall curves from one or more prediction sets.

    Parameters
    ----------
    dfs:
        List of DataFrames and/or CSV paths.
    pos_label:
        Which label to treat as "positive" for Precision/Recall.
        Default ``0`` (no-growth / minority class).
    y_true_col, y_prob_col:
        Column names for ground-truth labels and P(growth).
    titles:
        Legend labels.
    palette:
        Seaborn palette name.
    figsize:
        Overall figure size.
    fig_title:
        Optional suptitle.

    Returns
    -------
    Figure
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    if not dfs:
        raise ValueError("`dfs` must contain at least one DataFrame or path")

    dataframes = [_as_dataframe(d) for d in dfs]

    if titles is None:
        titles = [
            Path(d).stem if isinstance(d, (str, Path)) else f"Run {i + 1}"
            for i, d in enumerate(dfs)
        ]

    colors = _color_cycle(len(dataframes), palette=palette)
    fig, ax = plt.subplots(figsize=figsize)

    for df, label, color in zip(dataframes, titles, colors):
        y_true = df[y_true_col].values
        y_prob = df[y_prob_col].values
        if len(np.unique(y_true)) < 2:
            continue

        if pos_label == 0:
            y_binary = (y_true == 0).astype(int)
            scores = 1.0 - y_prob
        else:
            y_binary = y_true
            scores = y_prob

        precision, recall, _ = precision_recall_curve(y_binary, scores)
        ap = average_precision_score(y_binary, scores)
        ax.plot(recall, precision, color=color, lw=1.8, label=f"{label} (AP={ap:.3f})")

    prevalence = None
    if dataframes:
        pooled_true = np.concatenate([df[y_true_col].values for df in dataframes])
        prevalence = (pooled_true == pos_label).mean()
    if prevalence is not None:
        ax.axhline(prevalence, ls="--", lw=0.8, color="grey", alpha=0.6, label=f"random ({prevalence:.3f})")

    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    fig.tight_layout(rect=(0, 0, 1.0, 0.94 if fig_title else 1.0))
    if fig_title:
        fig.suptitle(fig_title, fontsize=12, fontweight="bold")
    return fig


def plot_confusion_matrix(
    dfs: list[pd.DataFrame | str | Path],
    *,
    threshold: float | list[float] | None = None,
    y_true_col: str = "y_true",
    y_prob_col: str = "y_prob",
    y_pred_col: str = "y_pred",
    titles: list[str] | None = None,
    class_names: list[str] | None = None,
    col_wrap: int = 3,
    figsize: tuple[float, float] | None = None,
    normalize: str | None = "true",
    cmap: str = "Blues",
    fig_title: str | None = None,
) -> Figure:
    """Heatmap confusion matrices for one or more prediction sets.

    If ``threshold`` is provided and ``y_prob_col`` exists, hard labels are
    derived as ``(y_prob >= threshold)``; otherwise uses ``y_pred_col``.

    Parameters
    ----------
    dfs:
        List of DataFrames and/or CSV paths.
    threshold:
        Decision threshold(s). A single float applies to all; a list
        provides per-panel thresholds. ``None`` uses existing ``y_pred_col``.
    y_true_col, y_prob_col, y_pred_col:
        Column names.
    titles:
        Per-panel titles.
    class_names:
        Axis tick labels (e.g. ``["no-growth", "growth"]``).
    col_wrap:
        Max columns before wrapping.
    figsize:
        Overall figure size.
    normalize:
        ``"true"`` normalizes by row (true class), ``"pred"`` by column,
        ``"all"`` by total, ``None`` for raw counts.
    cmap:
        Matplotlib colormap name.
    fig_title:
        Optional suptitle.

    Returns
    -------
    Figure
    """
    from sklearn.metrics import confusion_matrix as _sklearn_cm

    if not dfs:
        raise ValueError("`dfs` must contain at least one DataFrame or path")

    dataframes = [_as_dataframe(d) for d in dfs]
    n = len(dataframes)

    if titles is None:
        titles = [
            Path(d).stem if isinstance(d, (str, Path)) else f"Panel {i + 1}"
            for i, d in enumerate(dfs)
        ]
    if class_names is None:
        class_names = ["no-growth", "growth"]

    thresholds: list[float | None]
    if threshold is None:
        thresholds = [None] * n
    elif isinstance(threshold, (int, float)):
        thresholds = [float(threshold)] * n
    else:
        thresholds = list(threshold)

    fig, axes, n_rows, n_cols = _panel_subfigure(
        n, col_wrap=col_wrap, figsize=figsize, panel_w=4.0, panel_h=3.8
    )

    for idx, (df, title, thr) in enumerate(zip(dataframes, titles, thresholds)):
        ax = axes[idx // n_cols][idx % n_cols]

        y_true = df[y_true_col].values
        if thr is not None and y_prob_col in df.columns:
            y_pred = (df[y_prob_col].values >= thr).astype(int)
            title = f"{title} (thr={thr:.3f})"
        else:
            y_pred = df[y_pred_col].values

        cm = _sklearn_cm(y_true, y_pred, labels=[0, 1])
        if normalize == "true":
            cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            cm_display = cm.astype(float) / cm.sum(axis=0, keepdims=True)
        elif normalize == "all":
            cm_display = cm.astype(float) / cm.sum()
        else:
            cm_display = cm.astype(float)

        im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap, vmin=0)

        for i in range(2):
            for j in range(2):
                val = cm_display[i, j]
                raw = cm[i, j]
                text = f"{val:.2f}\n({raw})" if normalize else f"{raw}"
                ax.text(j, i, text, ha="center", va="center", fontsize=9,
                        color="white" if val > cm_display.max() / 2 else "black")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        ax.set_title(title, fontsize=10)

    _hide_unused_panel_axes(axes, n, n_rows, n_cols)
    fig.tight_layout(rect=(0, 0, 1.0, 0.93 if fig_title else 1.0))
    if fig_title:
        fig.suptitle(fig_title, fontsize=12, fontweight="bold")
    return fig


def plot_per_group_metric(
    dfs: list[pd.DataFrame | str | Path],
    metric: str = "recall_nogrowth",
    *,
    group_col: str | None = None,
    min_neg: int = 2,
    top_n: int | None = 30,
    sort_by: str = "metric",
    titles: list[str] | None = None,
    col_wrap: int = 2,
    figsize: tuple[float, float] | None = None,
    palette: str = "Set1",
    fig_title: str | None = None,
) -> Figure:
    """Horizontal bar chart of a metric per group (strain/media).

    Takes one or more ``metrics_by_*_withneg.csv`` tables (or DataFrames)
    produced by the experiment pipeline.  Groups are filtered to those with
    at least ``min_neg`` no-growth samples to avoid noisy single-sample bars.

    Parameters
    ----------
    dfs:
        List of per-group metric DataFrames or CSV paths.  The index (or
        first column) is the group name; columns include the ``metric`` and
        ``n_neg``.
    metric:
        Column name to plot (e.g. ``"recall_nogrowth"``, ``"mcc"``).
    group_col:
        Column containing group names.  If ``None``, uses the DataFrame
        index.
    min_neg:
        Minimum ``n_neg`` for a group to appear.
    top_n:
        Show only the top N groups (by ``sort_by``); ``None`` shows all.
    sort_by:
        ``"metric"`` sorts descending by the metric value; ``"n_neg"``
        sorts descending by sample count.
    titles:
        Per-panel titles.
    col_wrap:
        Max columns before wrapping.
    figsize:
        Overall figure size.
    palette:
        Seaborn palette name.
    fig_title:
        Optional suptitle.

    Returns
    -------
    Figure
    """
    if not dfs:
        raise ValueError("`dfs` must contain at least one DataFrame or path")

    dataframes = [_as_dataframe(d) for d in dfs]

    if titles is None:
        titles = [
            Path(d).stem if isinstance(d, (str, Path)) else f"Panel {i + 1}"
            for i, d in enumerate(dfs)
        ]

    n = len(dataframes)
    fig, axes, n_rows, n_cols = _panel_subfigure(
        n, col_wrap=col_wrap, figsize=figsize, panel_w=6.0, panel_h=max(5.0, 0.35 * (top_n or 30))
    )

    colors = _color_cycle(n, palette=palette)

    for idx, (df, title, color) in enumerate(zip(dataframes, titles, colors)):
        ax = axes[idx // n_cols][idx % n_cols]

        if group_col and group_col in df.columns:
            df = df.set_index(group_col)

        if "n_neg" in df.columns:
            df = df[df["n_neg"] >= min_neg]

        if metric not in df.columns:
            ax.set_title(f"{title}\n('{metric}' not found)", fontsize=10)
            continue

        df = df.dropna(subset=[metric])

        if sort_by == "n_neg" and "n_neg" in df.columns:
            df = df.sort_values("n_neg", ascending=True)
        else:
            df = df.sort_values(metric, ascending=True)

        if top_n is not None:
            df = df.tail(top_n)

        ax.barh(
            range(len(df)), df[metric].values,
            color=color, edgecolor="white", linewidth=0.4, height=0.75,
        )
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df.index, fontsize=8)
        ax.set_xlabel(metric, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.axvline(0, color="grey", lw=0.5, alpha=0.5)

        if "n_neg" in df.columns:
            for i, (val, neg) in enumerate(zip(df[metric].values, df["n_neg"].values)):
                ax.text(
                    val + 0.01, i, f"n={int(neg)}", va="center", fontsize=7, color="0.4"
                )

    _hide_unused_panel_axes(axes, n, n_rows, n_cols)
    fig.tight_layout(rect=(0, 0, 1.0, 0.94 if fig_title else 1.0))
    if fig_title:
        fig.suptitle(fig_title, fontsize=12, fontweight="bold")
    return fig


def plot_threshold_sweep(
    dfs: list[pd.DataFrame | str | Path],
    *,
    metrics_to_plot: list[str] | None = None,
    y_true_col: str = "y_true",
    y_prob_col: str = "y_prob",
    n_thresholds: int = 100,
    titles: list[str] | None = None,
    col_wrap: int = 2,
    figsize: tuple[float, float] | None = None,
    fig_title: str | None = None,
) -> Figure:
    """Plot classification metrics as a function of decision threshold.

    For each prediction set, sweeps thresholds from 0 to 1 and plots how
    MCC, balanced accuracy, recall (no-growth), and precision (no-growth)
    vary. Useful for choosing an operating point.

    Parameters
    ----------
    dfs:
        List of DataFrames and/or CSV paths.
    metrics_to_plot:
        Which metrics to show as curves.  Defaults to
        ``["mcc", "balanced_accuracy", "recall_nogrowth", "precision_nogrowth"]``.
    y_true_col, y_prob_col:
        Column names.
    n_thresholds:
        Number of evenly spaced thresholds to evaluate.
    titles:
        Per-panel titles.
    col_wrap:
        Max columns before wrapping.
    figsize:
        Overall figure size.
    fig_title:
        Optional suptitle.

    Returns
    -------
    Figure
    """
    from sklearn.metrics import (
        balanced_accuracy_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    if not dfs:
        raise ValueError("`dfs` must contain at least one DataFrame or path")

    if metrics_to_plot is None:
        metrics_to_plot = ["mcc", "balanced_accuracy", "recall_nogrowth", "precision_nogrowth"]

    _metric_fns = {
        "mcc": lambda yt, yp: matthews_corrcoef(yt, yp),
        "balanced_accuracy": lambda yt, yp: balanced_accuracy_score(yt, yp),
        "recall_nogrowth": lambda yt, yp: recall_score(yt, yp, pos_label=0, zero_division=0),
        "precision_nogrowth": lambda yt, yp: precision_score(yt, yp, pos_label=0, zero_division=0),
    }

    dataframes = [_as_dataframe(d) for d in dfs]
    n = len(dataframes)

    if titles is None:
        titles = [
            Path(d).stem if isinstance(d, (str, Path)) else f"Panel {i + 1}"
            for i, d in enumerate(dfs)
        ]

    fig, axes, n_rows, n_cols = _panel_subfigure(n, col_wrap=col_wrap, figsize=figsize)
    line_colors = _color_cycle(len(metrics_to_plot), palette="tab10")

    for idx, (df, title) in enumerate(zip(dataframes, titles)):
        ax = axes[idx // n_cols][idx % n_cols]
        y_true = df[y_true_col].values
        y_prob = df[y_prob_col].values

        thresholds = np.linspace(0, 1, n_thresholds)
        for m_name, color in zip(metrics_to_plot, line_colors):
            fn = _metric_fns.get(m_name)
            if fn is None:
                continue
            scores = [fn(y_true, (y_prob >= t).astype(int)) for t in thresholds]
            ax.plot(thresholds, scores, label=m_name, color=color, lw=1.5)

        ax.set_xlabel("threshold", fontsize=9)
        ax.set_ylabel("score", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8, frameon=True, loc="best")
        ax.axhline(0, color="grey", lw=0.5, alpha=0.5)

    _hide_unused_panel_axes(axes, n, n_rows, n_cols)
    fig.tight_layout(rect=(0, 0, 1.0, 0.94 if fig_title else 1.0))
    if fig_title:
        fig.suptitle(fig_title, fontsize=12, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# BacDive
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pathway-matrix analysis plots (docs/plans/pathway_features.md "Analysis")
# ---------------------------------------------------------------------------

# Colorblind-safe categorical triplet (blue/orange/aqua), validated for
# scatter all-pairs separation. Groups beyond 3 fold into "Other".
_ACADEMIC_CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]
_OTHER_COLOR = "#9a9a94"


def _fold_small_groups(series: pd.Series, max_groups: int = 3, other_label: str = "Other") -> pd.Series:
    """Keep the `max_groups` most frequent categories; fold the rest into `other_label`."""
    top = series.value_counts().nlargest(max_groups).index
    return series.where(series.isin(top), other_label)


def _style_axis(ax) -> None:
    """Despine, recessive gridlines, publication-sized tick labels."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, color="0.88", linewidth=0.6)
    ax.tick_params(labelsize=9.5)


def _label_outliers(ax, x: pd.Series, y: pd.Series, labels: pd.Series, n: int | None = 5) -> list:
    """Text-label points (all, if n=None; else the n most extreme by
    centroid distance), for `_declutter_labels` to position.
    """
    if len(x) == 0 or (n is not None and n <= 0):
        return []
    if n is None:
        top_idx = x.index
    else:
        xz = (x - x.mean()) / (x.std() or 1)
        yz = (y - y.mean()) / (y.std() or 1)
        dist = np.hypot(xz, yz)
        top_idx = dist.nlargest(min(n, len(dist))).index
    return [ax.text(x.loc[i], y.loc[i], labels.loc[i], fontsize=8, color="0.25") for i in top_idx]


def _declutter_labels(ax, texts: list, avoid_x=None, avoid_y=None) -> None:
    """Auto-repel overlapping labels (adjustText), with a leader line back
    to each moved label's point. Uses point coordinates, not adjustText's
    `objects=` bbox avoidance, which crashes on a log-scale axis. Defaults
    for time/force/movement are raised -- too low for 25+ dense labels.
    """
    if not texts:
        return
    kwargs = {}
    if avoid_x is not None:
        kwargs["x"] = list(avoid_x)
        kwargs["y"] = list(avoid_y)
    adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6, shrinkA=0, shrinkB=2),
        time_lim=25, force_text=(0.6, 0.8), max_move=(60, 60), expand=(1.3, 1.4),
        **kwargs,
    )


def _scatter_by_group(ax, df: pd.DataFrame, x: str, y: str, color_by: str | None, palette: str | None):
    """Scatter df[x] vs df[y] colored by df[color_by]; returns legend handles."""
    if color_by is None:
        ax.scatter(df[x], df[y], s=45, alpha=0.85, color=_ACADEMIC_CATEGORICAL[0],
                   edgecolors="white", linewidths=0.7)
        return {}

    grouped_col = _fold_small_groups(df[color_by].astype(str), max_groups=3)
    real_groups = sorted(g for g in grouped_col.unique() if g != "Other")
    has_other = "Other" in grouped_col.unique()
    groups = real_groups + (["Other"] if has_other else [])

    swatches = _color_cycle(len(real_groups), palette=palette) if palette else _ACADEMIC_CATEGORICAL[: len(real_groups)]
    colors = dict(zip(real_groups, swatches))
    colors["Other"] = _OTHER_COLOR

    handles: dict = {}
    for gval in groups:
        mask = grouped_col == gval
        scatter = ax.scatter(
            df.loc[mask, x], df.loc[mask, y], color=colors[gval], s=45, alpha=0.85,
            edgecolors="white", linewidths=0.7,
        )
        handles[gval] = scatter
    return handles


def plot_pathway_count_vs_assembly_quality(
    pathway_df: pd.DataFrame | str | Path,
    metadata_df: pd.DataFrame | str | Path,
    *,
    completeness_threshold: float = 0.0,
    color_by: str = "assembly_level",
    label_top_n: int = 5,
    palette: str | None = None,
    figsize: tuple[float, float] = (7, 6),
    fig_title: str | None = "Pathway count vs assembly quality",
) -> Figure:
    """Sanity check: pathway count vs assembly fragmentation (num_contigs, log scale)."""
    pdf = _as_dataframe(pathway_df)
    pdf = pdf.set_index("strain") if "strain" in pdf.columns else pdf
    metadata_df = _as_dataframe(metadata_df)

    pathway_count = (pdf > completeness_threshold).sum(axis=1).rename("pathway_count")
    merged = metadata_df.merge(pathway_count, left_on="strain", right_index=True, how="inner")

    fig, ax = plt.subplots(figsize=figsize)
    handles = _scatter_by_group(ax, merged, "num_contigs", "pathway_count", color_by, palette)
    ax.set_xscale("log")
    ax.set_xlabel("num_contigs (log scale)", fontsize=11)
    ax.set_ylabel("pathway count", fontsize=11)
    _style_axis(ax)

    texts = _label_outliers(ax, merged["num_contigs"], merged["pathway_count"], merged["strain"], n=label_top_n)
    _declutter_labels(ax, texts, avoid_x=merged["num_contigs"], avoid_y=merged["pathway_count"])

    if handles:
        fig.tight_layout(rect=(0, 0, 0.72, 0.94 if fig_title else 1.0))
        fig.legend(handles.values(), handles.keys(), title=color_by,
                   loc="center left", bbox_to_anchor=(0.74, 0.5), frameon=True, fontsize=9.5, title_fontsize=10)
    else:
        fig.tight_layout(rect=(0, 0, 1.0, 0.94 if fig_title else 1.0))
    if fig_title:
        fig.suptitle(fig_title, fontsize=13, fontweight="bold")

    return fig


def plot_pathway_pca_biplot(
    pathway_df: pd.DataFrame | str | Path,
    color_df: pd.DataFrame | str | Path | None = None,
    *,
    color_col: str = "phylum",
    binary: bool = False,
    completeness_threshold: float = 0.0,
    n_top_loadings: int = 10,
    label_top_n: int | None = None,
    palette: str | None = None,
    figsize: tuple[float, float] = (11, 9),
    fig_title: str | None = "Pathway PCA biplot",
) -> Figure:
    """PCA biplot: do pathways separate strains biologically? Arrow labels
    are module IDs (full names collide); ``binary=True`` thresholds to
    presence/absence instead of continuous completeness.
    """
    pdf = _as_dataframe(pathway_df)
    pdf = pdf.set_index("strain") if "strain" in pdf.columns else pdf
    if binary:
        pdf = (pdf > completeness_threshold).astype(int)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(pdf.values)
    scores_df = pd.DataFrame(scores, index=pdf.index, columns=["PC1", "PC2"]).reset_index()
    scores_df = scores_df.rename(columns={scores_df.columns[0]: "strain"})

    if color_df is not None:
        color_df = _as_dataframe(color_df)
        scores_df = scores_df.merge(color_df[["strain", color_col]], on="strain", how="left")

    fig, ax = plt.subplots(figsize=figsize)
    handles = _scatter_by_group(
        ax, scores_df, "PC1", "PC2", color_col if color_df is not None else None, palette
    )
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]:.1%} var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]:.1%} var)", fontsize=11)
    _style_axis(ax)

    point_texts = _label_outliers(ax, scores_df["PC1"], scores_df["PC2"], scores_df["strain"], n=label_top_n)

    # Top-N loading arrows, scaled to roughly span the score cloud.
    loadings = pca.components_[:2, :]
    magnitude = np.linalg.norm(loadings, axis=0)
    top_idx = np.argsort(magnitude)[::-1][:n_top_loadings]
    scale = 0.8 * max(scores_df["PC1"].abs().max(), scores_df["PC2"].abs().max()) / max(magnitude[top_idx].max(), 1e-9)
    arrow_x = loadings[0, top_idx] * scale
    arrow_y = loadings[1, top_idx] * scale
    arrow_texts = []
    for i, ax_i, ay_i in zip(top_idx, arrow_x, arrow_y):
        ax.annotate(
            "", xy=(ax_i, ay_i), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="0.4", alpha=0.8, lw=1.1),
        )
        arrow_texts.append(ax.text(ax_i, ay_i, pdf.columns[i], fontsize=8, color="0.3"))

    # ax.annotate's arrows don't participate in autoscaling -- expand the
    # limits explicitly so arrow tips (which can extend past the score
    # cloud) aren't silently clipped outside the visible axes, before
    # adjustText runs (it needs final axes limits to place labels well).
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    pad_x = 0.08 * (x_hi - x_lo)
    pad_y = 0.08 * (y_hi - y_lo)
    ax.set_xlim(min(x_lo, arrow_x.min()) - pad_x, max(x_hi, arrow_x.max()) + pad_x)
    ax.set_ylim(min(y_lo, arrow_y.min()) - pad_y, max(y_hi, arrow_y.max()) + pad_y)

    # One combined declutter pass so strain labels and arrow labels repel
    # each other too, not just within their own group. Every strain point
    # (not just labeled ones) and every arrow tip is passed as something
    # to avoid, so labels don't drift onto an unlabeled point either.
    _declutter_labels(
        ax, point_texts + arrow_texts,
        avoid_x=list(scores_df["PC1"]) + list(arrow_x),
        avoid_y=list(scores_df["PC2"]) + list(arrow_y),
    )

    if handles:
        fig.tight_layout(rect=(0, 0, 0.72, 0.94 if fig_title else 1.0))
        fig.legend(handles.values(), handles.keys(), title=color_col,
                   loc="center left", bbox_to_anchor=(0.74, 0.5), frameon=True, fontsize=9.5, title_fontsize=10)
    else:
        fig.tight_layout(rect=(0, 0, 1.0, 0.94 if fig_title else 1.0))
    if fig_title:
        fig.suptitle(fig_title, fontsize=12, fontweight="bold")

    return fig


def plot_pathway_count_vs_growth(
    pathway_df: pd.DataFrame | str | Path,
    curves_df: pd.DataFrame | str | Path,
    *,
    growth_col: str = "max_od",
    completeness_threshold: float = 0.0,
    color_df: pd.DataFrame | str | Path | None = None,
    color_col: str = "phylum",
    label_top_n: int = 5,
    palette: str | None = None,
    figsize: tuple[float, float] = (7, 6),
    fig_title: str | None = "Pathway count vs growth",
) -> Figure:
    """Pathway count vs mean growth (e.g. max_od), with Pearson r."""
    pdf = _as_dataframe(pathway_df)
    pdf = pdf.set_index("strain") if "strain" in pdf.columns else pdf
    pathway_count = (pdf > completeness_threshold).sum(axis=1).rename("pathway_count")

    curves_df = _as_dataframe(curves_df)
    mean_growth = curves_df.groupby("strain_name")[growth_col].mean().rename(growth_col)

    merged = pd.DataFrame(pathway_count).join(mean_growth, how="inner").reset_index()
    merged = merged.rename(columns={merged.columns[0]: "strain"})

    if color_df is not None:
        color_df = _as_dataframe(color_df)
        merged = merged.merge(color_df[["strain", color_col]], on="strain", how="left")

    fig, ax = plt.subplots(figsize=figsize)
    handles = _scatter_by_group(
        ax, merged, "pathway_count", growth_col, color_col if color_df is not None else None, palette
    )
    ax.set_xlabel("pathway count", fontsize=11)
    ax.set_ylabel(f"mean {growth_col}", fontsize=11)
    _style_axis(ax)

    texts = _label_outliers(ax, merged["pathway_count"], merged[growth_col], merged["strain"], n=label_top_n)
    _declutter_labels(ax, texts, avoid_x=merged["pathway_count"], avoid_y=merged[growth_col])

    r = np.corrcoef(merged["pathway_count"], merged[growth_col])[0, 1] if len(merged) > 2 else float("nan")
    ax.text(0.05, 0.95, f"Pearson r = {r:.2f}", transform=ax.transAxes,
            fontsize=10, va="top", bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    if handles:
        fig.tight_layout(rect=(0, 0, 0.72, 0.94 if fig_title else 1.0))
        fig.legend(handles.values(), handles.keys(), title=color_col,
                   loc="center left", bbox_to_anchor=(0.74, 0.5), frameon=True, fontsize=9.5, title_fontsize=10)
    else:
        fig.tight_layout(rect=(0, 0, 1.0, 0.94 if fig_title else 1.0))
    if fig_title:
        fig.suptitle(fig_title, fontsize=13, fontweight="bold")

    return fig


def plot_pca_scree(
    pathway_df: pd.DataFrame | str | Path,
    *,
    max_components: int | None = None,
    reference_lines: tuple[float, ...] = (0.8, 0.9, 0.95),
    figsize: tuple[float, float] = (7, 5),
    fig_title: str | None = "Cumulative explained variance",
) -> Figure:
    """Cumulative explained variance vs number of PCA components."""
    pdf = _as_dataframe(pathway_df)
    pdf = pdf.set_index("strain") if "strain" in pdf.columns else pdf

    n = max_components or min(pdf.shape)
    pca = PCA(n_components=n).fit(pdf.values)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(np.arange(1, n + 1), cumvar, marker="o", ms=4, lw=1.8, color=_ACADEMIC_CATEGORICAL[0])
    for ref in reference_lines:
        ax.axhline(ref, color="0.6", lw=0.8, ls="--", alpha=0.8)
        ax.text(n, ref, f"{ref:.0%}", fontsize=8.5, color="0.35", va="bottom", ha="right")
    ax.set_xlabel("number of components", fontsize=11)
    ax.set_ylabel("cumulative explained variance", fontsize=11)
    ax.set_ylim(0, 1.02)
    _style_axis(ax)
    fig.tight_layout(rect=(0, 0, 1.0, 0.94 if fig_title else 1.0))
    if fig_title:
        fig.suptitle(fig_title, fontsize=13, fontweight="bold")

    return fig


def plot_loso_vs_n_components(
    results_df: pd.DataFrame | str | Path,
    *,
    n_col: str = "n_components",
    mean_col: str = "metric_mean",
    std_col: str = "metric_std",
    metric_label: str = "RMSE",
    baseline_value: float | None = None,
    baseline_label: str = "baseline features",
    figsize: tuple[float, float] = (7, 5),
    fig_title: str | None = "LOSO performance vs number of PCA components",
) -> Figure:
    """LOSO performance (mean +/- std) vs number of PCA components, with an
    optional baseline reference line. Expects `results_df` already computed.
    """
    df = _as_dataframe(results_df)

    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(
        df[n_col], df[mean_col], yerr=df[std_col],
        marker="o", ms=6, lw=1.8, capsize=3, capthick=1.2, color=_ACADEMIC_CATEGORICAL[0],
        markeredgecolor="white", markeredgewidth=0.7, label=metric_label,
    )
    if baseline_value is not None:
        ax.axhline(baseline_value, color="0.5", lw=1.2, ls="--", alpha=0.9, label=baseline_label)

    ax.set_xlabel("number of PCA components", fontsize=11)
    ax.set_ylabel(metric_label, fontsize=11)
    _style_axis(ax)
    ax.legend(frameon=True, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1.0, 0.94 if fig_title else 1.0))
    if fig_title:
        fig.suptitle(fig_title, fontsize=13, fontweight="bold")

    return fig

plot_