"""Utility functions for plotting modeling results"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
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
    axes_right: float = 0.72,
) -> None:
    """Reserve space on the right and draw one figure-level legend."""
    if legend_handles:
        fig.tight_layout(rect=(0, 0, axes_right, 0.96))
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            title=title,
            loc="center left",
            bbox_to_anchor=(axes_right + 0.02, 0.5),
            frameon=True,
            borderaxespad=0.5,
        )
    else:
        fig.tight_layout(rect=(0, 0, 1.0, 0.96))


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
        Seaborn palette name.Falls back to the default matplotlib colour cycle when ``None``.

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
    _add_shared_side_legend(fig, legend_handles, title=color_by)

    return fig


def plot_metrics_comparison(
    jsons: list[dict | str | Path],
    metrics: list[str],
    *,
    metric_titles: list[str] | None = None,
    json_titles: list[str] | None = None,
    legend_title: str | None = "experiment",
    col_wrap: int = 3,
    figsize: tuple[float, float] | None = None,
    palette: str = "Set1",
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
        ax.set_xticks(x)
        ax.set_xticklabels(json_titles, rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="grey", lw=0.6, alpha=0.5, zorder=0)

    _hide_unused_panel_axes(axes, n_metrics, n_rows, n_cols)
    _add_shared_side_legend(fig, legend_handles, title=legend_title)

    return fig
