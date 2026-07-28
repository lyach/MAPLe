"""Utility functions for plotting modeling results"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def read_dataframe_from_csv(path: str) -> pd.DataFrame:
    """Read dataframe from CSV file"""
    return pd.read_csv(path)


def _as_dataframe(df_or_path: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Coerce a DataFrame-or-CSV-path input into a DataFrame."""
    if isinstance(df_or_path, pd.DataFrame):
        return df_or_path
    return read_dataframe_from_csv(str(df_or_path))


def _color_cycle(n: int) -> list:
    """Return *n* distinct colours from the current colour cycle."""
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return [prop_cycle[i % len(prop_cycle)] for i in range(n)]


def plot_scatter_predictions(
    dfs: list[pd.DataFrame | str | Path],
    color: str,
    *,
    y_pred_col: str = "y_pred",
    y_true_col: str = "y_true",
    titles: list[str] | None = None,
    col_wrap: int = 2,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Plot predicted-vs-true scatter panels for one or more prediction sets.

    Parameters
    ----------
    dfs:
        List of dataframes and/or CSV paths.
    color:
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
    n_rows = int(np.ceil(n / n_cols))

    # Shared color mapping, pooled across all panels, for a single legend.
    color_vals = sorted(
        pd.concat([df[color] for df in dataframes], ignore_index=True).dropna().unique()
    )
    colors = {v: c for v, c in zip(color_vals, _color_cycle(len(color_vals)))}

    default_w = max(4.5 * n_cols, 5)
    default_h = max(4.0 * n_rows, 4)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize or (default_w, default_h),
        squeeze=False,
    )

    legend_handles: dict = {}

    for idx, (df, title) in enumerate(zip(dataframes, titles)):
        ax = axes[idx // n_cols][idx % n_cols]

        for cval, grp in df.groupby(color, sort=True):
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

    # hide unused axes
    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    # Reserve space on the right for a single, shared legend
    axes_right = 1.0
    if legend_handles:
        axes_right = 0.72
        fig.tight_layout(rect=(0, 0, axes_right, 0.96))
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            title=color,
            loc="center left",
            bbox_to_anchor=(axes_right + 0.02, 0.5),
            frameon=True,
            borderaxespad=0.5,
        )
    else:
        fig.tight_layout(rect=(0, 0, axes_right, 0.96))

    return fig
