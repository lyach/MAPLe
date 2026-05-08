"""Growth curve plotting functions.

All functions accept a DataFrame loaded from the pipeline parquet output
(see docs/dataset_pipeline.md for the full schema).  They return 
`matplotlib.figure.Figure` objects.

Usage (notebook):
    import matplotlib.pyplot as plt
    from src.dataset.plots import load_data, plot_mean_curves
    df = load_data("data/processed/20260325_aerobic_tidy.parquet")
    figs = plot_mean_curves(df)  # all experiments
    plt.show()
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


STYLE_DEFAULTS: dict = {
    "figure.figsize": (12, 5),
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
}

_CV_THRESHOLD = 0.15


ExpIdArg = Union[str, list[str], None]


def load_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load the pipeline parquet output into a DataFrame.

    Parameters
    ----------
    path:
        Path to the ``.parquet`` file produced by ``run_dataset_pipeline.py``.
    """
    return pd.read_parquet(path)


def _filter_exp(df: pd.DataFrame, exp_id: ExpIdArg) -> pd.DataFrame:
    """Return a copy of *df* restricted to the requested experiment(s).
    """
    if exp_id is None:
        return df
    if isinstance(exp_id, str):
        exp_id = [exp_id]
    missing = set(exp_id) - set(df["exp_id"].unique())
    if missing:
        raise ValueError(f"exp_id(s) not found in DataFrame: {sorted(missing)}")
    return df[df["exp_id"].isin(exp_id)].copy()


def _time_label(df_exp: pd.DataFrame) -> str:
    """Return an axis label like ``'Time (min)'``."""
    unit = df_exp["time_unit"].dropna().iloc[0] if not df_exp.empty else ""
    return f"Time ({unit})" if unit else "Time"


def _apply_style(rc_overrides: dict | None = None) -> None:
    """Apply STYLE_DEFAULTS then any caller overrides to rcParams."""
    plt.rcParams.update(STYLE_DEFAULTS)
    if rc_overrides:
        plt.rcParams.update(rc_overrides)


def _color_cycle(n: int) -> list:
    """Return *n* distinct colours from the current colour cycle."""
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return [prop_cycle[i % len(prop_cycle)] for i in range(n)]


def plot_growth_curves(
    df: pd.DataFrame,
    exp_id: ExpIdArg = None,
    *,
    od_col: str = "od600",
    hue: str = "sample_name",
    col_wrap: int = 4,
    show_no_growth: bool = True,
    exclude_blanks: bool = True,
    rc: dict | None = None,
    figsize: tuple[float, float] | None = None,
) -> list[Figure]:
    """Plot individual well growth curves, by media.

    One figure per experiment. Subplots are arranged in a grid, 
    one subplot per media. Each line represents one well.
    No growth wells are drawn as thin dashed grey lines.

    Parameters
    ----------
    df:
        DataFrame from ``load_data``.
    exp_id:
        Experiment(s) to plot.  ``None`` → all.
    od_col:
        OD column to use (``"od600"`` or ``"od600_raw"``).
    hue:
        Column used to colour lines (``"sample_name"`` or ``"well_id"``).
    col_wrap:
        Maximum number of subplot columns per row.
    show_no_growth:
        If ``False``, no-growth wells are omitted entirely.
    exclude_blanks:
        Skip blank wells (``is_blank=True``).
    rc:
        Optional rcParams overrides.
    figsize:
        Override the default figure size.

    Returns
    -------
    list[Figure]
        One figure per experiment.
    """
    _apply_style(rc)
    df_sel = _filter_exp(df, exp_id)
    if exclude_blanks and "is_blank" in df_sel.columns:
        df_sel = df_sel[~df_sel["is_blank"]]

    figs: list[Figure] = []

    for eid, df_exp in df_sel.groupby("exp_id", sort=True):
        medias = sorted(df_exp["media_name"].dropna().unique())
        n_media = len(medias)
        n_cols = min(col_wrap, n_media)
        n_rows = int(np.ceil(n_media / n_cols))

        default_w = max(3.5 * n_cols, 8)
        default_h = max(3.0 * n_rows, 4)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize or (default_w, default_h),
            squeeze=False,
        )
        fig.suptitle(f"Growth curves — {eid}", fontsize=11, fontweight="bold")

        hue_vals = sorted(df_exp[hue].dropna().unique())
        colors = {v: c for v, c in zip(hue_vals, _color_cycle(len(hue_vals)))}
        legend_handles: dict = {}

        for idx, media in enumerate(medias):
            ax = axes[idx // n_cols][idx % n_cols]
            df_media = df_exp[df_exp["media_name"] == media]

            for well_id, df_well in df_media.groupby("well_id", sort=False):
                df_well = df_well.sort_values("time_value")
                hue_val = df_well[hue].iloc[0]
                ng = bool(df_well["no_growth"].iloc[0]) if "no_growth" in df_well else False

                if ng and not show_no_growth:
                    continue

                lw = 0.7 if ng else STYLE_DEFAULTS["lines.linewidth"]
                ls = "--" if ng else "-"
                alpha = 0.45 if ng else 0.85
                color = "grey" if ng else colors[hue_val]

                line, = ax.plot(
                    df_well["time_value"],
                    df_well[od_col],
                    color=color,
                    lw=lw,
                    ls=ls,
                    alpha=alpha,
                )
                if not ng and hue_val not in legend_handles:
                    legend_handles[hue_val] = line

                if "plate_cv_flag" in df_well.columns and df_well["plate_cv_flag"].any():
                    ax.set_facecolor("#fff8f0")

            ax.set_title(media, pad=4)
            ax.set_xlabel(_time_label(df_exp))
            ax.set_ylabel(od_col)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
            ax.set_ylim(bottom=0)

        # hide unused axes
        for idx in range(n_media, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        if legend_handles:
            fig.legend(
                legend_handles.values(),
                legend_handles.keys(),
                title=hue,
                loc="lower right",
                bbox_to_anchor=(1.0, 0.01),
                frameon=True,
                borderaxespad=0.5,
            )

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        figs.append(fig)

    return figs


def plot_mean_curves(
    df: pd.DataFrame,
    exp_id: ExpIdArg = None,
    *,
    od_col: str = "od600",
    ci: int = 95,
    col_wrap: int = 4,
    exclude_blanks: bool = True,
    exclude_no_growth: bool = False,
    rc: dict | None = None,
    figsize: tuple[float, float] | None = None,
) -> list[Figure]:
    """Plot mean growth curves with a shaded confidence interval.

    Aggregates all wells per experiment sharing the same (sample_name, media_name)
    at each timepoint.  A shaded band shows the ci-percentile interval
    around the mean.

    Parameters
    ----------
    df:
        DataFrame from ``load_data``.
    exp_id:
        Experiment(s) to plot.
    od_col:
        OD column (``"od600"`` or ``"od600_raw"``).
    ci:
        Confidence interval width in percent (e.g. ``95`` → 2.5–97.5 percentile band).
    col_wrap:
        Maximum subplot columns per row.
    exclude_blanks:
        Skip blank wells.
    exclude_no_growth:
        If ``True``, no-growth wells are excluded before aggregation.
    rc:
        Optional rcParams overrides.
    figsize:
        Override default figure size.

    Returns
    -------
    list[Figure]
    """
    _apply_style(rc)
    df_sel = _filter_exp(df, exp_id)
    if exclude_blanks and "is_blank" in df_sel.columns:
        df_sel = df_sel[~df_sel["is_blank"]]
    if exclude_no_growth and "no_growth" in df_sel.columns:
        df_sel = df_sel[~df_sel["no_growth"]]

    lo = (100 - ci) / 2
    hi = 100 - lo

    figs: list[Figure] = []

    # group by experiment
    for eid, df_exp in df_sel.groupby("exp_id", sort=True):
        # group by media
        medias = sorted(df_exp["media_name"].dropna().unique())
        n_media = len(medias)
        n_cols = min(col_wrap, n_media)
        n_rows = int(np.ceil(n_media / n_cols))

        default_w = max(3.5 * n_cols, 8)
        default_h = max(3.0 * n_rows, 4)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize or (default_w, default_h),
            squeeze=False,
        )
        fig.suptitle(f"{eid}", fontsize=11, fontweight="bold")

        samples = sorted(df_exp["sample_name"].dropna().unique())
        colors = {s: c for s, c in zip(samples, _color_cycle(len(samples)))}
        legend_handles: dict = {}

        for idx, media in enumerate(medias):
            ax = axes[idx // n_cols][idx % n_cols]
            df_media = df_exp[df_exp["media_name"] == media]

            # group by sample
            for sample, df_s in df_media.groupby("sample_name", sort=True):
                stats = (
                    df_s.groupby("time_value")[od_col]
                    .agg(
                        mean="mean",
                        lo=lambda x, _lo=lo: np.percentile(x, _lo),
                        hi=lambda x, _hi=hi: np.percentile(x, _hi),
                    )
                    .reset_index()
                    .sort_values("time_value")
                )
                color = colors[sample]
                (line,) = ax.plot(stats["time_value"], stats["mean"], color=color, lw=1.6)
                ax.fill_between(
                    stats["time_value"],
                    stats["lo"],
                    stats["hi"],
                    color=color,
                    alpha=0.18,
                )
                if sample not in legend_handles:
                    legend_handles[sample] = line

            ax.set_title(media, pad=4)
            ax.set_xlabel(_time_label(df_exp))
            ax.set_ylabel(od_col)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
            ax.set_ylim(bottom=0)

        for idx in range(n_media, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        if legend_handles:
            fig.legend(
                legend_handles.values(),
                legend_handles.keys(),
                title="sample",
                loc="lower right",
                bbox_to_anchor=(1.0, 0.01),
                frameon=True,
                borderaxespad=0.5,
            )

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        figs.append(fig)

    return figs


def plot_plate_grid(
    df: pd.DataFrame,
    exp_id: ExpIdArg = None,
    *,
    od_col: str = "od600",
    well_size: float = 1.5,
    rc: dict | None = None,
) -> list[Figure]:
    """Plate-layout view: one tiny subplot per well arranged by (row, col).

    Wells flagged ``no_growth=True`` have a light red background; blank wells
    have a light blue background.

    Parameters
    ----------
    df:
        DataFrame from ``load_data``.
    exp_id:
        Experiment(s) to plot.
    od_col:
        OD column to plot.
    well_size:
        Width and height (inches) of each well subplot.
    rc:
        Optional rcParams overrides.

    Returns
    -------
    list[Figure]
    """
    _apply_style(rc)
    df_sel = _filter_exp(df, exp_id)

    figs: list[Figure] = []

    for eid, df_exp in df_sel.groupby("exp_id", sort=True):
        rows_sorted = sorted(df_exp["row"].dropna().unique())
        cols_sorted = sorted(df_exp["col"].dropna().unique())
        n_rows, n_cols = len(rows_sorted), len(cols_sorted)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(well_size * n_cols, well_size * n_rows),
            squeeze=False,
        )
        fig.suptitle(f"Plate grid — {eid}", fontsize=11, fontweight="bold", y=1.01)

        row_idx = {r: i for i, r in enumerate(rows_sorted)}
        col_idx = {c: i for i, c in enumerate(cols_sorted)}

        # collect per-well data
        for (row, col), df_well in df_exp.groupby(["row", "col"], sort=False):
            ax = axes[row_idx[row]][col_idx[col]]
            df_well = df_well.sort_values("time_value")

            is_blank = bool(df_well["is_blank"].any()) if "is_blank" in df_well else False
            ng = bool(df_well["no_growth"].iloc[0]) if "no_growth" in df_well else False

            if is_blank:
                ax.set_facecolor("#ddeeff")
                color = "#5588bb"
            elif ng:
                ax.set_facecolor("#fde8e8")
                color = "#cc4444"
            else:
                color = "#2266cc"

            ax.plot(df_well["time_value"], df_well[od_col], color=color, lw=0.8)
            ax.set_ylim(bottom=0)
            ax.set_xticks([])
            ax.set_yticks([])

            # row/col labels on outer edges only
            if col_idx[col] == 0:
                ax.set_ylabel(row.upper(), fontsize=7, labelpad=2, rotation=0, va="center")
            if row_idx[row] == 0:
                ax.set_title(str(col), fontsize=7, pad=2)

        # hide axes not covered by well data
        used = {
            (row_idx[r], col_idx[c])
            for r, c in df_exp.groupby(["row", "col"]).groups
        }
        for ri in range(n_rows):
            for ci in range(n_cols):
                if (ri, ci) not in used:
                    axes[ri][ci].set_visible(False)

        fig.tight_layout()
        figs.append(fig)

    return figs


def plot_qc_summary(
    df: pd.DataFrame,
    exp_id: ExpIdArg = None,
    *,
    cv_threshold: float = _CV_THRESHOLD,
    rc: dict | None = None,
    figsize: tuple[float, float] | None = None,
) -> list[Figure]:
    """Two-panel QC summary per experiment.

    Left panel — ``blank_cv`` over time with a dashed threshold line.
    Right panel — bar chart of no-growth well counts per ``(sample_name, media_name)``.

    Parameters
    ----------
    df:
        DataFrame from ``load_data``.
    exp_id:
        Experiment(s) to plot.
    cv_threshold:
        Threshold drawn as a dashed line on the CV panel (default 0.15).
    rc:
        Optional rcParams overrides.
    figsize:
        Override default figure size ``(10, 4)``.

    Returns
    -------
    list[Figure]
    """
    _apply_style(rc)
    df_sel = _filter_exp(df, exp_id)

    figs: list[Figure] = []

    for eid, df_exp in df_sel.groupby("exp_id", sort=True):
        fig, (ax_cv, ax_ng) = plt.subplots(
            1, 2, figsize=figsize or (10, 4)
        )
        fig.suptitle(f"QC summary — {eid}", fontsize=11, fontweight="bold")

        # ---- left: blank_cv over time ----
        if "blank_cv" in df_exp.columns:
            cv_ts = (
                df_exp.groupby("time_value")["blank_cv"]
                .mean()
                .reset_index()
                .sort_values("time_value")
            )
            ax_cv.plot(cv_ts["time_value"], cv_ts["blank_cv"], marker="o", ms=3, lw=1.4)
            ax_cv.axhline(cv_threshold, color="red", ls="--", lw=1.0, label=f"threshold ({cv_threshold})")
            flagged = (cv_ts["blank_cv"] > cv_threshold).sum()
            ax_cv.set_title(f"Blank CV over time  ({flagged} timepoints flagged)")
        else:
            ax_cv.text(0.5, 0.5, "blank_cv not available", ha="center", va="center",
                       transform=ax_cv.transAxes)
            ax_cv.set_title("Blank CV over time")

        ax_cv.set_xlabel(_time_label(df_exp))
        ax_cv.set_ylabel("blank_cv")
        ax_cv.set_ylim(bottom=0)
        ax_cv.legend(fontsize=8)

        # ---- right: no-growth counts ----
        if "no_growth" in df_exp.columns:
            df_ng = df_exp[~df_exp.get("is_blank", pd.Series(False, index=df_exp.index))]
            ng_counts = (
                df_ng[df_ng["no_growth"]]
                .drop_duplicates(subset=["well_id"])
                .groupby(["sample_name", "media_name"])
                .size()
                .reset_index(name="count")
            )

            if ng_counts.empty:
                ax_ng.text(0.5, 0.5, "No no-growth wells", ha="center", va="center",
                           transform=ax_ng.transAxes)
            else:
                labels = [f"{r.sample_name}\n{r.media_name}" for r in ng_counts.itertuples()]
                ax_ng.barh(labels, ng_counts["count"], color="#cc4444", alpha=0.75)
                ax_ng.set_xlabel("no-growth well count")
                ax_ng.invert_yaxis()
        else:
            ax_ng.text(0.5, 0.5, "no_growth not available", ha="center", va="center",
                       transform=ax_ng.transAxes)

        ax_ng.set_title("No-growth wells by sample × media")
        ax_ng.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        figs.append(fig)

    return figs
