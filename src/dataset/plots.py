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
import seaborn as sns
from matplotlib.colors import ListedColormap
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


# Time units supported for axis display
_TIME_UNIT_ALIASES: dict[str, str] = {
    "min": "min",
    "minute": "min",
    "minutes": "min",
    "hr": "hr",
    "hour": "hr",
    "hours": "hr",
    "h": "hr",
    "day": "day",
    "days": "day",
    "d": "day",
}

# Duration of one unit in minutes (for conversions)
_MINUTES_PER_CANONICAL: dict[str, float] = {"min": 1.0, "hr": 60.0, "day": 1440.0}


def _canonical_time_unit(u: str) -> str:
    key = u.strip().lower()
    try:
        return _TIME_UNIT_ALIASES[key]
    except KeyError as e:
        allowed = ", ".join(sorted(set(_TIME_UNIT_ALIASES.values())))
        raise ValueError(
            f"Unknown time unit {u!r}; use one of {allowed} (common aliases accepted)."
        ) from e


def _time_axis_scale_and_label(
    df_exp: pd.DataFrame,
    time_unit: str | None,
) -> tuple[float, str]:
    """Return ``(scale, xlabel)`` so ``plot_x = time_value * scale``.
    """
    if time_unit is None:
        return 1.0, _time_label(df_exp)
    display = _canonical_time_unit(time_unit)
    if df_exp.empty or "time_unit" not in df_exp.columns:
        src = "min"
    else:
        src = _canonical_time_unit(str(df_exp["time_unit"].dropna().iloc[0]))
    scale = _MINUTES_PER_CANONICAL[src] / _MINUTES_PER_CANONICAL[display]
    return scale, f"Time ({display})"


def _apply_style(rc_overrides: dict | None = None) -> None:
    """Apply STYLE_DEFAULTS then any caller overrides to rcParams."""
    plt.rcParams.update(STYLE_DEFAULTS)
    if rc_overrides:
        plt.rcParams.update(rc_overrides)


def _color_cycle(n: int) -> list:
    """Return *n* distinct colours from the current colour cycle."""
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return [prop_cycle[i % len(prop_cycle)] for i in range(n)]


def _exp_date_and_id(df_exp: pd.DataFrame) -> tuple[str, str]:
    """Return ``(date, exp_id)`` for a single experiment.
    """
    eid_date = str(df_exp["date"].iloc[0])
    eid = str(df_exp["exp_id"].iloc[0])
    return eid_date, eid


# ---------------------------------------------------------------------------
# Growth curve plots
# ---------------------------------------------------------------------------

def plot_growth_curves(
    df: pd.DataFrame,
    exp_id: ExpIdArg = None,
    *,
    od_col: str = "od600",
    hue: str = "strain_name",
    col_wrap: int = 4,
    show_no_growth: bool = True,
    exclude_blanks: bool = True,
    rc: dict | None = None,
    figsize: tuple[float, float] | None = None,
    time_unit: str | None = None,
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
        Column used to colour lines (``"strain_name"`` or ``"well_id"``).
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
    time_unit:
        Unit for the time axis: ``'min'``, ``'hr'``, or ``'day'``.
        ``None`` → plot raw ``time_value`` with the stored unit label.

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

    for _, df_exp in df_sel.groupby("exp_id", sort=True):
        eid_date, eid = _exp_date_and_id(df_exp)
        t_scale, x_time_label = _time_axis_scale_and_label(df_exp, time_unit)

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
        fig.suptitle(f"Growth - {eid_date} {eid}", fontsize=11, fontweight="bold")

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
                    df_well["time_value"].to_numpy(dtype=float) * t_scale,
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
            ax.set_xlabel(x_time_label)
            ax.set_ylabel(od_col)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
            ax.set_ylim(bottom=0)

        # hide unused axes
        for idx in range(n_media, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        # Reserve space on the right for legend
        axes_right = 1.0
        if legend_handles:
            axes_right = 0.72
            fig.tight_layout(rect=(0, 0, axes_right, 0.96))
            fig.legend(
                legend_handles.values(),
                legend_handles.keys(),
                title=hue,
                loc="center left",
                bbox_to_anchor=(axes_right + 0.02, 0.5),
                frameon=True,
                borderaxespad=0.5,
            )
        else:
            fig.tight_layout(rect=(0, 0, axes_right, 0.96))
        figs.append(fig)

    return figs


def plot_growth_curves_by_strain(
    df: pd.DataFrame,
    exp_id: ExpIdArg = None,
    *,
    od_col: str = "od600",
    hue: str = "media_name",
    col_wrap: int = 4,
    show_no_growth: bool = True,
    exclude_blanks: bool = True,
    water_media_name: str = "water",
    rc: dict | None = None,
    figsize: tuple[float, float] | None = None,
    time_unit: str | None = None,
) -> list[Figure]:
    """Plot individual well growth curves, by strain.

    One figure per experiment. Subplots are arranged in a grid,
    one subplot per strain. Each line represents one well.
    No-growth wells are drawn as thin dashed grey lines.
    Wells whose ``media_name`` matches *water_media_name* are always drawn
    as black dashed lines regardless of the *hue* colour mapping.

    Parameters
    ----------
    df:
        DataFrame from ``load_data``.
    exp_id:
        Experiment(s) to plot.  ``None`` → all.
    od_col:
        OD column to use (``"od600"`` or ``"od600_raw"``).
    hue:
        Column used to colour lines (``"media_name"`` or ``"well_id"``).
    col_wrap:
        Maximum number of subplot columns per row.
    show_no_growth:
        If ``False``, no-growth wells are omitted entirely.
    exclude_blanks:
        Skip blank wells (``is_blank=True``).
    water_media_name:
        Value of ``media_name`` treated as the water/blank control; plotted
        as a black dashed line.
    rc:
        Optional rcParams overrides.
    figsize:
        Override the default figure size.
    time_unit:
        Unit for the time axis: ``'min'``, ``'hr'``, or ``'day'``.
        ``None`` → plot raw ``time_value`` with the stored unit label.

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

    for _, df_exp in df_sel.groupby("exp_id", sort=True):
        eid_date, eid = _exp_date_and_id(df_exp)
        t_scale, x_time_label = _time_axis_scale_and_label(df_exp, time_unit)

        strains = sorted(df_exp["strain_name"].dropna().unique())
        n_strains = len(strains)
        n_cols = min(col_wrap, n_strains)
        n_rows = int(np.ceil(n_strains / n_cols))

        default_w = max(3.5 * n_cols, 8)
        default_h = max(3.0 * n_rows, 4)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize or (default_w, default_h),
            squeeze=False,
        )
        fig.suptitle(f"Growth by Strain - {eid_date} {eid}", fontsize=11, fontweight="bold")

        hue_vals = sorted(df_exp[hue].dropna().unique())
        colors = {v: c for v, c in zip(hue_vals, _color_cycle(len(hue_vals)))}
        legend_handles: dict = {}

        for idx, strain in enumerate(strains):
            ax = axes[idx // n_cols][idx % n_cols]
            df_strain = df_exp[df_exp["strain_name"] == strain]

            for well_id, df_well in df_strain.groupby("well_id", sort=False):
                df_well = df_well.sort_values("time_value")
                hue_val = df_well[hue].iloc[0]
                ng = bool(df_well["no_growth"].iloc[0]) if "no_growth" in df_well else False
                is_water = str(df_well["media_name"].iloc[0]) == water_media_name

                if ng and not show_no_growth:
                    continue

                if is_water:
                    color = "black"
                    lw = STYLE_DEFAULTS["lines.linewidth"]
                    ls = "--"
                    alpha = 0.7
                elif ng:
                    color = "grey"
                    lw = 0.7
                    ls = "--"
                    alpha = 0.45
                else:
                    color = colors[hue_val]
                    lw = STYLE_DEFAULTS["lines.linewidth"]
                    ls = "-"
                    alpha = 0.85

                line, = ax.plot(
                    df_well["time_value"].to_numpy(dtype=float) * t_scale,
                    df_well[od_col],
                    color=color,
                    lw=lw,
                    ls=ls,
                    alpha=alpha,
                )
                if is_water:
                    if water_media_name not in legend_handles:
                        legend_handles[water_media_name] = line
                elif not ng and hue_val not in legend_handles:
                    legend_handles[hue_val] = line

                if "plate_cv_flag" in df_well.columns and df_well["plate_cv_flag"].any():
                    ax.set_facecolor("#fff8f0")

            ax.set_title(strain, pad=4)
            ax.set_xlabel(x_time_label)
            ax.set_ylabel(od_col)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
            ax.set_ylim(bottom=0)

        # hide unused axes
        for idx in range(n_strains, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        # Reserve space on the right for legend
        axes_right = 1.0
        if legend_handles:
            axes_right = 0.72
            fig.tight_layout(rect=(0, 0, axes_right, 0.96))
            fig.legend(
                legend_handles.values(),
                legend_handles.keys(),
                title=hue,
                loc="center left",
                bbox_to_anchor=(axes_right + 0.02, 0.5),
                frameon=True,
                borderaxespad=0.5,
            )
        else:
            fig.tight_layout(rect=(0, 0, axes_right, 0.96))
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
    time_unit: str | None = None,
) -> list[Figure]:
    """Plot mean growth curves with a shaded confidence interval.

    Aggregates all wells per experiment sharing the same (strain_name, media_name)
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
    time_unit:
        Unit for the time axis: ``'min'``, ``'hr'``, or ``'day'``.
        ``None`` → plot raw ``time_value`` with the stored unit label.

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
    for _, df_exp in df_sel.groupby("exp_id", sort=True):
        eid_date, eid = _exp_date_and_id(df_exp)
        t_scale, x_time_label = _time_axis_scale_and_label(df_exp, time_unit)
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
        fig.suptitle(f"Mean growth — {eid_date} {eid}", fontsize=11, fontweight="bold")

        strains = sorted(df_exp["strain_name"].dropna().unique())
        colors = {s: c for s, c in zip(strains, _color_cycle(len(strains)))}
        legend_handles: dict = {}

        for idx, media in enumerate(medias):
            ax = axes[idx // n_cols][idx % n_cols]
            df_media = df_exp[df_exp["media_name"] == media]

            # group by strain
            for strain, df_s in df_media.groupby("strain_name", sort=True):
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
                color = colors[strain]
                tx = stats["time_value"].to_numpy(dtype=float) * t_scale
                (line,) = ax.plot(tx, stats["mean"], color=color, lw=1.6)
                ax.fill_between(
                    tx,
                    stats["lo"],
                    stats["hi"],
                    color=color,
                    alpha=0.18,
                )
                if strain not in legend_handles:
                    legend_handles[strain] = line

            ax.set_title(media, pad=4)
            ax.set_xlabel(x_time_label)
            ax.set_ylabel(od_col)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
            ax.set_ylim(bottom=0)

        for idx in range(n_media, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        axes_right = 1.0
        if legend_handles:
            axes_right = 0.72
            fig.tight_layout(rect=(0, 0, axes_right, 0.96))
            fig.legend(
                legend_handles.values(),
                legend_handles.keys(),
                title="strain",
                loc="center left",
                bbox_to_anchor=(axes_right + 0.02, 0.5),
                frameon=True,
                borderaxespad=0.5,
            )
        else:
            fig.tight_layout(rect=(0, 0, axes_right, 0.96))
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

    for _, df_exp in df_sel.groupby("exp_id", sort=True):
        eid_date, eid = _exp_date_and_id(df_exp)
        rows_sorted = sorted(df_exp["row"].dropna().unique())
        cols_sorted = sorted(df_exp["col"].dropna().unique())
        n_rows, n_cols = len(rows_sorted), len(cols_sorted)

        _leg_ratio = 0.2
        fig = plt.figure(
            figsize=(well_size * n_cols * (1 + _leg_ratio), well_size * n_rows),
            layout="constrained",
        )
        fig.get_layout_engine().set(rect=(0, 0, 1, 0.98))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, _leg_ratio], wspace=0.05)
        gs_plate = gs[0, 0].subgridspec(n_rows, n_cols, hspace=0.25, wspace=0.25)
        axes = [
            [fig.add_subplot(gs_plate[i, j]) for j in range(n_cols)]
            for i in range(n_rows)
        ]
        ax_leg = fig.add_subplot(gs[0, 1])
        ax_leg.axis("off")

        fig.suptitle(
            f"Plate grid — {eid_date} {eid}",
            fontsize=11,
            fontweight="bold",
        )

        row_idx = {r: i for i, r in enumerate(rows_sorted)}
        col_idx = {c: i for i, c in enumerate(cols_sorted)}

        # collect per-well data
        for (row, col), df_well in df_exp.groupby(["row", "col"], sort=False):
            ax = axes[row_idx[row]][col_idx[col]]
            df_well = df_well.sort_values("time_value")

            is_blank = bool(df_well["is_blank"].any()) if "is_blank" in df_well else False

            if is_blank:
                ax.set_facecolor("#ddeeff")
                color = "#5588bb"
            else:
                color = "#2266cc"

            ax.plot(df_well["time_value"], df_well[od_col], color=color, lw=0.8)
            ax.set_ylim(bottom=0)
            ax.set_xticks([])
            ax.set_yticks([])

            # row/col labels on outer edges only
            if col_idx[col] == 0:
                ax.set_ylabel(row.upper(), fontsize=9, labelpad=3, rotation=0, va="center")
            if row_idx[row] == 0:
                ax.set_title(str(col), fontsize=9, pad=3)

        # hide axes not covered by well data
        used = {
            (row_idx[r], col_idx[c])
            for r, c in df_exp.groupby(["row", "col"]).groups
        }
        for ri in range(n_rows):
            for ci in range(n_cols):
                if (ri, ci) not in used:
                    axes[ri][ci].set_visible(False)

        leg_lines: list[str] = []
        if "strain_name" in df_exp.columns:
            leg_lines.extend(["Rows → strain_name", ""])
            for r_ in rows_sorted:
                df_r = df_exp.loc[df_exp["row"] == r_]
                if df_r.empty:
                    continue
                sn = str(df_r["strain_name"].iloc[0])
                leg_lines.append(f"  {str(r_).upper()}  →  {sn}")
        if "media_name" in df_exp.columns:
            if leg_lines:
                leg_lines.extend(["", "Cols → media_name", ""])
            else:
                leg_lines.extend(["Cols → media_name", ""])
            for c_ in cols_sorted:
                df_c = df_exp.loc[df_exp["col"] == c_]
                if df_c.empty:
                    continue
                mn = str(df_c["media_name"].iloc[0])
                leg_lines.append(f"  {c_}  →  {mn}")
        if leg_lines:
            ax_leg.text(
                0,
                1,
                "\n".join(leg_lines),
                transform=ax_leg.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                linespacing=1.35,
                family="monospace",
            )

        figs.append(fig)

    return figs


# ---------------------------------------------------------------------------
# Growth parameter plots
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: list[str] = ["max_od", "growth_rate", "lag_time_h"]

_PARAM_LABELS: dict[str, str] = {
    "max_od": "Max OD600",
    "auc": "AUC (OD·h)",
    "growth_rate": "Growth rate (h⁻¹)",
    "lag_time_h": "Lag time (h)",
    "time_to_max_od_h": "Time to max OD (h)",
    "growth_class": "Growth class",
}


def load_curves(path: Union[str, Path]) -> pd.DataFrame:
    """Load the curves CSV produced by ``extract_growth_curves``.

    Parameters
    ----------
    path:
        Path to the ``*_curves.csv`` file produced by ``run_dataset_pipeline.py``.
    """
    return pd.read_csv(path)


def _prep_curves(
    df: pd.DataFrame,
    *,
    exclude_water: bool,
    water_media_name: str,
    exclude_no_growth: bool,
) -> pd.DataFrame:
    df = df.copy()
    if exclude_no_growth and "growth_class" in df.columns:
        df = df[df["growth_class"] == 1]
    if exclude_water and "media_name" in df.columns:
        df = df[df["media_name"] != water_media_name]
    return df


def _filter_param_outliers(df: pd.DataFrame, params: list[str]) -> pd.DataFrame:
    """Drop rows that are outliers using the Tukey IQR fence.
    """
    df = df.copy()
    for col in params:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    return df


def plot_parameter_distributions(
    curves: pd.DataFrame,
    params: list[str] | None = None,
    *,
    x: str = "media_name",
    hue: str | None = None,
    exclude_water: bool = True,
    water_media_name: str = "water",
    exclude_no_growth: bool = True,
    rc: dict | None = None,
    figsize: tuple[float, float] | None = None,
    data_title: str | None = None,
    filter_outliers: bool = True,
) -> Figure:
    """Violin + strip plots of growth parameters grouped by a categorical variable.

    One subplot per parameter; violin bodies show the distribution, overlaid
    strips show individual datapoints.

    Parameters
    ----------
    curves:
        Wide-format DataFrame from ``load_curves``.
    params:
        Parameters to plot.  Defaults to ``["max_od", "growth_rate", "lag_time_h"]``.
    x:
        Column for the x-axis grouping (``"media_name"`` or ``"strain_name"``).
    hue:
        Optional column to split each violin (e.g. ``"atmosphere"`` or
        ``"temperature"``).
    exclude_water:
        Drop rows where ``media_name == water_media_name`` before plotting.
    water_media_name:
        Value of ``media_name`` treated as the water/blank control.
    exclude_no_growth:
        Drop wells with ``growth_class == 0`` before plotting.
    filter_outliers:
        If ``True`` (default), values outside the Tukey IQR fence.
    rc:
        Optional rcParams overrides.
    figsize:
        Override default figure size.

    Returns
    -------
    Figure
    """
    _apply_style(rc)
    df = _prep_curves(
        curves,
        exclude_water=exclude_water,
        water_media_name=water_media_name,
        exclude_no_growth=exclude_no_growth,
    )

    if params is None:
        params = [p for p in _DEFAULT_PARAMS if p in df.columns]

    if filter_outliers:
        df = _filter_param_outliers(df, params)

    x_order = sorted(df[x].dropna().unique())
    n_params = len(params)
    default_w = max(8, len(x_order) * 0.9 + (2 if hue else 0))
    default_h = 3.0 * n_params

    fig, axes = plt.subplots(
        n_params, 1, figsize=figsize or (default_w, default_h), squeeze=False
    )

    for i, param in enumerate(params):
        ax = axes[i, 0]

        sns.violinplot(
            data=df,
            x=x,
            y=param,
            hue=hue,
            order=x_order,
            ax=ax,
            inner="quart",
            cut=0,
            linewidth=0.8,
        )
        sns.stripplot(
            data=df,
            x=x,
            y=param,
            hue=hue,
            order=x_order,
            ax=ax,
            size=2.5,
            alpha=0.5,
            dodge=hue is not None,
            legend=False,
        )

        ax.set_ylabel(_PARAM_LABELS.get(param, param))
        ax.set_xlabel("")
        if i < n_params - 1:
            ax.set_xticklabels([])
        else:
            ax.tick_params(axis="x", rotation=35, labelsize=8)
            for lbl in ax.get_xticklabels():
                lbl.set_ha("right")

        leg = ax.get_legend()
        if hue and i == 0:
            if leg:
                leg.set_title(hue)
        elif leg:
            leg.remove()

    fig.suptitle(f"Growth parameters {data_title}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_growth_heatmap(
    curves: pd.DataFrame,
    *,
    value: str = "max_od",
    exclude_water: bool = True,
    water_media_name: str = "water",
    rc: dict | None = None,
    figsize: tuple[float, float] | None = None,
) -> list[Figure]:
    """Strain × media heatmap of a growth parameter.

    One figure per unique (atmosphere, temperature) condition.  Values are
    averaged across any replicates sharing the same (strain, media, condition).

    Parameters
    ----------
    curves:
        Wide-format DataFrame from ``load_curves``.
    value:
        Parameter to display (e.g. ``"max_od"``, ``"growth_rate"``,
        or ``"growth_class"``).
    exclude_water:
        Drop rows where ``media_name == water_media_name``.
    water_media_name:
        Value of ``media_name`` treated as the water/blank control.
    rc:
        Optional rcParams overrides.
    figsize:
        Override default figure size.

    Returns
    -------
    list[Figure]
        One figure per (atmosphere, temperature) group.
    """
    _apply_style(rc)
    df = _prep_curves(
        curves,
        exclude_water=exclude_water,
        water_media_name=water_media_name,
        exclude_no_growth=False,
    )

    is_binary = value == "growth_class"
    cmap = ListedColormap(["#f5c6c6", "#95d17e"]) if is_binary else "YlGn"
    fmt = "d" if is_binary else ".2f"

    cond_cols = [c for c in ["atmosphere", "temperature"] if c in df.columns]
    groups = list(df.groupby(cond_cols, sort=True)) if cond_cols else [("all", df)]

    figs: list[Figure] = []

    for keys, group in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        cond_label = "  ".join(f"{c}={k}" for c, k in zip(cond_cols, keys))

        pivot = (
            group.groupby(["strain_name", "media_name"])[value]
            .mean()
            .unstack("media_name")
        )
        if is_binary:
            pivot = pivot.round(0).astype(int)

        n_r, n_c = pivot.shape
        default_w = max(6, n_c * 1.0 + 2.5)
        default_h = max(3.5, n_r * 0.7 + 1.5)

        fig, ax = plt.subplots(figsize=figsize or (default_w, default_h))
        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            annot=True,
            fmt=fmt,
            linewidths=0.5,
            linecolor="white",
            vmin=0,
            vmax=1 if is_binary else None,
            cbar_kws={"shrink": 0.7, "label": _PARAM_LABELS.get(value, value)},
        )
        ax.set_title(
            f"{_PARAM_LABELS.get(value, value)} — {cond_label}",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("Media")
        ax.set_ylabel("Strain")
        ax.tick_params(axis="x", rotation=40)
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()
        figs.append(fig)

    return figs


def plot_condition_comparison(
    curves: pd.DataFrame,
    param: str = "max_od",
    *,
    condition: str = "temperature",
    hue: str = "strain_name",
    exclude_water: bool = True,
    water_media_name: str = "water",
    exclude_no_growth: bool = False,
    rc: dict | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Slopegraph comparing a growth parameter across conditions.

    For each (strain, media) pair a line connects the mean parameter value at
    each level of ``condition`` (e.g. 25 °C → 37 °C, aerobic → anaerobic).
    Pairs with missing data at any condition level are skipped.

    Parameters
    ----------
    curves:
        Wide-format DataFrame from ``load_curves``.
    param:
        Parameter to compare (e.g. ``"max_od"`` or ``"growth_rate"``).
    condition:
        Column defining the condition axis (``"temperature"`` or
        ``"atmosphere"``).
    hue:
        Column used to colour lines (``"strain_name"`` or ``"media_name"``).
    exclude_water:
        Drop rows where ``media_name == water_media_name``.
    water_media_name:
        Value of ``media_name`` treated as the water/blank control.
    exclude_no_growth:
        Drop wells with ``growth_class == 0``.
    rc:
        Optional rcParams overrides.
    figsize:
        Override default figure size.

    Returns
    -------
    Figure
    """
    _apply_style(rc)
    df = _prep_curves(
        curves,
        exclude_water=exclude_water,
        water_media_name=water_media_name,
        exclude_no_growth=exclude_no_growth,
    )

    cond_vals = sorted(df[condition].dropna().unique())
    x_labels = [str(v) for v in cond_vals]
    hue_vals = sorted(df[hue].dropna().unique())
    colors = {v: c for v, c in zip(hue_vals, _color_cycle(len(hue_vals)))}

    fig, ax = plt.subplots(figsize=figsize or (max(5, len(cond_vals) * 2.5), 5))

    for _, pair in df.groupby(["strain_name", "media_name"]):
        hue_val = pair[hue].iloc[0]
        series = pair.groupby(condition)[param].mean().reindex(cond_vals)
        if series.isna().any():
            continue
        ax.plot(
            x_labels,
            series.values,
            color=colors[hue_val],
            lw=1.0,
            alpha=0.4,
            marker="o",
            markersize=4,
        )

    handles = [
        plt.Line2D([0], [0], color=colors[v], lw=2, label=str(v))
        for v in hue_vals
    ]
    ax.legend(
        handles=handles,
        title=hue,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
    )
    ax.set_xlabel(condition)
    ax.set_ylabel(_PARAM_LABELS.get(param, param))
    ax.set_title(
        f"{_PARAM_LABELS.get(param, param)} by {condition}",
        fontsize=11,
        fontweight="bold",
    )
    ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def plot_parameter_correlation(
    curves: pd.DataFrame,
    *,
    params: list[str] | None = None,
    exclude_water: bool = True,
    water_media_name: str = "water",
    exclude_no_growth: bool = True,
    rc: dict | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Pairwise Pearson correlation matrix of growth parameters.

    Only the lower triangle (including diagonal) is shown; the strict upper
    triangle is masked to avoid redundancy.

    Parameters
    ----------
    curves:
        Wide-format DataFrame from ``load_curves``.
    params:
        Parameters to include.  Defaults to all numeric parameter columns
        present in ``curves``.
    exclude_water:
        Drop rows where ``media_name == water_media_name``.
    water_media_name:
        Value of ``media_name`` treated as the water/blank control.
    exclude_no_growth:
        Drop wells with ``growth_class == 0`` before computing correlations.
    rc:
        Optional rcParams overrides.
    figsize:
        Override default figure size.

    Returns
    -------
    Figure
    """
    _apply_style(rc)
    df = _prep_curves(
        curves,
        exclude_water=exclude_water,
        water_media_name=water_media_name,
        exclude_no_growth=exclude_no_growth,
    )

    _all_numeric = ["max_od", "auc", "growth_rate", "lag_time_h", "time_to_max_od_h"]
    if params is None:
        params = [p for p in _all_numeric if p in df.columns]

    corr = df[params].corr(method="pearson")
    labels = [_PARAM_LABELS.get(p, p) for p in params]
    n = len(params)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=figsize or (max(5, n * 1.3), max(4, n * 1.1)))
    sns.heatmap(
        corr,
        ax=ax,
        mask=mask,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
    )
    ax.set_title("Growth parameter correlations", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=40)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return fig


def plot_growth_scatter(
    curves: pd.DataFrame,
    *,
    x: str = "lag_time_h",
    y: str = "growth_rate",
    hue: str = "media_name",
    exclude_water: bool = True,
    water_media_name: str = "water",
    exclude_no_growth: bool = True,
    rc: dict | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Scatter plot of two growth parameters.

    Useful for exploring the growth-rate / lag-time trade-off across media or
    strains.

    Parameters
    ----------
    curves:
        Wide-format DataFrame from ``load_curves``.
    x:
        Parameter for the x-axis (default ``"lag_time_h"``).
    y:
        Parameter for the y-axis (default ``"growth_rate"``).
    hue:
        Column used to colour points (``"media_name"`` or ``"strain_name"``).
    exclude_water:
        Drop rows where ``media_name == water_media_name``.
    water_media_name:
        Value of ``media_name`` treated as the water/blank control.
    exclude_no_growth:
        Drop wells with ``growth_class == 0``.
    rc:
        Optional rcParams overrides.
    figsize:
        Override default figure size.

    Returns
    -------
    Figure
    """
    _apply_style(rc)
    df = _prep_curves(
        curves,
        exclude_water=exclude_water,
        water_media_name=water_media_name,
        exclude_no_growth=exclude_no_growth,
    )

    hue_vals = sorted(df[hue].dropna().unique())
    colors = {v: c for v, c in zip(hue_vals, _color_cycle(len(hue_vals)))}

    fig, ax = plt.subplots(figsize=figsize or (7, 5))

    for hue_val, grp in df.groupby(hue, sort=True):
        ax.scatter(
            grp[x],
            grp[y],
            color=colors[hue_val],
            label=hue_val,
            s=45,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
        )

    ax.set_xlabel(_PARAM_LABELS.get(x, x))
    ax.set_ylabel(_PARAM_LABELS.get(y, y))
    ax.set_title(
        f"{_PARAM_LABELS.get(y, y)} vs {_PARAM_LABELS.get(x, x)}",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(title=hue, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_ranking(
    curves: pd.DataFrame,
    *,
    value: str = "growth_class",
    group_by: str = "media_name",
    exclude_water: bool = True,
    water_media_name: str = "water",
    rc: dict | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Horizontal bar chart ranking groups by mean growth parameter.

    Answers "which media support the broadest growth?" when
    ``value="growth_class"`` (fraction of strains growing per medium), or
    "which strains are most versatile?" when ``group_by="strain_name"``.

    Parameters
    ----------
    curves:
        Wide-format DataFrame from ``load_curves``.
    value:
        Parameter to aggregate per group (default ``"growth_class"`` → fraction
        growing).
    group_by:
        Column to rank (``"media_name"`` or ``"strain_name"``).
    exclude_water:
        Drop rows where ``media_name == water_media_name``.
    water_media_name:
        Value of ``media_name`` treated as the water/blank control.
    rc:
        Optional rcParams overrides.
    figsize:
        Override default figure size.

    Returns
    -------
    Figure
    """
    _apply_style(rc)
    df = _prep_curves(
        curves,
        exclude_water=exclude_water,
        water_media_name=water_media_name,
        exclude_no_growth=False,
    )

    ranked = df.groupby(group_by)[value].mean().sort_values(ascending=True)
    offset = max(ranked.max() * 0.01, 0.005)

    colors = _color_cycle(len(ranked))
    default_h = max(4, len(ranked) * 0.45)
    fig, ax = plt.subplots(figsize=figsize or (7, default_h))

    bars = ax.barh(
        ranked.index, ranked.values, color=colors, edgecolor="white", linewidth=0.5
    )
    for bar, val in zip(bars, ranked.values):
        ax.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    label = _PARAM_LABELS.get(value, value)
    ax.set_xlabel(f"Mean {label}")
    ax.set_ylabel(group_by.replace("_", " ").title())
    ax.set_title(
        f"{group_by.replace('_', ' ').title()} ranked by {label}",
        fontsize=11,
        fontweight="bold",
    )
    if value == "growth_class":
        ax.set_xlim(0, 1.2)
        ax.axvline(0.5, color="grey", lw=0.8, ls="--", alpha=0.7)
    ax.set_ylim(-0.5, len(ranked) - 0.5)
    fig.tight_layout()
    return fig