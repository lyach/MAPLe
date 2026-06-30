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
from matplotlib import cm
from matplotlib.colors import ListedColormap, Normalize
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
    od_col: str = "od600_raw",
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
        OD column to plot (``"od600"`` or ``"od600_raw"``).
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
    """Mask outlier values as NaN using the Tukey IQR fence."""
    df = df.copy()
    for col in params:
        if col not in df.columns:
            continue
        values = df[col]
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        outlier_mask = values.notna() & ((values < lo) | (values > hi))
        n_outliers = int(outlier_mask.sum())
        if n_outliers:
            df.loc[outlier_mask, col] = np.nan
        print(
            f"Filtered {n_outliers} outliers in {col!r} "
            f"(IQR fence [{lo:.4g}, {hi:.4g}])"
        )
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
    filter_outliers: bool = False,
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

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def load_align_report(path: Union[str, Path]) -> pd.DataFrame:
    """Load the AlignBio report CSV produced by ``create_align_df.py``.

    Parameters
    ----------
    path:
        Path to the ``.csv`` report file.
    """
    return pd.read_csv(path)


def _prep_align_csv(
    df: pd.DataFrame,
    *,
    growth_threshold: float,
) -> pd.DataFrame:
    """Return a copy with a boolean ``grows`` column."""
    df = df.copy()
    df["grows"] = df["MaxOD"] > growth_threshold
    return df


def _shorten_organism(name: str, max_len: int = 22) -> str:
    """Abbreviate long organism names to keep axis labels readable."""
    parts = name.split()
    if len(parts) >= 2:
        abbrev = f"{parts[0][0].upper()}. {' '.join(parts[1:])}"
        return abbrev[:max_len] if len(abbrev) > max_len else abbrev
    return name[:max_len]


MEDIA_MAP: dict[str, str] = {
    "lb": "LB Broth",
    "mb": "Marine Broth",
    "nb": "Nutrient Broth",
    "bhi": "BHI Broth",
    "mrs": "MRS Broth",
    "thy": "TH+Y Broth",
    "tsb": "Tryptic Soy Broth",
    "gifu": "GAM Broth",
    "mrc": "MRC Broth",
    "cdm": "CDM",
    "m9": "M9",
}


def _map_media(name: str) -> str:
    return MEDIA_MAP.get(str(name).lower(), str(name))


def _prep_curves_csv(df: pd.DataFrame, 
                     growth_threshold: float = 0.1) -> pd.DataFrame:
    df = df.copy()
    # drop empty or other organisms
    df = df.drop(df[(df["strain_name"].str.contains("empty")) | (df["strain_name"].str.contains("other"))].index)
    # drop water media
    df = df.drop(df[df["media_name"].str.contains("water")].index)
    # rename columns
    df = df.rename(columns={"strain_name": "Organism", 
                            "media_name": "Media",
                            "max_od": "MaxOD"})
    df["Media"] = df["Media"].map(_map_media)
    print(f"Prepared Curves DataFrame has {len(df)} rows")
    # drop rows with MaxOD below growth threshold
    df["grows"] = df["MaxOD"] > growth_threshold
    print(f"Prepared Curves DataFrame has {len(df)} rows after growth threshold removal")
    return df

# ---------------------------------------------------------------------------
# MaxOD600 plots
# ---------------------------------------------------------------------------

def plot_maxod_heatmap(
    df: pd.DataFrame,
    *,
    align_df: bool = False,
    facet_col: str | None = None,
    growth_threshold: float = 0.1,
    annot: bool = True,
    cmap: str = "YlOrRd",
    filter_param_outliers: bool = False,
    rc: dict | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Organism × Media heatmap of mean MaxOD600.

    Rows and columns are sorted by their overall mean MaxOD so the
    best performing combinations appear in the top right corner.

    Parameters
    ----------
    df:
        DataFrame from curves.csv or align.csv
    align_df:
        Whether the input dataframe is the AlignBio DataFrame.
    facet_col:
        If given (e.g. ``"Temperature"``), one subplot column is drawn per
        unique value so conditions can be compared side by side.
    growth_threshold:
        Values below this threshold are masked (grey) to distinguish true
        growth from noise.
    annot:
        Annotate each cell with its value.
    cmap:
        Matplotlib colourmap name.
    filter_param_outliers:
        If ``True``, values outside the Tukey IQR fence for ``MaxOD`` are
        masked before aggregation.
    rc:
        Optional rcParams overrides.
    figsize:
        Override default figure size.

    Returns
    -------
    Figure
    """
    _apply_style(rc)
    if align_df:
        df = _prep_align_csv(df, growth_threshold=growth_threshold)
    else:
        df = _prep_curves_csv(df)

    if filter_param_outliers:
        df = _filter_param_outliers(df, ["MaxOD"])

    facet_vals = sorted(df[facet_col].unique()) if facet_col else [None]
    n_facets = len(facet_vals)

    organisms = (
        df.groupby("Organism")["MaxOD"].mean().sort_values(ascending=False).index.tolist()
    )
    medias = (
        df.groupby("Media")["MaxOD"].mean().sort_values(ascending=False).index.tolist()
    )
    org_labels = [_shorten_organism(o) for o in organisms]

    default_w = max(10, len(medias) * 0.9 * n_facets)
    default_h = max(6, len(organisms) * 0.45)
    fig, axes = plt.subplots(
        1,
        n_facets,
        figsize=figsize or (default_w, default_h),
        sharey=True,
        layout="constrained",
    )
    if n_facets == 1:
        axes = [axes]

    vmax = df["MaxOD"].quantile(0.98)

    for ax, fval in zip(axes, facet_vals):
        sub = df if fval is None else df[df[facet_col] == fval]
        pivot = (
            sub.groupby(["Organism", "Media"])["MaxOD"]
            .mean()
            .unstack("Media")
            .reindex(index=organisms, columns=medias)
        )
        mask = pivot < growth_threshold

        sns.heatmap(
            pivot,
            ax=ax,
            mask=mask,
            cmap=cmap,
            vmin=growth_threshold,
            vmax=vmax,
            annot=annot,
            fmt=".2f",
            annot_kws={"size": 7},
            linewidths=0.4,
            linecolor="white",
            cbar=False,
        )
        # grey out sub-threshold cells
        sns.heatmap(
            pivot,
            ax=ax,
            mask=~mask,
            cmap=ListedColormap(["#e0e0e0"]),
            annot=annot,
            fmt=".2f",
            annot_kws={"size": 7, "color": "#aaaaaa"},
            linewidths=0.4,
            linecolor="white",
            cbar=False,
            vmin=0,
            vmax=1,
        )
        title = f"{facet_col} = {fval}" if fval is not None else "Mean MaxOD600"
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Media", fontsize=9)
        ax.set_ylabel("Organism" if ax is axes[0] else "", fontsize=9)
        ax.set_yticklabels(org_labels, rotation=0, fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)

    sm = cm.ScalarMappable(
        cmap=cmap,
        norm=Normalize(vmin=growth_threshold, vmax=vmax),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.6)
    return fig


def plot_maxod_distribution(
    df: pd.DataFrame,
    *,
    by: str = "Media",
    growth_threshold: float = 0.1,
    hue: str | None = "Organism",
    rc: dict | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Strip + box plot of MaxOD600 grouped by media or organism.

    Groups are sorted by median MaxOD (highest at top) for quick visual
    ranking.  Individual observations are overlaid as a strip to reveal
    organism-level variability within each medium.

    Parameters
    ----------
    df:
        AlignBio report DataFrame from ``load_align_report``.
    by:
        Column to use as the primary grouping axis — ``"Media"`` or
        ``"Organism"``.
    growth_threshold:
        Horizontal reference line drawn at this value.
    hue:
        Column used to colour the strip markers.  Set to ``None`` to
        use a single colour.
    rc:
        Optional rcParams overrides.
    figsize:
        Override default figure size.

    Returns
    -------
    Figure
    """
    _apply_style(rc)

    order = (
        df.groupby(by)["MaxOD"].median().sort_values(ascending=True).index.tolist()
    )
    default_h = max(5, len(order) * 0.5)
    fig, ax = plt.subplots(figsize=figsize or (9, default_h))

    sns.boxplot(
        data=df,
        y=by,
        x="MaxOD",
        order=order,
        color="#d6e4f0",
        width=0.5,
        fliersize=0,
        linewidth=0.8,
        ax=ax,
    )

    hue_vals = sorted(df[hue].dropna().unique()) if hue else [None]
    palette = dict(zip(hue_vals, _color_cycle(len(hue_vals)))) if hue else None

    sns.stripplot(
        data=df,
        y=by,
        x="MaxOD",
        order=order,
        hue=hue if hue else None,
        palette=palette,
        size=5,
        alpha=0.7,
        jitter=True,
        dodge=False,
        ax=ax,
    )

    ax.axvline(growth_threshold, color="crimson", lw=1.0, ls="--", alpha=0.8,
               label=f"growth threshold ({growth_threshold})")
    ax.set_xlabel("MaxOD600", fontsize=9)
    ax.set_ylabel(by, fontsize=9)
    ax.set_title(
        f"MaxOD600 distribution by {by}",
        fontsize=11,
        fontweight="bold",
    )
    if hue:
        ax.legend(
            title=hue,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=7,
            markerscale=0.9,
        )
    else:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_maxod_ranking(
    df: pd.DataFrame,
    *,
    group_by: str = "Media",
    facet_col: str = "atmosphere",
    growth_threshold: float = 0.1,
    rc: dict | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Horizontal bar chart ranking groups by mean MaxOD600 ± std.

    One subplot is drawn per unique ``facet_col`` value (default: atmosphere)
    so aerobic and anaerobic conditions can be compared side by side.

    Parameters
    ----------
    df:
        DataFrame from curves.csv.
    group_by:
        Column to rank — ``"Media"`` or ``"Organism"``.
    facet_col:
        Column to facet on (default ``"atmosphere"``).
    growth_threshold:
        Passed to ``_prep_curves_csv`` to flag sub-threshold wells.
    rc:
        Optional rcParams overrides.
    figsize:
        Override default figure size.

    Returns
    -------
    Figure
    """
    _apply_style(rc)
    df = _prep_curves_csv(df, growth_threshold=growth_threshold)

    facet_vals = sorted(df[facet_col].dropna().unique())
    n_facets = len(facet_vals)

    groups = (
        df.groupby(group_by)["MaxOD"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )
    y_labels = (
        [_shorten_organism(g) for g in groups]
        if group_by == "Organism"
        else groups
    )
    colors = dict(zip(groups, _color_cycle(len(groups))))
    y_pos = np.arange(len(groups))

    default_w = max(7, 5.0 * n_facets)
    default_h = max(4, len(groups) * 0.5)
    fig, axes = plt.subplots(
        1,
        n_facets,
        figsize=figsize or (default_w, default_h),
        sharey=True,
        sharex=True,
        layout="constrained",
    )
    if n_facets == 1:
        axes = [axes]

    x_max = 0.0
    for ax, fval in zip(axes, facet_vals):
        sub = df[df[facet_col] == fval]
        agg = (
            sub.groupby(group_by)["MaxOD"]
            .agg(["mean", "std"])
            .reindex(groups)
        )
        bars = ax.barh(
            y_pos,
            agg["mean"],
            xerr=agg["std"],
            color=[colors[g] for g in groups],
            edgecolor="white",
            linewidth=0.5,
            error_kw={"elinewidth": 0.8, "capsize": 3, "ecolor": "#555555"},
        )
        facet_x_max = (agg["mean"] + agg["std"].fillna(0)).max()
        if pd.notna(facet_x_max):
            x_max = max(x_max, facet_x_max)
        offset = max(facet_x_max * 0.01, 0.005) if pd.notna(facet_x_max) else 0.005
        for bar, (_, row) in zip(bars, agg.iterrows()):
            if pd.isna(row["mean"]):
                continue
            ax.text(
                row["mean"] + (row["std"] if pd.notna(row["std"]) else 0) + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{row['mean']:.2f}",
                va="center",
                ha="left",
                fontsize=8,
            )

        ax.set_title(f"{facet_col} = {fval}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean MaxOD600 ± std", fontsize=9)
        ax.set_ylim(-0.5, len(groups) - 0.5)

    axes[0].set_ylabel(group_by, fontsize=9)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(y_labels, fontsize=8)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    if x_max:
        for ax in axes:
            ax.set_xlim(0, x_max * 1.15)

    fig.suptitle(
        f"{group_by} ranked by mean MaxOD600",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    return fig


def plot_maxod_temperature_effect(
    df: pd.DataFrame,
    *,
    facet_by: str | None = "Media",
    col_wrap: int = 4,
    rc: dict | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Mean MaxOD600 vs Temperature, one line per Organism.

    Shows how growth yield responds to temperature for each organism.
    When ``facet_by`` is set, one subplot is drawn per unique value of
    that column (e.g., per medium).

    Parameters
    ----------
    df:
        AlignBio report DataFrame from ``load_align_report``.
    facet_by:
        Column to facet on (``"Media"`` or ``None`` for a single panel
        averaged across all media).
    col_wrap:
        Number of subplot columns before wrapping.
    rc:
        Optional rcParams overrides.
    figsize:
        Override default figure size.

    Returns
    -------
    Figure
    """
    _apply_style(rc)

    organisms = sorted(df["Organism"].unique())
    org_labels = {o: _shorten_organism(o) for o in organisms}
    colors = dict(zip(organisms, _color_cycle(len(organisms))))
    temps = sorted(df["Temperature"].unique())

    if facet_by is None:
        agg = (
            df.groupby(["Organism", "Temperature"])["MaxOD"]
            .mean()
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=figsize or (7, 5))
        for org, grp in agg.groupby("Organism", sort=True):
            grp = grp.sort_values("Temperature")
            ax.plot(
                grp["Temperature"],
                grp["MaxOD"],
                marker="o",
                color=colors[org],
                label=org_labels[org],
                linewidth=1.4,
            )
        ax.set_xlabel("Temperature (°C)", fontsize=9)
        ax.set_ylabel("Mean MaxOD600", fontsize=9)
        ax.set_title("Temperature effect on MaxOD600", fontsize=11, fontweight="bold")
        ax.set_xticks(temps)
        ax.legend(title="Organism", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        fig.tight_layout()
        return fig

    facet_vals = sorted(df[facet_by].unique())
    n_cols = min(col_wrap, len(facet_vals))
    n_rows = int(np.ceil(len(facet_vals) / n_cols))
    default_w = n_cols * 3.5
    default_h = n_rows * 3.2
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize or (default_w, default_h), sharey=True
    )
    axes_flat = np.array(axes).flatten()

    agg = (
        df.groupby(["Organism", facet_by, "Temperature"])["MaxOD"]
        .mean()
        .reset_index()
    )

    for i, fval in enumerate(facet_vals):
        ax = axes_flat[i]
        sub = agg[agg[facet_by] == fval]
        for org, grp in sub.groupby("Organism", sort=True):
            grp = grp.sort_values("Temperature")
            ax.plot(
                grp["Temperature"],
                grp["MaxOD"],
                marker="o",
                markersize=5,
                color=colors[org],
                linewidth=1.2,
            )
        ax.set_title(str(fval), fontsize=9, fontweight="bold")
        ax.set_xticks(temps)
        ax.set_xlabel("°C", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    handles = [
        plt.Line2D([0], [0], color=colors[o], marker="o", markersize=5, linewidth=1.2)
        for o in organisms
    ]
    fig.legend(
        handles,
        [org_labels[o] for o in organisms],
        title="Organism",
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        fontsize=7,
        ncol=1,
    )
    fig.supxlabel("Temperature (°C)", fontsize=9)
    fig.supylabel("Mean MaxOD600", fontsize=9)
    fig.suptitle(
        f"Temperature effect on MaxOD600  (faceted by {facet_by})",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_growth_breadth(
    df: pd.DataFrame,
    *,
    growth_threshold: float = 0.1,
    rc: dict | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Count of compatible media/organisms above the growth threshold.

    Left panel: number of media where each organism grows (mean MaxOD >
    ``growth_threshold``) — organism versatility.
    Right panel: number of organisms that grow in each medium — medium
    permissiveness.

    Parameters
    ----------
    df:
        AlignBio report DataFrame from ``load_align_report``.
    growth_threshold:
        MinOD considered as genuine growth.
    rc:
        Optional rcParams overrides.
    figsize:
        Override default figure size.

    Returns
    -------
    Figure
    """
    _apply_style(rc)

    mean_od = df.groupby(["Organism", "Media"])["MaxOD"].mean().reset_index()
    mean_od["grows"] = mean_od["MaxOD"] > growth_threshold

    org_breadth = (
        mean_od.groupby("Organism")["grows"].sum().sort_values(ascending=True)
    )
    media_breadth = (
        mean_od.groupby("Media")["grows"].sum().sort_values(ascending=True)
    )

    n_media = len(mean_od["Media"].unique())

    default_h = max(5, max(len(org_breadth), len(media_breadth)) * 0.45)
    fig, (ax_org, ax_med) = plt.subplots(
        1, 2, figsize=figsize or (12, default_h)
    )

    # Organism versatility
    colors_org = _color_cycle(len(org_breadth))
    ax_org.barh(
        [_shorten_organism(o) for o in org_breadth.index],
        org_breadth.values,
        color=colors_org,
        edgecolor="white",
        linewidth=0.5,
    )
    ax_org.axvline(n_media, color="grey", lw=0.8, ls=":", alpha=0.7,
                   label=f"all {n_media} media")
    ax_org.set_xlabel("# media supporting growth", fontsize=9)
    ax_org.set_ylabel("Organism", fontsize=9)
    ax_org.set_title("Organism versatility", fontsize=10, fontweight="bold")
    ax_org.tick_params(axis="y", labelsize=8)
    ax_org.set_xlim(0, n_media + 0.5)
    ax_org.legend(fontsize=8)

    # Media permissiveness
    n_org = len(mean_od["Organism"].unique())
    colors_med = _color_cycle(len(media_breadth))
    ax_med.barh(
        media_breadth.index,
        media_breadth.values,
        color=colors_med,
        edgecolor="white",
        linewidth=0.5,
    )
    ax_med.axvline(n_org, color="grey", lw=0.8, ls=":", alpha=0.7,
                   label=f"all {n_org} organisms")
    ax_med.set_xlabel("# organisms that grow", fontsize=9)
    ax_med.set_ylabel("Media", fontsize=9)
    ax_med.set_title("Media permissiveness", fontsize=10, fontweight="bold")
    ax_med.tick_params(axis="y", labelsize=8)
    ax_med.set_xlim(0, n_org + 0.5)
    ax_med.legend(fontsize=8)

    fig.suptitle(
        f"Growth breadth  (threshold MaxOD > {growth_threshold})",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Growth rate plots
# ---------------------------------------------------------------------------
def plot_growth_rate_violin(
    df: pd.DataFrame,
    organisms: list[str],
    *,
    growth_threshold: float = 0.1,
    exclude_no_growth: bool = False,
    filter_param_outliers: bool = False,
    rc: dict | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Violin plot of growth rate for selected organisms, faceted by atmosphere.

    One vertically stacked panel is drawn per unique ``atmosphere`` value.
    Each violin summarises the per-well ``growth_rate`` distribution for one
    organism across all media in that atmosphere.

    Parameters
    ----------
    df:
        Curves CSV with ``strain_name``, ``media_name``, ``max_od``,
        ``growth_rate``, and ``atmosphere`` columns.
    organisms:
        Organism names to include (``Organism`` column after normalisation).
    growth_threshold:
        Passed to ``_prep_curves_csv`` to flag sub-threshold wells.
    exclude_no_growth:
        If ``True``, drop wells with ``grows == False`` before plotting.
    filter_param_outliers:
        If ``True``, values outside the Tukey IQR fence for ``growth_rate``
        are masked before plotting.
    rc:
        Optional rcParams overrides.
    figsize:
        Override default figure size.

    Returns
    -------
    Figure
    """
    _apply_style(rc)

    df = _prep_curves_csv(df, growth_threshold=growth_threshold)
    if "growth_rate" not in df.columns:
        raise ValueError("Input df must contain a growth_rate column.")
    if exclude_no_growth:
        df = df[df["grows"]]

    if filter_param_outliers:
        df = _filter_param_outliers(df, ["growth_rate"])

    df = df[df["Organism"].isin(organisms)].dropna(subset=["growth_rate"])
    if df.empty:
        raise ValueError("No growth_rate data for the requested organisms.")

    atm_vals = sorted(df["atmosphere"].dropna().unique())
    org_order = [o for o in organisms if o in df["Organism"].unique()]
    org_labels = [_shorten_organism(o) for o in org_order]
    palette = dict(zip(org_order, _color_cycle(len(org_order))))

    default_w = max(8, len(org_order) * 1.2)
    default_h = max(4, 3.0 * len(atm_vals))
    fig, axes = plt.subplots(
        len(atm_vals),
        1,
        figsize=figsize or (default_w, default_h),
        sharex=True,
        layout="constrained",
    )
    if len(atm_vals) == 1:
        axes = [axes]

    for ax, atm in zip(axes, atm_vals):
        sub = df[df["atmosphere"] == atm]
        sns.violinplot(
            data=sub,
            x="Organism",
            y="growth_rate",
            order=org_order,
            hue="Organism",
            palette=palette,
            dodge=False,
            ax=ax,
            inner="quart",
            cut=0,
            linewidth=0.8,
            legend=False,
        )
        sns.stripplot(
            data=sub,
            x="Organism",
            y="growth_rate",
            order=org_order,
            hue="Organism",
            palette=palette,
            dodge=False,
            ax=ax,
            size=4,
            alpha=0.55,
            jitter=0.15,
            legend=False,
        )
        ax.set_title(f"atmosphere = {atm}", fontsize=10, fontweight="bold")
        ax.set_ylabel(_PARAM_LABELS["growth_rate"], fontsize=9)
        ax.set_xlabel("")
        if ax is axes[-1]:
            ax.set_xticks(range(len(org_order)))
            ax.set_xticklabels(org_labels, rotation=35, ha="right", fontsize=8)
        else:
            ax.tick_params(axis="x", labelbottom=False)

    fig.suptitle(
        "Growth rate by organism",
        fontsize=12,
        fontweight="bold",
    )
    return fig