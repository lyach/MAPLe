"""Growth curve parameter extraction.

Extracts per-well growth parameters (max OD, AUC, growth rate, lag time, etc.)
from OD600 time series. Optionally averages across replicates.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def _trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration; NumPy 2 uses ``trapezoid``, older uses ``trapz``."""
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    if trap is None:
        raise ImportError("NumPy must provide trapezoid or trapz")
    return float(trap(y, x))


# Smoothing functions

def smooth_moving_average(od: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered moving average with edge replicated padding."""
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(od, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(od)]


def smooth_savgol(
    od: np.ndarray, window_length: int = 7, polyorder: int = 3
) -> np.ndarray:
    """Savitzky-Golay filter. Adjusts window for short series."""
    wl = min(window_length, len(od))
    if wl % 2 == 0:
        wl -= 1
    if wl < polyorder + 2:
        return od.copy()
    return savgol_filter(od, window_length=wl, polyorder=polyorder)


SMOOTHERS: dict[str, Callable] = {
    "moving_average": smooth_moving_average,
    "savgol": smooth_savgol,
}


# Single-curve parameter extraction

_LOG_EPS = 1e-6
_SLIDE_WINDOW = 4


def extract_params(time: np.ndarray, od: np.ndarray) -> dict:
    """Extract growth parameters from a single smoothed OD600 curve.

    Parameters
    ----------
    time:
        Time values (hours).
    od:
        Smoothed, blank-subtracted OD600 values aligned with *time*.

    Returns
    -------
    dict with keys: max_od, auc, growth_rate, lag_time_h, time_to_max_od_h.
    """
    # max od600
    max_od = float(np.max(od))
    # auc
    auc = _trapz_compat(od, time)
    # time to max od600
    time_to_max_od_h = float(time[np.argmax(od)])

    # growth rate
    # Use log(OD + eps) so zero/negative corrected OD values stay finite.
    log_od = np.log(np.maximum(od, 0.0) + _LOG_EPS)
    best_slope = -np.inf
    best_intercept = 0.0
    best_t_start = time[0]

    # Fit each 4-point log-phase window and keep the steepest slope.
    for i in range(len(log_od) - _SLIDE_WINDOW + 1):
        t_win = time[i : i + _SLIDE_WINDOW]
        y_win = log_od[i : i + _SLIDE_WINDOW]
        slope, intercept = np.polyfit(t_win, y_win, 1)
        if slope > best_slope:
            best_slope = slope
            best_intercept = intercept
            best_t_start = t_win[0]

    growth_rate = float(max(best_slope, 0.0))

    # lag time
    baseline_log = float(log_od[0])
    if growth_rate > 0:
        # Extrapolate the best tangent back to the initial log OD baseline.
        lag_time_h = float((baseline_log - best_intercept) / best_slope)
        # Clamp to non-negative lag time.
        lag_time_h = max(lag_time_h, 0.0)
    # If no growth rate is detected, set lag time to the last timepoint.
    else:
        lag_time_h = float(time[-1])

    return {
        "max_od": max_od,
        "auc": auc,
        "growth_rate": growth_rate,
        "lag_time_h": lag_time_h,
        "time_to_max_od_h": time_to_max_od_h,
    }


# Growth-class threshold

def compute_growth_threshold(df: pd.DataFrame) -> pd.Series:
    """Compute per-experiment growth/no-growth threshold.

    Threshold = 2 * std(blank od600_raw) applied on blank-subtracted od600
    (the mean component cancels after blank subtraction).

    Returns a Series indexed by ``exp_id``.
    """
    od_col = "od600_raw" if "od600_raw" in df.columns else "od600"
    blank = df.loc[df["is_blank"]]

    threshold = (
        blank.groupby("exp_id")[od_col]
        .std()
        .mul(2)
        .fillna(0.0)
        .rename("growth_threshold")
    )
    return threshold


# Main entry point

_ID_COLS = ["exp_id", "date", "strain_name", "media_name", "atmosphere", "temperature"]
_REPLICATE_GROUP = ["strain_name", "media_name", "atmosphere", "temperature"]
_PARAM_COLS = ["max_od", "auc", "growth_rate", "lag_time_h", "time_to_max_od_h", "growth_class"]


def extract_growth_curves(
    df: pd.DataFrame,
    smoother: str = "savgol",
    smoother_kwargs: dict | None = None,
    with_replicates: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract growth parameters from long-format DataFrame post QC.

    Parameters
    ----------
    df:
        Long-format DataFrame produced after ``run_qc``.
    smoother:
        Key in ``SMOOTHERS`` — ``"savgol"`` or ``"moving_average"``.
    smoother_kwargs:
        Extra keyword arguments forwarded to the smoother function.
    with_replicates:
        If ``True``, average parameters across replicates sharing
        ``(strain_name, media_name, atmosphere, temperature)``.

    Returns
    -------
    tuple of (wide_df, long_df):

    wide_df
        Wide-format DataFrame with one row per well and growth parameter
        columns.
    long_df
        Copy of the input *df* with an additional ``od600_smooth`` column.
    """
    if smoother not in SMOOTHERS:
        raise ValueError(
            f"Unknown smoother {smoother!r}. Choose from {list(SMOOTHERS)}"
        )
    smooth_fn = SMOOTHERS[smoother]
    smooth_kw = smoother_kwargs or {}

    samples = df.loc[~df["is_blank"]].copy()

    thresholds = compute_growth_threshold(df)

    long_out = df.copy()
    long_out["od600_smooth"] = np.nan

    records: list[dict] = []

    for (exp_id, well_id), grp in samples.groupby(["exp_id", "well_id"], sort=False):
        grp = grp.sort_values("timepoint")
        time_raw = grp["time_value"].to_numpy(dtype=float)
        time_unit = grp["time_unit"].iloc[0] if "time_unit" in grp.columns else "h"
        time = time_raw / 60.0 if time_unit == "min" else time_raw  # convert minutes to hours
        od = grp["od600"].to_numpy(dtype=float)

        od_smooth = smooth_fn(od, **smooth_kw)

        long_out.loc[grp.index, "od600_smooth"] = od_smooth

        params = extract_params(time, od_smooth)

        thresh = thresholds.get(exp_id, 0.0)
        params["growth_class"] = int(params["max_od"] > thresh)

        meta = grp.iloc[0]
        row = {col: meta[col] for col in _ID_COLS if col in meta.index}
        if "replicate" in meta.index:
            row["replicate"] = meta["replicate"]
        row.update(params)
        records.append(row)

    result = pd.DataFrame(records)

    if with_replicates:
        agg_map = {p: "mean" for p in _PARAM_COLS if p != "growth_class"}
        agg_map["growth_class"] = lambda s: int(s.mean() >= 0.5)
        result = (
            result.groupby(_REPLICATE_GROUP, sort=False)
            .agg(agg_map)
            .reset_index()
        )

    return result, long_out
