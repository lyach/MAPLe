"""Growth curve parameter extraction.

Processes each time-series well od600 reads as:

    ``od600`` ──► hampel_filter ──► ``od600_clean`` ──► smoother ──► ``od600_smooth``
                                                                    │
                                                                    └► extract_params
                                                                        (max_od, auc,
                                                                        growth_rate,
                                                                        lag_time_h,
                                                                        time_to_max_od_h)

A Hampel filter  handles experimental artifacts (condensation, bubbles) 
across the whole curve. Smoothing is then applied curve, and every 
growth parameter is computed from the filtered and smoothed array.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger
from scipy.signal import savgol_filter


_MAD_SCALE = 1.4826
_HAMPEL_WINDOW = 7
_HAMPEL_N_SIGMAS = 3.0


def _trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration; NumPy 2 uses ``trapezoid``, older uses ``trapz``."""
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    if trap is None:
        raise ImportError("NumPy must provide trapezoid or trapz")
    return float(trap(y, x))


# -----------------------------------------------------------------------------
# Despiking (Hampel filter)
# -----------------------------------------------------------------------------

def hampel_filter(
    od: np.ndarray,
    window: int = _HAMPEL_WINDOW,
    n_sigmas: float = _HAMPEL_N_SIGMAS,
) -> tuple[np.ndarray, np.ndarray]:
    """Spike removal using the Hampel filter.

    For each timepoint the local median and MAD are computed within a
    centered window (clamped at the edges). A point is flagged when

        ``|x_i - median_local| > n_sigmas * 1.4826 * MAD_local``

    Flagged points are replaced by ``median_local``.

    Handles early and late spikes in the reads.

    Parameters
    ----------
    od:
        1-D OD600 array, sorted by time.
    window:
        Centered window size (odd values recommended; clamped at edges).
    n_sigmas:
        Outlier threshold in modified-z-score units (3.0 ≈ 3·σ on
        Gaussian noise).

    Returns
    -------
    clean : np.ndarray
        Despiked OD array.
    mask : np.ndarray of bool
        ``True`` where a value was replaced.
    """
    od = np.asarray(od, dtype=float)
    n = od.size
    clean = od.copy()
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return clean, mask

    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        local = od[lo:hi]
        med = np.median(local)
        scale = _MAD_SCALE * np.median(np.abs(local - med))
        if scale > 1e-12 and abs(od[i] - med) > n_sigmas * scale:
            clean[i] = med
            mask[i] = True
    return clean, mask


# -----------------------------------------------------------------------------
# Smoothing
# -----------------------------------------------------------------------------

def smooth_moving_average(od: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered moving average with edge-replicated padding."""
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(od, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(od)]


def smooth_savgol(
    od: np.ndarray,
    window_length: int = 7,
    polyorder: int = 3,
) -> np.ndarray:
    """Savitzky-Golay smoothing."""
    wl = min(window_length, len(od))
    if wl % 2 == 0:
        wl -= 1
    if wl < polyorder + 2:
        return np.asarray(od, dtype=float).copy()
    return savgol_filter(od, window_length=wl, polyorder=polyorder)


SMOOTHERS: dict[str, Callable] = {
    "moving_average": smooth_moving_average,
    "savgol": smooth_savgol,
}


# -----------------------------------------------------------------------------
# Parameter extraction
# -----------------------------------------------------------------------------

_LOG_EPS = 1e-6
_SLIDE_WINDOW = 4
_SUSTAIN_WINDOW = 3


def extract_params(
    time: np.ndarray,
    od: np.ndarray,
    sustain_window: int = _SUSTAIN_WINDOW,
) -> dict:
    """Extract growth parameters from a single smoothed OD600 curve.

    Parameters
    ----------
    time:
        Time values (hours).
    od:
        Blank-subtracted, filtered and smoothed OD600 values aligned with *time*.
    sustain_window:
        Number of consecutive sliding windows whose slopes are averaged
        before selecting the maximum. ``1`` recovers single-window behaviour.

    Returns
    -------
    dict with keys: max_od, auc, growth_rate, lag_time_h, time_to_max_od_h.
    """
    max_od = float(np.nanmax(od))
    auc = _trapz_compat(od, time)
    time_to_max_od_h = float(time[np.argmax(od)])

    log_od = np.log(np.maximum(od, 0.0) + _LOG_EPS)

    n_windows = len(log_od) - _SLIDE_WINDOW + 1
    if n_windows <= 0:
        return {
            "max_od": max_od,
            "auc": auc,
            "growth_rate": 0.0,
            "lag_time_h": float(time[-1]),
            "time_to_max_od_h": time_to_max_od_h,
        }

    slopes = np.empty(n_windows)
    intercepts = np.empty(n_windows)
    for i in range(n_windows):
        t_win = time[i : i + _SLIDE_WINDOW]
        y_win = log_od[i : i + _SLIDE_WINDOW]
        slopes[i], intercepts[i] = np.polyfit(t_win, y_win, 1)

    sw = min(sustain_window, n_windows)

    if sw <= 1:
        best_idx = int(np.argmax(slopes))
    else:
        cumsum = np.cumsum(slopes)
        rolling_mean = (cumsum[sw - 1 :] - np.concatenate([[0.0], cumsum[: -sw]])) / sw
        best_start = int(np.argmax(rolling_mean))
        region_slopes = slopes[best_start : best_start + sw]
        best_idx = best_start + int(np.argmax(region_slopes))

    best_slope = float(slopes[best_idx])
    best_intercept = float(intercepts[best_idx])

    growth_rate = float(max(best_slope, 0.0))

    baseline_log = float(log_od[0])
    if growth_rate > 0:
        lag_time_h = float((baseline_log - best_intercept) / best_slope)
        lag_time_h = max(lag_time_h, 0.0)
    else:
        lag_time_h = float(time[-1])

    return {
        "max_od": max_od,
        "auc": auc,
        "growth_rate": growth_rate,
        "lag_time_h": lag_time_h,
        "time_to_max_od_h": time_to_max_od_h,
    }


# -----------------------------------------------------------------------------
# Growth-class threshold
# -----------------------------------------------------------------------------

def compute_growth_threshold(df: pd.DataFrame) -> pd.Series:
    """Per-experiment growth/no-growth threshold = 2 × std(blank OD600).

    Threshold is applied on blank-subtracted OD (the mean component
    cancels), so the spread reflects raw instrument noise.

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


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

_ID_COLS = [
    "exp_id",
    "date",
    "well_id",
    "row",
    "col",
    "well_index",
    "strain_name",
    "media_name",
    "atmosphere",
    "temperature",
]
_REPLICATE_GROUP = ["strain_name", "media_name", "atmosphere", "temperature"]
_PARAM_COLS = [
    "max_od",
    "auc",
    "growth_rate",
    "lag_time_h",
    "time_to_max_od_h",
    "growth_class",
]


def extract_growth_curves(
    df: pd.DataFrame,
    smoother: str = "savgol",
    smoother_kwargs: dict | None = None,
    hampel_kwargs: dict | None = None,
    with_replicates: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract growth parameters from a long-format QC'd DataFrame.

    All parameters are computed from the same ``od600_smooth`` array.

    Parameters
    ----------
    df:
        Long-format DataFrame produced by ``run_qc``.
    smoother:
        Key in :data:`SMOOTHERS` — ``"savgol"`` (default) or ``"moving_average"``.
    smoother_kwargs:
        Extra keyword arguments forwarded to the smoother.
    hampel_kwargs:
        Extra keyword arguments forwarded to :func:`hampel_filter`
        (e.g. ``{"window": 9, "n_sigmas": 4}``).
    with_replicates:
        If ``True``, average parameters across replicates sharing
        ``(strain_name, media_name, atmosphere, temperature)``.

    Returns
    -------
    wide_df : pd.DataFrame
        One row per well (or per replicate group) with growth parameters.
    long_df : pd.DataFrame
        Copy of *df* with two new columns: ``od600_clean`` and ``od600_smooth``.
    """
    if smoother not in SMOOTHERS:
        raise ValueError(
            f"Unknown smoother {smoother!r}. Choose from {list(SMOOTHERS)}"
        )
    smooth_fn = SMOOTHERS[smoother]
    smooth_kw = smoother_kwargs or {}
    hampel_kw = hampel_kwargs or {}

    samples = df.loc[~df["is_blank"]].copy()
    thresholds = compute_growth_threshold(df)

    long_out = df.copy()
    long_out["od600_clean"] = long_out["od600"].astype(float)
    long_out["od600_smooth"] = np.nan

    records: list[dict] = []
    spike_stats: dict[str, dict[str, int]] = {}

    for (exp_id, well_id), grp in samples.groupby(["exp_id", "well_id"], sort=False):
        grp = grp.sort_values("timepoint")
        time_raw = grp["time_value"].to_numpy(dtype=float)
        time_unit = grp["time_unit"].iloc[0] if "time_unit" in grp.columns else "h"
        time = time_raw / 60.0 if time_unit == "min" else time_raw
        od = grp["od600"].to_numpy(dtype=float)

        od_clean, spike_mask = hampel_filter(od, **hampel_kw)
        od_smooth = smooth_fn(od_clean, **smooth_kw)

        long_out.loc[grp.index, "od600_clean"] = od_clean
        long_out.loc[grp.index, "od600_smooth"] = od_smooth

        n_spikes = int(spike_mask.sum())
        if n_spikes:
            s = spike_stats.setdefault(exp_id, {"n_points": 0, "n_wells": 0})
            s["n_points"] += n_spikes
            s["n_wells"] += 1

        params = extract_params(time, od_smooth)
        thresh = thresholds.get(exp_id, 0.0)
        params["growth_class"] = int(params["max_od"] > thresh)

        meta = grp.iloc[0]
        row = {c: meta[c] for c in _ID_COLS if c in meta.index}
        if "replicate" in meta.index:
            row["replicate"] = meta["replicate"]
        row.update(params)
        records.append(row)

    if spike_stats:
        total_points = sum(s["n_points"] for s in spike_stats.values())
        total_wells = sum(s["n_wells"] for s in spike_stats.values())
        logger.info(
            "Hampel filter replaced {} timepoint(s) across {} well(s).",
            total_points,
            total_wells,
        )
        for exp_id, s in spike_stats.items():
            logger.debug(
                "  {:<20s} {:>4d} timepoint(s) in {:>3d} well(s)",
                exp_id,
                s["n_points"],
                s["n_wells"],
            )
    else:
        logger.info("Hampel filter: no spikes detected.")

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
