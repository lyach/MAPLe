"""Quality control: blank subtraction, growth flagging, and plate coeff of variation (CV)."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from loguru import logger


_BLANK_GROUP_COND = ["atmosphere", "temperature", "shaking", "media_name"]

BlankMethod = Literal["exp_median", "cond_avg"]


def subtract_blanks(
    df: pd.DataFrame,
    method: BlankMethod = "exp_median",
) -> pd.DataFrame:
    """Subtract a static blank value from each well.

    Two-stage approach:
      1. Per experiment: median OD600 of blank wells across all timepoints
         (collapses time to a single scalar per experiment+media).
      2. Per condition group: average of per-experiment medians sharing the
         same (atmosphere, temperature, shaking, media_name).

    Parameters
    ----------
    method
        Which blank estimate to subtract:
        - ``"exp_median"``: per-experiment median (stage 1 only).
        - ``"cond_avg"``: average of experiment medians across shared
          conditions (stage 1 + stage 2).

    Adds columns ``od600_raw``, ``blank_exp_median``, ``blank_cond_avg``,
    ``blank_cond_cv`` and overwrites ``od600`` with the blank-subtracted values.
    ``blank_cond_cv`` is the within condition-group coefficient of variation.
    """
    df = df.copy()
    df["od600_raw"] = df["od600"]

    # Stage 1: per-experiment median across all timepoints
    exp_median = (
        df.loc[df["is_blank"]]
        .groupby(["exp_id", "media_name"])["od600"]
        .median()
        .rename("blank_exp_median")
        .reset_index()
    )

    # Stage 2: average + CV of experiment medians per condition group
    exp_meta = (
        df[["exp_id"] + _BLANK_GROUP_COND]
        .drop_duplicates()
    )
    exp_median_with_cond = exp_median.merge(exp_meta, on=["exp_id", "media_name"], how="left")
    cond_stats = (
        exp_median_with_cond
        .groupby(_BLANK_GROUP_COND)["blank_exp_median"]
        .agg(blank_cond_avg="mean", _std="std")
        .assign(blank_cond_cv=lambda x: x["_std"] / x["blank_cond_avg"])
        .drop(columns="_std")
        .reset_index()
    )

    # Merge all statistics onto main df
    df = df.merge(exp_median, on=["exp_id", "media_name"], how="left")
    df = df.merge(cond_stats, on=_BLANK_GROUP_COND, how="left")
    df["blank_exp_median"] = df["blank_exp_median"].fillna(0.0)
    df["blank_cond_avg"] = df["blank_cond_avg"].fillna(0.0)
    df["blank_cond_cv"] = df["blank_cond_cv"].fillna(0.0)

    subtract_col = "blank_exp_median" if method == "exp_median" else "blank_cond_avg"
    df["od600"] = df["od600"] - df[subtract_col]
    return df


def report_blank_cv(
    df: pd.DataFrame,
    threshold: float = 0.1,
) -> pd.DataFrame:
    """Print condition groups where blank CV exceeds ``threshold``.

    Expects ``df`` to have been processed by :func:`subtract_blanks` so that
    ``blank_exp_median``, ``blank_cond_avg``, and ``blank_cond_cv`` are present.

    Parameters
    ----------
    threshold
        CV threshold above which a condition group is flagged (default 0.1).

    Returns
    -------
    DataFrame of flagged condition groups with columns
    ``_BLANK_GROUP_COND + [blank_cond_cv, blank_cond_avg, n_experiments]``.
    """
    required = {"blank_exp_median", "blank_cond_avg", "blank_cond_cv"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing}. Run subtract_blanks() first.")

    exp_level = (
        df[["exp_id"] + _BLANK_GROUP_COND + ["blank_exp_median", "blank_cond_avg", "blank_cond_cv"]]
        .drop_duplicates(subset=["exp_id"] + _BLANK_GROUP_COND)
    )
    flagged = (
        exp_level
        .groupby(_BLANK_GROUP_COND + ["blank_cond_avg", "blank_cond_cv"], sort=False)
        .agg(n_experiments=("exp_id", "nunique"))
        .reset_index()
        .query("blank_cond_cv > @threshold")
        .sort_values("blank_cond_cv", ascending=False)
        .reset_index(drop=True)
    )

    if flagged.empty:
        logger.info("No condition groups with blank CV > {}.", threshold)
    else:
        logger.warning(
            "{} condition group(s) with blank CV > {}:\n{}",
            len(flagged),
            threshold,
            flagged[_BLANK_GROUP_COND + ["blank_cond_cv", "blank_cond_avg", "n_experiments"]]
            .to_string(index=False, float_format="{:.4f}".format),
        )

    return flagged


def flag_no_growth(
    df: pd.DataFrame, threshold_factor: float = 2.0
) -> pd.DataFrame:
    """ DISCARDED, DONE IN CURVES.PY
    Flag wells (blank-subtracted) with low OD - no growth. 

    Blank noise threshold per media group = ``threshold_factor * std(blank_raw)``.
    A well is flagged ``no_growth=True`` when its ``max(od600)``  stays below 
    a blank noise threshold.
    """
    df = df.copy()
    od_col = "od600_raw" if "od600_raw" in df.columns else "od600"

    blank_noise = (
        df.loc[df["is_blank"]]
        .groupby(_BLANK_GROUP_COND)[od_col]
        .std()
        .rename("blank_std")
        .reset_index()
    )
    blank_noise["blank_std"] = blank_noise["blank_std"].fillna(0.0)
    blank_noise["growth_threshold"] = threshold_factor * blank_noise["blank_std"]

    merge_keys = _BLANK_GROUP_COND
    well_max = (
        df.groupby(merge_keys + ["well_id"])["od600"]
        .max()
        .rename("well_max_od")
        .reset_index()
    )
    well_max = well_max.merge(blank_noise[merge_keys + ["growth_threshold"]], on=merge_keys, how="left")
    well_max["growth_threshold"] = well_max["growth_threshold"].fillna(0.0)
    well_max["no_growth"] = well_max["well_max_od"] < well_max["growth_threshold"]

    flag_map = well_max.set_index(merge_keys + ["well_id"])["no_growth"]
    df["no_growth"] = (
        df.set_index(merge_keys + ["well_id"]).index.map(flag_map).values
    )
    return df


def clip_negatives(df: pd.DataFrame) -> pd.DataFrame:
    """Clip ``od600`` values to a minimum of 0."""
    df = df.copy()
    df["od600"] = df["od600"].clip(lower=0)
    return df


def compute_well_cv(
    df: pd.DataFrame, cv_threshold: float = 0.15
) -> pd.DataFrame:
    """Compute CV of each well across replicates per (exp_id, well_id, timepoint) and flag wells.

    CV is defined as std / mean of ``od600`` across replicates sharing the
    same (exp_id, well_id, timepoint).  Adds ``well_cv`` and ``well_cv_flag``
    columns.
    """
    df = df.copy()
    od_col = "od600_raw" if "od600_raw" in df.columns else "od600"

    well_stats = (
        df.groupby(["exp_id", "well_id", "timepoint"])[od_col]
        .agg(["mean", "std"])
    )
    well_stats["well_cv"] = well_stats["std"] / well_stats["mean"]
    well_stats = well_stats[["well_cv"]].reset_index()

    df = df.merge(well_stats, on=["exp_id", "well_id", "timepoint"], how="left")
    df["well_cv"] = df["well_cv"].fillna(0.0)
    df["well_cv_flag"] = df["well_cv"] > cv_threshold
    return df


def run_qc(
    df: pd.DataFrame,
    blank_method: BlankMethod = "cond_avg",
    threshold_factor: float = 2.0,
    cv_threshold: float = 0.15,
) -> pd.DataFrame:
    """Run the full QC pipeline in order."""
    df = subtract_blanks(df, method=blank_method)
    df = clip_negatives(df)
    # df = compute_well_cv(df, cv_threshold=cv_threshold) # no replicates yet
    return df
