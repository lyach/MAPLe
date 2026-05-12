"""Quality control: blank subtraction, growth flagging, and plate coeff of variation (CV)."""

from __future__ import annotations

import pandas as pd


_BLANK_GROUP = ["exp_id", "timepoint", "media_name"]
_BLANK_GROUP_COND = ["atmosphere", "temperature", "shaking", "timepoint", "media_name"] # currently used as there are no replicates

def subtract_blanks(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract the media-matched blank mean from each well.

    For every group, the mean OD600 of blank wells sharing that media 
    is subtracted from all wells in that group.

    Adds ``od600_raw`` (original values) and overwrites ``od600`` with the
    blank-subtracted values.
    """
    df = df.copy()
    df["od600_raw"] = df["od600"]

    blank_stats = (
        df.loc[df["is_blank"]]
        .groupby(_BLANK_GROUP_COND)["od600"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "blank_mean", "std": "blank_std"})
    )
    blank_stats["blank_cv"] = blank_stats["blank_std"] / blank_stats["blank_mean"]

    df = df.merge(blank_stats[["blank_mean", "blank_cv"]], on=_BLANK_GROUP_COND, how="left")
    df["blank_mean"] = df["blank_mean"].fillna(0.0)
    df["blank_cv"] = df["blank_cv"].fillna(0.0)
    df["od600"] = df["od600"] - df["blank_mean"]
    return df.drop(columns=["blank_mean"])


def flag_no_growth(
    df: pd.DataFrame, threshold_factor: float = 2.0
) -> pd.DataFrame:
    """Flag wells (blank-subtracted) with low OD - no growth.

    Blank noise threshold per media group = ``threshold_factor * std(blank_raw)``.
    A well is flagged ``no_growth=True`` when its ``max(od600)``  stays below 
    a blank noise threshold.
    """
    df = df.copy()
    od_col = "od600_raw" if "od600_raw" in df.columns else "od600"

    blank_noise = (
        df.loc[df["is_blank"]]
        .groupby(_BLANK_GROUP_COND[:3] + ["media_name"])[od_col]
        .std()
        .rename("blank_std")
        .reset_index()
    )
    blank_noise["blank_std"] = blank_noise["blank_std"].fillna(0.0)
    blank_noise["growth_threshold"] = threshold_factor * blank_noise["blank_std"]

    merge_keys = _BLANK_GROUP_COND[:3] + ["media_name"]
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
    threshold_factor: float = 2.0,
    cv_threshold: float = 0.15,
) -> pd.DataFrame:
    """Run the full QC pipeline in order."""
    df = subtract_blanks(df)
    df = flag_no_growth(df, threshold_factor=threshold_factor)
    df = clip_negatives(df)
    # df = compute_well_cv(df, cv_threshold=cv_threshold) # no replicates yet
    return df
