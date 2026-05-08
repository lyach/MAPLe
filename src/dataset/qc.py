"""Quality control: blank subtraction, growth flagging, and plate coeff of variation (CV)."""

from __future__ import annotations

import pandas as pd


def subtract_blanks(df: pd.DataFrame) -> pd.DataFrame:
    """Subtract per-(exp_id, timepoint) blank mean from all wells.

    Adds ``od600_raw`` (original) and overwrites ``od600`` with the
    blank-subtracted values.
    """
    df = df.copy()
    df["od600_raw"] = df["od600"]

    blank_mean = (
        df.loc[df["is_blank"]]
        .groupby(["exp_id", "timepoint"])["od600"]
        .mean()
        .rename("blank_mean")
    )

    df = df.merge(blank_mean, on=["exp_id", "timepoint"], how="left")
    df["blank_mean"] = df["blank_mean"].fillna(0.0)
    df["od600"] = df["od600"] - df["blank_mean"]
    return df.drop(columns=["blank_mean"])


def flag_no_growth(
    df: pd.DataFrame, threshold_factor: float = 2.0
) -> pd.DataFrame:
    """Flag wells where ``max(od600) < threshold_factor * mean(blank_od_raw)``.
    """
    df = df.copy()
    od_col = "od600_raw" if "od600_raw" in df.columns else "od600"

    blank_global_mean = (
        df.loc[df["is_blank"]]
        .groupby("exp_id")[od_col]
        .mean()
        .rename("blank_global_mean")
    )

    well_max = (
        df.groupby(["exp_id", "well_id"])["od600"]
        .max()
        .rename("well_max_od")
    )

    merged = (
        well_max.reset_index()
        .merge(blank_global_mean, on="exp_id", how="left")
    )
    merged["blank_global_mean"] = merged["blank_global_mean"].fillna(0.0)
    merged["no_growth"] = (
        merged["well_max_od"] < threshold_factor * merged["blank_global_mean"]
    )

    flag_map = merged.set_index(["exp_id", "well_id"])["no_growth"]
    df["no_growth"] = (
        df.set_index(["exp_id", "well_id"]).index.map(flag_map).values
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
