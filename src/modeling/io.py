"""Tabular I/O helpers for CSV and Parquet files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_SUPPORTED_SUFFIXES = {".csv", ".parquet"}


def read_table(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Parquet file into a DataFrame."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(
        f"unsupported table format {suffix!r} for {path}; "
        f"expected one of {sorted(_SUPPORTED_SUFFIXES)}"
    )


def write_table(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    """Write a DataFrame to CSV or Parquet based on the file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False, **kwargs)
        return
    if suffix == ".parquet":
        df.to_parquet(path, index=False, **kwargs)
        return
    raise ValueError(
        f"unsupported table format {suffix!r} for {path}; "
        f"expected one of {sorted(_SUPPORTED_SUFFIXES)}"
    )
