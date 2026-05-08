"""Plate layout parsing and well map construction."""

from __future__ import annotations

import string
from pathlib import Path

import pandas as pd

from src.dataset.schemas import PlateLayout, WellInfo


def parse_layout_csv(path: str | Path) -> list[PlateLayout]:
    """Load a layout CSV and return one ``PlateLayout`` per row."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    return [PlateLayout(**row) for row in df.to_dict(orient="records")]


def build_well_map(layout: PlateLayout) -> list[WellInfo]:
    """Build a list of ``WellInfo`` from a plate layout.

    ``name_rows`` maps positionally to plate rows (A, B, ...) and
    ``name_cols`` maps positionally to plate columns (1, 2, ...).
    """
    n_rows = len(layout.name_rows)
    n_cols = len(layout.name_cols)
    letters = list(string.ascii_lowercase[:n_rows])

    blank_rows = set(layout.blank_rows_index)
    blank_cols = set(layout.blank_cols_index)

    wells: list[WellInfo] = []
    for r_idx, row_letter in enumerate(letters):
        sample_name = layout.name_rows[r_idx]
        for c_idx, col_num in enumerate(range(1, n_cols + 1)):
            media_name = layout.name_cols[c_idx]
            is_blank = row_letter in blank_rows and col_num in blank_cols
            well_index = r_idx * n_cols + c_idx
            well_id = f"{row_letter}_{col_num}_{well_index}"

            wells.append(
                WellInfo(
                    well_id=well_id,
                    row=row_letter,
                    col=col_num,
                    well_index=well_index,
                    sample_name=sample_name,
                    media_name=media_name,
                    is_blank=is_blank,
                )
            )
    return wells


def well_map_to_df(wells: list[WellInfo]) -> pd.DataFrame:
    """Convert a well map to a DataFrame for merging with raw readings."""
    return pd.DataFrame([w.model_dump() for w in wells])

