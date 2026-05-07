"""Orchestrator: combine metadata + layout + raw readings into a DataFrame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dataset.schemas import ExperimentMetadata, PlateLayout
from src.dataset.parsers import get_parser
from src.dataset.plate import (
    build_time_axis,
    build_well_map,
    parse_layout_csv,
    well_map_to_df,
)


def load_metadata(path: str | Path) -> list[ExperimentMetadata]:
    """Load the metadata CSV and return one model per row."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    return [ExperimentMetadata(**row) for row in df.to_dict(orient="records")]


def _resolve_raw_path(raw_dir: Path, date: str, exp_id: str) -> Path:
    """Find the raw CSV for a given experiment.

    Naming convention: ``<date>_<exp_id>_raw*.csv``
    """
    candidates = sorted(raw_dir.glob(f"{date}_{exp_id}_raw*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No raw file matching '{date}_{exp_id}_raw*.csv' in {raw_dir}"
        )
    return candidates[0]


def load_experiment(
    metadata_path: str | Path,
    layout_path: str | Path,
    raw_dir: str | Path,
) -> pd.DataFrame:
    """Main entry point: load all experiments described by the metadata/layout pair.

    Returns a concatenated long-format DataFrame ready for QC.
    """
    metadata_path = Path(metadata_path)
    layout_path = Path(layout_path)
    raw_dir = Path(raw_dir)

    metadata_list = load_metadata(metadata_path)
    layout_list = parse_layout_csv(layout_path)

    meta_by_id = {m.exp_id: m for m in metadata_list}
    layout_by_id = {l.exp_id: l for l in layout_list}

    frames: list[pd.DataFrame] = []

    for exp_id, layout in layout_by_id.items():
        meta = meta_by_id.get(exp_id)
        if meta is None:
            raise KeyError(
                f"exp_id '{exp_id}' found in layout but missing from metadata."
            )

        parser = get_parser(layout.equipment)
        raw_path = _resolve_raw_path(raw_dir, layout.date, exp_id)
        raw_df = parser.parse(raw_path, layout)

        wells_df = well_map_to_df(build_well_map(layout))
        time_axis = build_time_axis(layout)

        merged = raw_df.merge(wells_df, on=["row", "col"], how="left")
        merged["time_value"] = merged["timepoint"].map(
            dict(enumerate(time_axis))
        )
        merged["time_unit"] = layout.time_unit

        meta_dict = meta.model_dump()
        for key, val in meta_dict.items():
            if key not in merged.columns:
                merged[key] = val

        frames.append(merged)

    if not frames:
        raise ValueError("No experiments were loaded — check input files.")

    combined = pd.concat(frames, ignore_index=True)

    column_order = [
        "exp_id",
        "date",
        "replicate",
        "atmosphere",
        "temperature",
        "shaking",
        "equipment",
        "well_id",
        "row",
        "col",
        "well_index",
        "sample_name",
        "media_name",
        "is_blank",
        "timepoint",
        "time_value",
        "time_unit",
        "od600",
    ]
    extra_cols = [c for c in combined.columns if c not in column_order]
    return combined[column_order + extra_cols]
