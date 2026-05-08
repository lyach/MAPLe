"""Pydantic data models for experiment metadata, plate layouts, and well info."""

from __future__ import annotations

import re
import string

from pydantic import BaseModel, ConfigDict, field_validator


class ExperimentMetadata(BaseModel):
    """Experiment metadata parsed from the metadata CSV.

    Includes new metadata fields automatically.
    """

    model_config = ConfigDict(extra="allow", coerce_numbers_to_str=True)

    exp_id: str
    replicate: int
    date: str
    atmosphere: str
    temperature: float
    shaking: float
    equipment: str


class PlateLayout(BaseModel):
    """Plate geometry, well names, blank positions, and timing."""

    model_config = ConfigDict(extra="allow", coerce_numbers_to_str=True)

    exp_id: str
    date: str
    replicate: int
    equipment: str
    time_unit: str
    time_interval: float
    no_wells: int
    used_cols_index: int
    used_rows_index: int
    name_cols: list[str]
    name_rows: list[str]
    blank_cols_index: list[int]
    blank_rows_index: list[str]

    @field_validator("name_cols", "name_rows", mode="before")
    @classmethod
    def split_comma_list(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [item.strip().lower() for item in v.split(",")]
        return [item.lower() for item in v]

    @field_validator("blank_cols_index", mode="before")
    @classmethod
    def parse_int_list(cls, v: str | int | list[int]) -> list[int]:
        if isinstance(v, (int, float)):
            return [int(v)]
        if isinstance(v, str):
            return [int(x.strip()) for x in v.split(",")]
        return [int(x) for x in v]

    @field_validator("blank_rows_index", mode="before")
    @classmethod
    def parse_row_range(cls, v: str | list[str]) -> list[str]:
        """Parse range notation like ``A-H`` or comma-separated ``A,B,C``."""
        if isinstance(v, list):
            return [x.lower() for x in v]
        v = v.strip()
        range_match = re.match(r"^([A-Za-z])-([A-Za-z])$", v)
        if range_match:
            start = range_match.group(1).upper()
            end = range_match.group(2).upper()
            letters = list(string.ascii_uppercase)
            i_start, i_end = letters.index(start), letters.index(end)
            return [letters[i].lower() for i in range(i_start, i_end + 1)]
        return [x.strip().lower() for x in v.split(",")]


class WellInfo(BaseModel):
    """Derived record for a single well on the plate."""

    well_id: str
    row: str
    col: int
    well_index: int
    strain_name: str
    media_name: str
    is_blank: bool
