"""Parser for BioSpa plate reader.

Raw file format: CSV with time as rows and wells as columns.
Header line: ``Time,Temperature,A1,A2,...,H12``
Data lines: ``H:MM:SS,temp,od1,od2,...``
"""

from __future__ import annotations

import logging
import re
import string
from pathlib import Path

import pandas as pd

from src.dataset.schemas import PlateLayout
from src.dataset.parsers import register
from src.dataset.parsers.base import BaseParser

log = logging.getLogger(__name__)

_WELL_RE = re.compile(r"^([A-Ha-h])(\d{1,2})$")


@register("biospa")
class BiospaParser(BaseParser):
    def parse(self, path: Path, layout: PlateLayout) -> pd.DataFrame:
        df = pd.read_csv(path)

        well_cols = [c for c in df.columns if _WELL_RE.match(c)]
        if not well_cols:
            raise ValueError(f"File {path.name}: no well columns found in header.")

        n_rows = len(layout.name_rows)
        n_cols = len(layout.name_cols)
        valid_rows = set(string.ascii_uppercase[:n_rows])
        used_wells = [
            c
            for c in well_cols
            if c[0].upper() in valid_rows and 1 <= int(c[1:]) <= n_cols
        ]

        if not used_wells:
            raise ValueError(
                f"File {path.name}: no well columns match plate dimensions "
                f"({n_rows} rows × {n_cols} cols)."
            )

        log.info(
            "%s: %d timepoints × %d wells", path.name, len(df), len(used_wells)
        )

        df = df.reset_index(drop=True)
        df["timepoint"] = df.index

        long = df.melt(
            id_vars=["timepoint"],
            value_vars=used_wells,
            var_name="well",
            value_name="od600",
        )

        long["row"] = long["well"].str[0].str.lower()
        long["col"] = long["well"].str[1:].astype(int)

        return long[["timepoint", "row", "col", "od600"]]