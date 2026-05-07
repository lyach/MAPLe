"""Parser for Sunrise plate readers (sunrise 1 / sunrise 2).

Raw file format: consecutive blocks, one per timepoint.
Each block = 1 header line (``<>,1,2,...,N``) + *n_rows* data lines (``A,...``).
"""

from __future__ import annotations

import logging
import string
from pathlib import Path

import pandas as pd

from src.dataset.schemas import PlateLayout
from src.dataset.parsers import register
from src.dataset.parsers.base import BaseParser

log = logging.getLogger(__name__)


@register("sunrise 1", "sunrise 2")
class SunriseParser(BaseParser):
    def parse(self, path: Path, layout: PlateLayout) -> pd.DataFrame:
        n_rows = len(layout.name_rows)
        n_cols = len(layout.name_cols)
        block_size = n_rows + 1  # header + data rows

        with open(path) as fh:
            lines = [line.strip() for line in fh if line.strip()]

        if len(lines) % block_size != 0:
            raise ValueError(
                f"File {path.name}: line count ({len(lines)}) is not divisible "
                f"by block size ({block_size}). Check time_points or plate dimensions."
            )

        n_blocks = len(lines) // block_size
        if n_blocks != layout.time_points:
            log.warning(
                "File %s: found %d blocks but layout specifies %d timepoints. "
                "Parsing %d available blocks.",
                path.name,
                n_blocks,
                layout.time_points,
                n_blocks,
            )

        row_letters = list(string.ascii_lowercase[:n_rows])
        records: list[dict] = []

        for tp in range(n_blocks):
            offset = tp * block_size
            # skip header line (offset), parse data lines
            for r_idx in range(n_rows):
                parts = lines[offset + 1 + r_idx].split(",")
                row_letter = parts[0].strip().lower()
                if row_letter != row_letters[r_idx]:
                    raise ValueError(
                        f"Expected row '{row_letters[r_idx]}' but got "
                        f"'{row_letter}' at timepoint {tp}, data line {r_idx}."
                    )
                values = [float(v) for v in parts[1 : n_cols + 1]]
                for c_idx, od in enumerate(values):
                    records.append(
                        {
                            "timepoint": tp,
                            "row": row_letter,
                            "col": c_idx + 1,
                            "od600": od,
                        }
                    )

        return pd.DataFrame(records)
