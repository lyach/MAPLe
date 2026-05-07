"""Base class for equipment parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from src.dataset.schemas import PlateLayout


class BaseParser(ABC):
    """
    Returns a long-format DataFrame with columns:
    ``[timepoint, row, col, od600]`` where *row* is a lowercase letter
    and *col* is a 1-based integer.
    """

    @abstractmethod
    def parse(self, path: Path, layout: PlateLayout) -> pd.DataFrame: ...
