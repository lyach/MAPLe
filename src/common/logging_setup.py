"""
logging_setup.py
================
Shared dual-output logging config for pipeline scripts that need a forensic
run record (as opposed to run_dataset_pipeline.py's loguru-based banners,
used for quick single-shot transforms). Ported from
strain_metadata/logging_setup.py so a third near-identical copy doesn't
appear the next time a multi-stage, external-tool-adjacent pipeline is added.

  * CONSOLE  -- clean, human-friendly progress (INFO and above).
  * LOG FILE -- everything at DEBUG level, one timestamped file per run,
                kept permanently so a run can be diffed or investigated later.

The log file lives at logs/run_<YYYY-MM-DD_HH-MM-SS>.log relative to the
project root. The function returns the path so the caller can mention it
to the user at the end of the run.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

_FILE_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s:%(lineno)d  %(message)s"
_CONSOLE_FORMAT = "%(asctime)s  %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    project_root: Path,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    quiet_console: bool = False,
) -> Path:
    """Configure root logging with a clean console stream and a verbose
    timestamped file. Returns the path to the log file created.

    Parameters
    ----------
    project_root : Path
        Used to locate the logs/ directory.
    console_level : int
        Minimum level shown on the console (default INFO).
    file_level : int
        Minimum level written to the log file (default DEBUG -- everything).
    quiet_console : bool
        If True, the console only shows WARNING and above (useful for
        cron jobs); the file still captures everything.
    """
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = logs_dir / f"run_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_FILE_DATE_FORMAT))
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    effective_console_level = logging.WARNING if quiet_console else console_level
    console_handler.setLevel(effective_console_level)
    console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console_handler)

    logging.getLogger("logging_setup").debug(
        "=== Run started %s | log file: %s ===",
        datetime.now().isoformat(timespec="seconds"), log_path,
    )

    return log_path
