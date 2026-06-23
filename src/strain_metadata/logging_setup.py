"""
logging_setup.py
================
Configures the dual-output logging system for the pipeline:

  * CONSOLE  -- clean, human-friendly progress. Shows INFO-level milestones
                (which strain is processing, success/failure) plus any
                warnings and errors. No API-call noise.
  * LOG FILE -- a complete forensic record. Captures EVERYTHING at DEBUG
                level: every NCBI API call, every resolution decision, every
                timing, every rejected candidate. One timestamped file per
                run, kept permanently so you can diff runs or investigate a
                problem days later.

The log file lives at logs/run_<YYYY-MM-DD_HH-MM-SS>.log relative to the
project root. The function returns the path so the pipeline can mention it
to the user at the end of the run.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


# A more detailed format for the file (includes module name + line number so
# you can jump straight to the code that emitted a message); a cleaner format
# for the console.
_FILE_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s:%(lineno)d  %(message)s"
_CONSOLE_FORMAT = "%(asctime)s  %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _ConsoleFilter(logging.Filter):
    """Lets through INFO-and-above by default, but suppresses DEBUG records
    on the console. (The file handler has no such filter, so DEBUG still
    goes to disk.) Also suppresses the very chatty per-API-call INFO lines
    by routing them to DEBUG-only loggers -- see usage in ncbi_client."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Console shows INFO and above only.
        return record.levelno >= logging.INFO


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

    # The root logger must be set to the most permissive level of its
    # handlers (DEBUG), otherwise it would filter records before they ever
    # reach the file handler.
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove any pre-existing handlers (e.g. from a previous run in the same
    # Python session, or a stray basicConfig call) so we don't double-log.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    # --- file handler: captures everything ---
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_FILE_DATE_FORMAT))
    root.addHandler(file_handler)

    # --- console handler: clean progress only ---
    console_handler = logging.StreamHandler(sys.stdout)
    effective_console_level = logging.WARNING if quiet_console else console_level
    console_handler.setLevel(effective_console_level)
    console_handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console_handler)

    # Record the run header in the file so each log is self-describing.
    logging.getLogger("logging_setup").debug(
        "=== Run started %s | log file: %s ===",
        datetime.now().isoformat(timespec="seconds"), log_path,
    )

    return log_path
