"""Create a modeling-ready dataset from curves tables + strain metadata.

Usage:
    python scripts/modeling/create_modeling_dataset.py config/dataset/modeling_dataset_aero.yaml
"""

import argparse
import sys
from datetime import date
from pathlib import Path

from loguru import logger


def _find_root(start: Path) -> Path:
    """Ascend from ``start`` to the directory containing ``pyproject.toml``."""
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return start.parents[2]


ROOT = _find_root(Path(__file__).resolve())
sys.path.insert(0, str(ROOT))

from src.modeling.config import load_dataset_config
from src.modeling.dataset import build_modeling_dataset
from src.modeling.io import write_table


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
        colorize=sys.stderr.isatty(),
    )
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}",
            colorize=False,
            mode="w",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to dataset YAML config.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    cfg = load_dataset_config(args.config)

    output_dir = ROOT / "data" / "interim"
    output_path = output_dir / cfg.output_name
    log_path = output_path.with_suffix(".log")

    setup_logging(args.log_level, log_file=log_path)

    logger.info("date: {}", date.today().isoformat())
    logger.info("config: {}", args.config)
    logger.info("grain: {}", cfg.grain)

    df = build_modeling_dataset(cfg, root=ROOT)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_table(df, output_path)
    logger.success(
        "saved modeling dataset -> {} ({} rows x {} cols, {:.1f} KB)",
        output_path,
        *df.shape,
        output_path.stat().st_size / 1024,
    )
    logger.success("log -> {}", log_path)


if __name__ == "__main__":
    main()
