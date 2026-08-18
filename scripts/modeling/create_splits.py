"""Precompute and save reusable CV splits from a processed dataset.

Splits are stored as a compact group->fold map (one row per ``split_on_column``
value) under ``data/modeling/splits_<dataset_stem>/``, alongside a log and a
machine-readable stats JSON describing the train/test distribution per fold.

Usage:
    python scripts/modeling/create_splits.py config/splits/bacdive_genus.yaml
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path


def _find_root(start: Path) -> Path:
    """Ascend from ``start`` to the directory containing ``pyproject.toml``."""
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return start.parents[2]


ROOT = _find_root(Path(__file__).resolve())
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402

from src.modeling.config import load_split_config  # noqa: E402
from src.modeling.io import read_table  # noqa: E402
from src.modeling.splits import (  # noqa: E402
    build_group_fold_map,
    fold_distribution_stats,
    save_split_map,
)


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


def _log_stats(stats: list[dict]) -> None:
    for s in stats:
        logger.info(
            "[{} | repeat {} | fold {}] train={} rows ({} groups) | "
            "test={} rows ({} groups)",
            s["strategy"], s["repeat"], s["fold"],
            s["n_train"], s["n_groups_train"], s["n_test"], s["n_groups_test"],
        )
        logger.info("    target train: {}", s["target_train"])
        logger.info("    target test:  {}", s["target_test"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to split YAML config.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    cfg = load_split_config(args.config)

    dataset_path = Path(cfg.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    stem = dataset_path.stem

    output_dir = ROOT / "data" / "modeling" / f"splits_{stem}"
    base = f"{cfg.split_on_column}_{stem}"
    ext = "csv" if cfg.output_format == "csv" else "parquet"
    map_path = output_dir / f"{base}.{ext}"
    log_path = output_dir / f"{base}.log"
    stats_path = output_dir / f"{base}_stats.json"

    setup_logging(args.log_level, log_file=log_path)

    logger.info("date: {}", date.today().isoformat())
    logger.info("config: {}", args.config)
    logger.info("dataset: {}", dataset_path)
    logger.info("split_on_column: {} | target: {}", cfg.split_on_column, cfg.target)
    logger.info(
        "strategies: {} | n_splits: {} | n_repeats: {} | seed: {}",
        cfg.cv.strategy, cfg.n_splits, cfg.cv.n_repeats, cfg.seed,
    )

    df = read_table(dataset_path)
    logger.info("loaded dataset: {} rows x {} cols", *df.shape)

    for col in (cfg.split_on_column, cfg.target):
        if col not in df.columns:
            raise ValueError(f"column '{col}' not in dataset {dataset_path}")

    n_groups = int(df[cfg.split_on_column].nunique())
    logger.info("unique '{}' groups: {}", cfg.split_on_column, n_groups)

    split_map = build_group_fold_map(
        df,
        split_on=cfg.split_on_column,
        strategies=cfg.cv.strategy,
        n_splits=cfg.n_splits,
        n_repeats=cfg.cv.n_repeats,
        seed=cfg.seed,
        y=df[cfg.target].values,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_split_map(split_map, map_path)
    logger.success("saved split map -> {} ({} rows)", map_path, len(split_map))

    logger.info("=" * 60)
    logger.info("SPLIT DISTRIBUTION STATS")
    logger.info("=" * 60)
    stats = fold_distribution_stats(df, split_map, cfg.split_on_column, cfg.target)
    _log_stats(stats)

    stats_payload = {
        "dataset_path": str(dataset_path),
        "dataset_shape": list(df.shape),
        "split_on_column": cfg.split_on_column,
        "target": cfg.target,
        "strategies": cfg.cv.strategy,
        "n_splits": cfg.n_splits,
        "n_repeats": cfg.cv.n_repeats,
        "seed": cfg.seed,
        "n_groups_total": n_groups,
        "created": date.today().isoformat(),
        "folds": stats,
    }
    with open(stats_path, "w") as f:
        json.dump(stats_payload, f, indent=2, default=str)
    logger.success("saved stats -> {}", stats_path)
    logger.success("log -> {}", log_path)


if __name__ == "__main__":
    main()
