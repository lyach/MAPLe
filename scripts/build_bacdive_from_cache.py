#!/usr/bin/env python3
"""
build_bacdive_from_cache.py
===========================
Rebuild a BacDive modeling table from ``data/external/bacdive_cache/*.json``
without calling the BacDive API.

Reads target + output feature subsets from ``bacdive/build_from_cache.yaml``
(see also ``src/bacdive/config.py``). Extraction / dedupe match
``get_bacdive_dataset.py``; only the fetch step is skipped.

USAGE:
    python scripts/build_bacdive_from_cache.py
    python scripts/build_bacdive_from_cache.py --config bacdive/build_from_cache.yaml
    python scripts/build_bacdive_from_cache.py --target metabolite_utilization
    python scripts/build_bacdive_from_cache.py --limit 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bacdive import config, pipeline
from src.strain_metadata.logging_setup import setup_logging

log = logging.getLogger("build_bacdive_from_cache")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", type=Path, default=config.BUILD_FROM_CACHE_YAML,
        help=f"YAML config (default: {config.BUILD_FROM_CACHE_YAML})",
    )
    p.add_argument(
        "--target", choices=sorted(config.TARGETS), default=None,
        help="Override target from the YAML config (uses that target's full "
             "default schema when it differs from the YAML target)",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Override output path from the YAML config",
    )
    p.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Override cache dir from the YAML config",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N cached JSON files (quick test)",
    )
    p.add_argument("--debug", action="store_true",
                   help="Show DEBUG detail on the console too")
    p.add_argument("--quiet", action="store_true",
                   help="Console shows warnings/errors only")
    p.add_argument("--no-progress", action="store_true",
                   help="Disable tqdm progress bars")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    console_level = logging.DEBUG if args.debug else logging.INFO
    log_path = setup_logging(
        config.ROOT, console_level=console_level, quiet_console=args.quiet,
    )

    if not args.config.exists():
        log.error("config file not found: %s", args.config)
        return 1

    try:
        cfg = config.load_build_from_cache_config(args.config)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    target = args.target or cfg.target
    cache_dir = args.cache_dir or cfg.cache_dir
    if not cache_dir.is_absolute():
        cache_dir = config.ROOT / cache_dir

    if target == cfg.target:
        fieldnames = cfg.fieldnames
        categorical_features = cfg.categorical_features
        output_file = args.out or cfg.output_file
        output_writer = cfg.output_writer
    else:
        log.warning(
            "CLI --target %r differs from YAML target %r; "
            "using the full default schema for %r",
            target, cfg.target, target,
        )
        base = config.TARGETS[target]
        fieldnames = list(base.fieldnames)
        categorical_features = list(base.categorical_features)
        output_file = args.out or base.output_file
        output_writer = cfg.output_writer

    if not output_file.is_absolute():
        output_file = config.ROOT / output_file

    if not cache_dir.is_dir():
        log.error("cache dir not found: %s", cache_dir)
        return 1

    log.info(
        "building from cache %s (target=%s, %d output columns)",
        cache_dir, target, len(fieldnames),
    )
    log.info("output columns: %s", fieldnames)

    n_rows, n_ok, n_failures = pipeline.run_from_cache(
        target=target,
        fieldnames=fieldnames,
        categorical_features=categorical_features,
        output_file=output_file,
        output_writer=output_writer,
        cache_dir=cache_dir,
        limit=args.limit,
        show_progress=not (args.no_progress or args.quiet),
    )

    print(
        f"\nDone. {n_rows} datapoints from {n_ok} cache record(s), "
        f"{n_failures} failed reads."
    )
    print(f"Output: {output_file}")
    print(f"Log file: {log_path}")
    if n_failures:
        print(
            f"Review failures in: "
            f"{config.EXTERNAL_DIR / 'bacdive_cache_failures.csv'}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
