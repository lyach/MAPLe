#!/usr/bin/env python3
"""
build_bacdive_from_cache.py
===========================
Rebuild a BacDive modeling table from ``data/external/bacdive_cache/*.json``
without calling the BacDive API.

Defaults to the target's complete column set (``--full-schema``, itself
default-on); pass ``--no-full-schema`` to use whatever subset
``bacdive/build_from_cache.yaml``'s ``strain_features``/``medium_features``
currently restrict to instead (see also ``src/bacdive/config.py``). That
YAML's ``strain_metadata`` key, if set, names a CSV that is left-merged onto
the table on ``strain_name`` == ``strain``. Extraction / dedupe match
``get_bacdive_dataset.py``; only the fetch step is skipped.

USAGE:
    python scripts/bacdive/build_bacdive_from_cache.py
    python scripts/bacdive/build_bacdive_from_cache.py --no-temperature-range-expansion
    python scripts/bacdive/build_bacdive_from_cache.py --no-full-schema
    python scripts/bacdive/build_bacdive_from_cache.py --no-strain-metadata
    python scripts/bacdive/build_bacdive_from_cache.py --strain-metadata data/external/bd_strains_metadata.csv
    python scripts/bacdive/build_bacdive_from_cache.py --config bacdive/build_from_cache.yaml
    python scripts/bacdive/build_bacdive_from_cache.py --target metabolite_utilization
    python scripts/bacdive/build_bacdive_from_cache.py --limit 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _find_root(start: Path) -> Path:
    """Ascend from ``start`` to the directory containing ``pyproject.toml``."""
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return start.parents[2]


ROOT = _find_root(Path(__file__).resolve())
sys.path.insert(0, str(ROOT))

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
        help="Override target from the YAML config",
    )
    p.add_argument(
        "--full-schema", action=argparse.BooleanOptionalAction, default=True,
        help="Use the target's complete default column set "
             "(config.TARGETS[target].fieldnames) -- the default. Pass "
             "--no-full-schema to use whatever subset the YAML config's "
             "strain_features/medium_features currently restrict to "
             "instead. Output file is determined independently: uses --out if "
             "provided, else the YAML config's output_file if set, else the "
             "target's default. Note: --target alone does NOT get you the "
             "YAML's column subset when it happens to equal the YAML's own "
             "target (as medium_growth normally does) -- that combination "
             "still uses --full-schema's default. Pass --no-full-schema "
             "explicitly whenever you want the YAML's current restrictions.",
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
        "--strain-metadata", type=Path, default=None,
        help="Override strain_metadata from the YAML config: a CSV of "
             "per-strain metadata to left-merge onto the output, joining the "
             "table's strain_name to the CSV's strain column. Columns already "
             "in the table (genus/family/order/phylum, ...) are kept from the "
             "table and dropped from the CSV. Pass --no-strain-metadata to "
             "skip the merge even when the YAML config sets it.",
    )
    p.add_argument(
        "--no-strain-metadata", dest="strain_metadata", action="store_const",
        const=Path(""),
        help="Skip the strain-metadata merge even if the YAML config sets it.",
    )
    p.add_argument(
        "--temperature-range-expansion", action=argparse.BooleanOptionalAction,
        default=None,
        help="Override temperature_range_expansion from the YAML config. "
             "Changes row count (a range like \"10-41\" becomes 1 midpoint "
             "row, or 3 min/max/mean rows), so the resolved value is always "
             "stamped onto the output filename -- see "
             "config.stamp_temperature_expansion.",
    )
    p.add_argument(
        "--dedup-w-negatives", action=argparse.BooleanOptionalAction,
        default=None,
        help="Override dedup_w_negatives from the YAML config. When true, "
             "an (isolate, unit) pair with both a growth and a no-growth "
             "label keeps both rows instead of resolving to positive. "
             "Changes row count, so the resolved value is always stamped "
             "onto the output filename -- see config.stamp_dedup_w_negatives.",
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
    expansion = (
        cfg.temperature_range_expansion
        if args.temperature_range_expansion is None
        else args.temperature_range_expansion
    )
    dedup_w_negatives = (
        cfg.dedup_w_negatives
        if args.dedup_w_negatives is None
        else args.dedup_w_negatives
    )
    if args.strain_metadata is None:
        strain_metadata = cfg.strain_metadata
    elif str(args.strain_metadata) in ("", "."):
        strain_metadata = None
    else:
        strain_metadata = args.strain_metadata
    if strain_metadata is not None and not strain_metadata.is_absolute():
        strain_metadata = config.ROOT / strain_metadata
    if strain_metadata is not None and not strain_metadata.is_file():
        log.error("strain metadata CSV not found: %s", strain_metadata)
        return 1

    use_full_schema = args.full_schema or target != cfg.target
    if use_full_schema:
        if not args.full_schema:
            log.warning(
                "CLI --target %r differs from YAML target %r; "
                "using the full default schema for %r",
                target, cfg.target, target,
            )
        base = config.TARGETS[target]
        fieldnames = list(base.fieldnames)
        categorical_features = list(base.categorical_features)
    else:
        fieldnames = cfg.fieldnames
        categorical_features = cfg.categorical_features

    # Determine output_file
    if args.out:
        output_file = args.out
    else:
        output_file = cfg.output_file or config.TARGETS[target].output_file
    output_writer = cfg.output_writer

    if not output_file.is_absolute():
        output_file = config.ROOT / output_file
    # Stamped from the resolved `expansion`/`dedup_w_negatives`, not the raw
    # path above -- so an --out (or YAML output_file) that doesn't already
    # name its settings can never collide with a differently-configured
    # build's output.
    output_file = config.stamp_temperature_expansion(output_file, expansion)
    output_file = config.stamp_dedup_w_negatives(output_file, dedup_w_negatives)

    if not cache_dir.is_dir():
        log.error("cache dir not found: %s", cache_dir)
        return 1

    log.info(
        "building from cache %s (target=%s, %d output columns, "
        "temperature_range_expansion=%s, dedup_w_negatives=%s, "
        "strain_metadata=%s) -> %s",
        cache_dir, target, len(fieldnames), expansion, dedup_w_negatives,
        strain_metadata, output_file,
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
        temperature_range_expansion=expansion,
        dedup_w_negatives=dedup_w_negatives,
        strain_metadata=strain_metadata,
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
