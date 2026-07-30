#!/usr/bin/env python3
"""
get_bacdive_dataset.py
======================
Download a taxonomy-scoped subset of BacDive (DSMZ) and build a tidy
``(strain x culture medium)`` table with a binary ``growth_class`` target that
feeds the modeling pipeline (``docs/modeling_pipeline.md``).

Writes:
    data/external/bacdive_growth.csv     -- one row per (strain, culture medium)
    data/external/bacdive_failures.csv   -- taxa that returned nothing / errored
    data/external/bacdive_cache/*.json   -- raw per-record cache
    logs/run_<timestamp>.log             -- full run log

The taxonomic scope is read from bacdive/taxa.yaml (edit it to control which
strains are downloaded). The public BacDive v2 API needs no credentials.

USAGE:
    python scripts/get_bacdive_dataset.py
    python scripts/get_bacdive_dataset.py --taxa bacdive/taxa.yaml
    python scripts/get_bacdive_dataset.py --limit 5          # quick smoke test
    python scripts/get_bacdive_dataset.py --out data/external/bacdive_growth.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bacdive import config, pipeline
from src.strain_metadata.logging_setup import setup_logging

log = logging.getLogger("get_bacdive_dataset")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--taxa", type=Path, default=config.TAXA_YAML,
                   help=f"Path to taxa.yaml (default: {config.TAXA_YAML})")
    p.add_argument("--out", type=Path, default=config.OUTPUT_CSV,
                   help=f"Output CSV path (default: {config.OUTPUT_CSV})")
    p.add_argument("--cache-dir", type=Path, default=config.CACHE_DIR,
                   help=f"Raw record cache dir (default: {config.CACHE_DIR})")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N records (quick test)")
    p.add_argument("--force", action="store_true",
                   help="Re-fetch records even if cached")
    p.add_argument("--debug", action="store_true",
                   help="Show DEBUG detail on the console too (always in the log file)")
    p.add_argument("--quiet", action="store_true",
                   help="Console shows warnings/errors only (full detail still logged)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    console_level = logging.DEBUG if args.debug else logging.INFO
    log_path = setup_logging(
        config.ROOT, console_level=console_level, quiet_console=args.quiet,
    )

    if not args.taxa.exists():
        log.error("taxa file not found: %s", args.taxa)
        return 1

    taxa = config.load_taxa(args.taxa)
    if not taxa:
        log.error("no taxa listed in %s", args.taxa)
        return 1
    log.info("loaded %d taxa from %s", len(taxa), args.taxa)

    n_rows, n_taxa_ok, n_failures = pipeline.run(
        taxa=taxa,
        output_csv=args.out,
        failures_csv=config.FAILURES_CSV,
        cache_dir=args.cache_dir,
        limit=args.limit,
        force=args.force,
    )

    print(f"\nDone. {n_rows} datapoints from {n_taxa_ok} taxa, {n_failures} failed.")
    print(f"Output: {args.out}")
    print(f"Log file: {log_path}")
    if n_failures:
        print(f"Review failures in: {config.FAILURES_CSV}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
