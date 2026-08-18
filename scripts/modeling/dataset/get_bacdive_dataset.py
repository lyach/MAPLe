#!/usr/bin/env python3
"""
get_bacdive_dataset.py
======================
Download a taxonomy-scoped subset of BacDive (DSMZ) and build a tidy table with
a binary ``growth_class`` target that feeds the modeling pipeline
(``docs/modeling_pipeline.md``). The extraction target selects what one row is:

    medium_growth           -- one row per (strain, culture medium)
    metabolite_utilization  -- one row per (strain, metabolite, assay kind)

Writes (target-dependent output name):
    data/external/bacdive_growth.csv | bacdive_metabolite.csv
    data/external/bacdive_failures.csv   -- taxa that returned nothing / errored
    data/external/bacdive_cache/*.json   -- raw per-record cache
    logs/run_<timestamp>.log             -- full run log

The taxonomic scope and target are read from bacdive/taxa.yaml (edit it to
control what is downloaded). The public BacDive v2 API needs no credentials.

USAGE:
    python scripts/get_bacdive_dataset.py
    python scripts/get_bacdive_dataset.py --taxa bacdive/taxa.yaml
    python scripts/get_bacdive_dataset.py --target metabolite_utilization
    python scripts/get_bacdive_dataset.py --limit 5          # quick smoke test
    python scripts/get_bacdive_dataset.py --no-progress       # disable tqdm bars
    python scripts/get_bacdive_dataset.py --max-workers 10 --taxa-workers 8
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
    p.add_argument("--target", choices=sorted(config.TARGETS), default=None,
                   help="Extraction target (default: 'target' key in taxa.yaml, "
                        f"else {config.DEFAULT_TARGET})")
    p.add_argument("--out", type=Path, default=None,
                   help="Output CSV path (default: resolved from the target)")
    p.add_argument("--cache-dir", type=Path, default=config.CACHE_DIR,
                   help=f"Raw record cache dir (default: {config.CACHE_DIR})")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N records (quick test)")
    p.add_argument("--force", action="store_true",
                   help="Re-fetch records even if cached")
    p.add_argument("--max-workers", type=int, default=config.MAX_WORKERS,
                   help="Concurrent threads fetching record batches within a "
                        f"taxon (default: {config.MAX_WORKERS})")
    p.add_argument("--taxa-workers", type=int, default=config.TAXA_WORKERS,
                   help="Taxa processed concurrently when >1 is listed "
                        f"(default: {config.TAXA_WORKERS})")
    p.add_argument("--debug", action="store_true",
                   help="Show DEBUG detail on the console too (always in the log file)")
    p.add_argument("--quiet", action="store_true",
                   help="Console shows warnings/errors only (full detail still logged)")
    p.add_argument("--no-progress", action="store_true",
                   help="Disable tqdm progress bars (e.g. for non-interactive/log-only runs)")
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

    try:
        target = args.target or config.load_target(args.taxa)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    output_file = args.out or config.TARGETS[target].output_file
    log.info("loaded %d taxa from %s (target: %s)", len(taxa), args.taxa, target)

    n_rows, n_taxa_ok, n_failures = pipeline.run(
        taxa=taxa,
        target=target,
        output_file=output_file,
        failures_csv=config.FAILURES_CSV,
        cache_dir=args.cache_dir,
        limit=args.limit,
        force=args.force,
        show_progress=not (args.no_progress or args.quiet),
        max_workers=args.max_workers,
        taxa_workers=args.taxa_workers,
    )

    print(f"\nDone. {n_rows} datapoints from {n_taxa_ok} taxa, {n_failures} failed.")
    print(f"Output: {output_file}")
    print(f"Log file: {log_path}")
    if n_failures:
        print(f"Review failures in: {config.FAILURES_CSV}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
