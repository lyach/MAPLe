#!/usr/bin/env python3
"""
get_strain_metadata_bacdive.py
==============================
Pull NCBI assembly metadata for each unique isolate in a BacDive parquet
(e.g. data/external/bacdive_growth_full.parquet), without downloading
FASTA/GBK.

Resolution uses the isolate-identity columns BacDive already stores
(genome_accession, culture_collection_ids, dsm_number, strain-level
ncbi_taxid, ...). Organism-name search is off -- a species hit is not
this strain's genome.

Writes:
    data/external/bacdive_strains_metadata.csv   -- one row per resolved isolate
    data/external/bacdive_strain_meta_failures.csv
    <input_stem>_strain_meta.parquet             -- original table + NCBI columns
    logs/run_<timestamp>.log

USAGE:
    python scripts/external_tools/get_strain_metadata_bacdive.py --limit 20 --debug
    python scripts/external_tools/get_strain_metadata_bacdive.py
    python scripts/external_tools/get_strain_metadata_bacdive.py \\
        --input data/external/bacdive_growth_full.parquet \\
        --output data/external/bacdive_growth_full_strain_meta.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.strain_metadata import config
from src.strain_metadata.logging_setup import setup_logging
from src.strain_metadata.credentials import resolve_ncbi_credentials, clear_saved_credentials
from src.strain_metadata.ncbi_client import NCBIAssemblyClient
from src.strain_metadata.bacdive_loader import (
    load_bacdive_strains,
    merge_metadata_onto_parquet,
)
from src.strain_metadata import pipeline

log = logging.getLogger("get_strain_metadata_bacdive")


def default_parquet_output(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_strain_meta{input_path.suffix}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=Path, default=config.BACDIVE_INPUT_PARQUET,
                   help=f"BacDive parquet (default: {config.BACDIVE_INPUT_PARQUET})")
    p.add_argument("--output", type=Path, default=None,
                   help="Merged parquet path (default: <input_stem>_strain_meta.parquet)")
    p.add_argument("--metadata-csv", type=Path, default=config.BACDIVE_METADATA_CSV,
                   help=f"Sidecar isolate metadata CSV (default: {config.BACDIVE_METADATA_CSV})")
    p.add_argument("--failures-csv", type=Path, default=config.BACDIVE_FAILURES_CSV,
                   help=f"Failure diagnostics CSV (default: {config.BACDIVE_FAILURES_CSV})")
    p.add_argument("--force", action="store_true",
                   help="Re-resolve isolates even if they are already in the sidecar CSV")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N unique isolates (quick test)")
    p.add_argument("--debug", action="store_true",
                   help="Show DEBUG detail on the console too (always in the log file)")
    p.add_argument("--quiet", action="store_true",
                   help="Console shows warnings/errors only (full detail still logged)")
    p.add_argument("--reset-credentials", action="store_true",
                   help="Forget saved NCBI email/API key and prompt again")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output_parquet = args.output or default_parquet_output(args.input)

    console_level = logging.DEBUG if args.debug else logging.INFO
    log_path = setup_logging(
        config.ROOT, console_level=console_level, quiet_console=args.quiet,
    )

    settings = config.load_settings()
    genus_ranges = None
    if "genus_ranges" in settings:
        from src.strain_metadata.validation import GENUS_RANGES
        genus_ranges = dict(GENUS_RANGES)
        for genus, vals in settings["genus_ranges"].items():
            genus_ranges[genus] = tuple(vals)
        log.info("Applied %d genus-range override(s) from settings.yaml",
                 len(settings["genus_ranges"]))

    if args.reset_credentials and clear_saved_credentials():
        log.info("Cleared saved NCBI credentials")

    try:
        email, api_key = resolve_ncbi_credentials()
    except RuntimeError as exc:
        log.error(str(exc))
        return 1

    try:
        strains = load_bacdive_strains(args.input)
    except (FileNotFoundError, ValueError) as exc:
        log.error(str(exc))
        return 1

    ncbi_client = NCBIAssemblyClient(email=email, api_key=api_key)

    rows_written, num_failures = pipeline.run(
        ncbi_client=ncbi_client,
        strains=strains,
        output_csv=args.metadata_csv,
        failures_csv=args.failures_csv,
        name_overrides_csv=config.NAME_OVERRIDES_CSV,
        forced_accessions_csv=config.FORCED_ACCESSIONS_CSV,
        fallback_hints_csv=config.FALLBACK_HINTS_CSV,
        fasta_dir=config.FASTA_DIR,
        gbk_dir=config.GBK_DIR,
        genus_ranges=genus_ranges,
        force=args.force,
        limit=args.limit,
        debug=args.debug,
        download_sequences=False,
        include_organism_search=False,
        metadata_cache_csv=args.metadata_csv,
        extra_output_fields=["bacdive_id"],
        cache_key_field="bacdive_id",
    )

    try:
        n_out = merge_metadata_onto_parquet(
            args.input, args.metadata_csv, output_parquet,
        )
    except Exception as exc:
        log.error("Failed to merge metadata onto parquet: %s", exc)
        return 1

    print(f"\nDone. {rows_written} isolate(s) in sidecar, {num_failures} failed this run.")
    print(f"Sidecar: {args.metadata_csv}")
    print(f"Merged parquet ({n_out} rows): {output_parquet}")
    print(f"Log file: {log_path}")
    if num_failures:
        print(f"Review failures in: {args.failures_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
