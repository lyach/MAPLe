#!/usr/bin/env python3
"""
get_strain_metadata.py
=======================
Pulls NCBI assembly metadata for each strain in data/metadata/align_strains.csv,
downloads the genome FASTA + GenBank files, computes derived sequence stats,
validates them against genus expectations, and writes:

    data/external/strains_metadata.csv   -- successful rows (+ confidence flags)
    data/external/failures.csv           -- diagnostics for unresolved strains
    logs/run_<timestamp>.log             -- full forensic log of the run

This file is intentionally thin: it parses CLI flags, sets up logging, resolves
credentials, and hands off to the strain_metadata package. All logic lives in:

    strain_metadata/config.py            paths, constants, settings.yaml loader
    strain_metadata/logging_setup.py      console + timestamped-file logging
    strain_metadata/credentials.py         NCBI email/API key resolution
    strain_metadata/align_loader.py         reads align_strains.csv + name overrides
    strain_metadata/overrides.py             forced + fallback accession overrides
    strain_metadata/ncbi_client.py            all NCBI Entrez/FTP calls + strategy chain
    strain_metadata/sequence_stats.py          GC%/N50/hash from FASTA
    strain_metadata/validation.py               genus-range sanity checks
    strain_metadata/taxonomy.py                  NCBI lineage (genus/family/order/phylum)
    strain_metadata/resolution_report.py         per-strain attempt tracking + summaries
    strain_metadata/pipeline.py                   orchestration

RESOLUTION STRATEGY (per strain, stops at first success):
    0. Forced accession   (forced_accessions.csv -- always wins)
    1. NCBI ATCC assembly search
    2. NCBI BioSample -> assembly cross-reference (organism-validated)
    3. NCBI organism-name search (broad -- flags requires_verification)
    4. Fallback hint      (fallback_hints.csv -- last resort)
    -> failures.csv if all fail

NCBI ACCOUNT: on first run you'll be prompted for an email (saved locally,
gitignored). Or set NCBI_ENTREZ_EMAIL / NCBI_API_KEY env vars.

USAGE:
    python scripts/get_strain_metadata.py
    python scripts/get_strain_metadata.py --limit 3 --debug
    python scripts/get_strain_metadata.py --force          # re-download cached files
    python scripts/get_strain_metadata.py --quiet          # warnings/errors on console only
    python scripts/get_strain_metadata.py --reset-credentials
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
from src.strain_metadata import pipeline

log = logging.getLogger("get_strain_metadata")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", type=Path, default=config.INPUT_CSV,
                   help=f"Path to align_strains.csv (default: {config.INPUT_CSV})")
    p.add_argument("--output", type=Path, default=config.OUTPUT_CSV,
                   help=f"Path to strains_metadata.csv (default: {config.OUTPUT_CSV})")
    p.add_argument("--force", action="store_true",
                   help="Re-download FASTA/GBK even if cached")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N strains (quick test)")
    p.add_argument("--debug", action="store_true",
                   help="Show DEBUG detail on the console too (always in the log file)")
    p.add_argument("--quiet", action="store_true",
                   help="Console shows warnings/errors only (full detail still logged)")
    p.add_argument("--reset-credentials", action="store_true",
                   help="Forget saved NCBI email/API key and prompt again")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # --- logging first, so everything from here is captured ---
    console_level = logging.DEBUG if args.debug else logging.INFO
    log_path = setup_logging(
        config.ROOT, console_level=console_level, quiet_console=args.quiet,
    )

    # --- optional settings.yaml overrides ---
    settings = config.load_settings()
    genus_ranges = None
    if "genus_ranges" in settings:
        # settings provides {Genus: [min_mb, max_mb, min_gc, max_gc]}; merge
        # over the built-in defaults so a partial file only overrides what it lists.
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

    ncbi_client = NCBIAssemblyClient(email=email, api_key=api_key)

    rows_written, num_failures = pipeline.run(
        ncbi_client=ncbi_client,
        input_csv=args.input,
        output_csv=args.output,
        failures_csv=config.FAILURES_CSV,
        name_overrides_csv=config.NAME_OVERRIDES_CSV,
        forced_accessions_csv=config.FORCED_ACCESSIONS_CSV,
        fallback_hints_csv=config.FALLBACK_HINTS_CSV,
        fasta_dir=config.FASTA_DIR,
        gbk_dir=config.GBK_DIR,
        genus_ranges=genus_ranges,
        force=args.force,
        limit=args.limit,
        debug=args.debug,
    )

    print(f"\nDone. {rows_written} resolved, {num_failures} failed.")
    print(f"Log file: {log_path}")
    if num_failures:
        print(f"Review failures in: {config.FAILURES_CSV}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
