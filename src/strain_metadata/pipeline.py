"""
pipeline.py
===========
Orchestrates the full per-strain workflow and writes all outputs:
  * strains_metadata.csv  -- successful rows, with resolution_method and
                             requires_verification confidence columns
  * failures.csv          -- diagnostics for strains that failed every strategy
  * console + log summary  -- end-of-run resolution breakdown

Takes plain arguments (paths, an NCBIAssemblyClient, loaded override dicts)
so it can be driven from the CLI, a notebook, or a test equally easily.
"""

from __future__ import annotations

import csv
import logging
from datetime import date
from pathlib import Path
from typing import Optional

from .align_loader import (
    StrainInput,
    load_align_strains,
    load_name_overrides,
    normalize_strain_name,
)
from .config import OUTPUT_FIELDNAMES
from .ncbi_client import NCBIAssemblyClient
from .overrides import load_forced_accessions, load_fallback_hints
from .resolution_report import (
    StrainResolutionReport,
    build_run_summary,
    write_failures_csv,
)
from .sequence_stats import summarize_fasta
from .validation import validate_genome_stats

log = logging.getLogger("pipeline")


def process_one_strain(
    strain_input: StrainInput,
    name_overrides: dict[str, str],
    forced_accessions: dict[str, str],
    fallback_hints: dict[str, str],
    ncbi_client: NCBIAssemblyClient,
    fasta_dir: Path,
    gbk_dir: Path,
    genus_ranges: Optional[dict],
    force: bool,
    debug: bool,
) -> tuple[Optional[dict], StrainResolutionReport]:
    """Run the full per-strain workflow. Returns (output_row, report).
    output_row is None only if all resolution strategies failed.
    report always carries the full attempt log + validation outcome."""

    canonical_name = normalize_strain_name(strain_input.strain, name_overrides)
    log.info("Processing strain: %s (raw: %s)", canonical_name, strain_input.strain)

    resolved, report = ncbi_client.resolve_assembly(
        strain_input,
        forced_accessions=forced_accessions,
        fallback_hints=fallback_hints,
        debug=debug,
    )

    if resolved is None:
        log.warning("FAILED to resolve %s", canonical_name)
        return None, report

    accession  = resolved.assembly_accession or resolved.assembly_uid
    fasta_path = fasta_dir / f"{accession}.fna"
    gbk_path   = gbk_dir   / f"{accession}.gbk"
    sequence_stats: dict = {}

    if not resolved.ftp_path:
        log.warning("No FTP path for %s (%s) -- metadata only.", canonical_name, accession)
    else:
        already_cached = fasta_path.exists() and gbk_path.exists()
        if force or not already_cached:
            log.info("Downloading FASTA + GBK for %s", accession)
            try:
                ncbi_client.download_via_ftp(resolved.ftp_path, fasta_path, gbk_path)
            except Exception as exc:
                log.error("Download failed for %s: %s", canonical_name, exc)
        else:
            log.info("Using cached FASTA/GBK for %s", accession)

        if fasta_path.exists():
            try:
                sequence_stats = summarize_fasta(fasta_path)
            except Exception as exc:
                log.error("FASTA parse failed for %s: %s", canonical_name, exc)

    # Prefer computed stats; fall back to NCBI-reported figures.
    genome_size_bp = (sequence_stats.get("genome_size_bp")
                      or resolved.meta_stats.get("total_length", ""))
    num_contigs    = (sequence_stats.get("num_contigs")
                      or resolved.meta_stats.get("contig_count", ""))
    n50            = (sequence_stats.get("n50")
                      or resolved.meta_stats.get(
                          "contig_n50", resolved.meta_stats.get("scaffold_n50", "")))
    gc_content_pct = sequence_stats.get("gc_content_pct", "")
    sequence_hash  = sequence_stats.get("sequence_hash", "")

    # --- second accuracy layer: validate stats against genus ranges ---
    report.validation_warnings = validate_genome_stats(
        canonical_name, genome_size_bp, gc_content_pct, genus_ranges,
    )

    row = {
        "strain":                  canonical_name,
        "atcc_number":             strain_input.atcc_number,
        "ncbi_refseq":             resolved.refseq_accession,
        "ncbi_assembly_accession": resolved.assembly_accession,
        "ncbi_taxon_id":           resolved.taxon_id if resolved.taxon_id is not None else "",
        "assembly_level":          resolved.assembly_level,
        "genome_size_bp":          genome_size_bp,
        "gc_content_pct":          gc_content_pct,
        "num_contigs":             num_contigs,
        "n50":                     n50,
        "sequence_hash":           sequence_hash,
        "date_retrieved":          date.today().isoformat(),
        # confidence / provenance columns
        "resolution_method":       report.winning_strategy.label() if report.winning_strategy else "",
        "requires_verification":   "yes" if report.requires_verification else "no",
    }
    return row, report


def run(
    ncbi_client: NCBIAssemblyClient,
    input_csv: Path,
    output_csv: Path,
    failures_csv: Path,
    name_overrides_csv: Path,
    forced_accessions_csv: Path,
    fallback_hints_csv: Path,
    fasta_dir: Path,
    gbk_dir: Path,
    genus_ranges: Optional[dict] = None,
    force: bool = False,
    limit: Optional[int] = None,
    debug: bool = False,
) -> tuple[int, int]:
    """Run the pipeline end-to-end. Writes strains_metadata.csv + failures.csv,
    prints a summary. Returns (rows_written, num_failures)."""

    fasta_dir.mkdir(parents=True, exist_ok=True)
    gbk_dir.mkdir(parents=True, exist_ok=True)

    strains           = load_align_strains(input_csv)
    if limit:
        strains = strains[:limit]
    name_overrides    = load_name_overrides(name_overrides_csv)
    forced_accessions = load_forced_accessions(forced_accessions_csv)
    fallback_hints    = load_fallback_hints(fallback_hints_csv)

    rows: list[dict] = []
    reports: list[StrainResolutionReport] = []

    for strain_input in strains:
        try:
            row, report = process_one_strain(
                strain_input, name_overrides, forced_accessions, fallback_hints,
                ncbi_client, fasta_dir, gbk_dir, genus_ranges, force, debug,
            )
        except Exception as exc:
            log.error("Unexpected error processing %r: %s", strain_input.strain, exc)
            report = StrainResolutionReport(
                strain=strain_input.strain,
                atcc_number=strain_input.atcc_number,
            )
            row = None

        reports.append(report)
        if row is not None:
            rows.append(row)

    # --- write successful rows ---
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote %d row(s) to %s", len(rows), output_csv)

    # --- write failure diagnostics (always, even if zero) ---
    num_failures = write_failures_csv(reports, failures_csv)
    if num_failures:
        log.info("Wrote %d failed strain(s) to %s", num_failures, failures_csv)

    # --- print summary to console + log ---
    summary = build_run_summary(reports)
    print(summary)
    log.info("Run summary:\n%s", summary)

    return len(rows), num_failures
