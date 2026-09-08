"""
pipeline.py
===========
Orchestrates the full per-strain workflow and writes all outputs:
  * strains_metadata.csv  -- successful rows, with taxonomy ranks
                             (genus/family/order/phylum)
  * failures.csv          -- diagnostics for strains that failed every strategy
  * console + log summary  -- end-of-run resolution breakdown

Takes plain arguments (paths, an NCBIAssemblyClient, loaded override dicts)
so it can be driven from the CLI, a notebook, or a test equally easily.
"""

from __future__ import annotations

import csv
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Optional

from .align_loader import (
    StrainInput,
    inferred_atcc_number,
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
from .taxonomy import LINEAGE_RANKS, fetch_lineage
from .validation import validate_genome_stats
from tqdm import tqdm

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
    download_sequences: bool = True,
    include_organism_search: bool = True,
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
        include_organism_search=include_organism_search,
    )

    if resolved is None:
        log.info("No assembly for %s", canonical_name)
        return None, report

    accession  = resolved.assembly_accession or resolved.assembly_uid
    fasta_path = fasta_dir / f"{accession}.fna"
    gbk_path   = gbk_dir   / f"{accession}.gbk"
    sequence_stats: dict = {}

    if download_sequences:
        if not resolved.ftp_path:
            log.warning("No FTP path for %s (%s) -- metadata only.",
                        canonical_name, accession)
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
    else:
        log.info("Skipping FASTA/GBK download for %s (metadata only)", accession)

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

    validate_name = strain_input.organism_name or canonical_name
    report.validation_warnings = validate_genome_stats(
        validate_name, genome_size_bp, gc_content_pct, genus_ranges,
    )

    atcc = inferred_atcc_number(strain_input) or (strain_input.atcc_number or "")
    row = {
        "strain":                  canonical_name,
        "atcc_number":             atcc,
        "ncbi_refseq":             resolved.refseq_accession,
        "ncbi_assembly_accession": resolved.assembly_accession,
        "ncbi_taxon_id":           resolved.taxon_id if resolved.taxon_id is not None else "",
        "genus":                   "",
        "family":                  "",
        "order":                   "",
        "phylum":                  "",
        "assembly_level":          resolved.assembly_level,
        "genome_size_bp":          genome_size_bp,
        "gc_content_pct":          gc_content_pct,
        "num_contigs":             num_contigs,
        "n50":                     n50,
        "sequence_hash":           sequence_hash,
        "date_retrieved":          date.today().isoformat(),
        "resolution_method":       report.winning_strategy.label() if report.winning_strategy else "",
        "requires_verification":   "yes" if report.requires_verification else "no",
    }
    if strain_input.strain_id:
        row["bacdive_id"] = strain_input.strain_id
    return row, report


def _enrich_rows_with_lineage(
    rows: list[dict],
    email: str,
    api_key: Optional[str] = None,
) -> None:
    """Fill genus/family/order/phylum on each row from NCBI Taxonomy (in place)."""
    taxon_ids: list[int] = []
    for row in rows:
        raw = row.get("ncbi_taxon_id", "")
        if raw == "" or raw is None:
            continue
        try:
            taxon_ids.append(int(raw))
        except (TypeError, ValueError):
            log.warning("Invalid ncbi_taxon_id %r for strain %r -- skipping lineage",
                        raw, row.get("strain"))

    if not taxon_ids:
        log.warning("No taxon IDs available -- genus/family/order/phylum left blank")
        return

    log.info("Looking up taxonomy lineage for %d unique taxon ID(s)", len(set(taxon_ids)))
    lineage = fetch_lineage(taxon_ids, email=email, api_key=api_key)

    for row in rows:
        raw = row.get("ncbi_taxon_id", "")
        if raw == "" or raw is None:
            continue
        try:
            ranks = lineage.get(int(raw), {})
        except (TypeError, ValueError):
            continue
        for rank in LINEAGE_RANKS:
            row[rank] = ranks.get(rank, "")


def _row_needs_lineage(row: dict) -> bool:
    return not any(row.get(rank) for rank in LINEAGE_RANKS)


def _cache_key(row: dict, key_field: str) -> str:
    return str(row.get(key_field) or row.get("strain") or "").strip()


def _load_cached_rows(path: Path, key_field: str) -> tuple[list[dict], set[str]]:
    """Load previously resolved metadata rows for resume."""
    if not path.exists():
        return [], set()
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keys = {_cache_key(r, key_field) for r in rows}
    keys.discard("")
    log.info("Resume cache: %d row(s) already in %s", len(rows), path)
    return rows, keys


def _append_metadata_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _write_metadata_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(
    ncbi_client: NCBIAssemblyClient,
    input_csv: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    failures_csv: Optional[Path] = None,
    name_overrides_csv: Optional[Path] = None,
    forced_accessions_csv: Optional[Path] = None,
    fallback_hints_csv: Optional[Path] = None,
    fasta_dir: Optional[Path] = None,
    gbk_dir: Optional[Path] = None,
    genus_ranges: Optional[dict] = None,
    force: bool = False,
    limit: Optional[int] = None,
    debug: bool = False,
    strains: Optional[list[StrainInput]] = None,
    download_sequences: bool = True,
    include_organism_search: bool = True,
    metadata_cache_csv: Optional[Path] = None,
    extra_output_fields: Optional[list[str]] = None,
    cache_key_field: str = "bacdive_id",
    workers: int = 1,
) -> tuple[int, int]:
    """Run the pipeline end-to-end. Writes strains_metadata.csv + failures.csv,
    prints a summary. Returns (rows_written, num_failures).

    `workers` > 1 overlaps NCBI HTTP latency across isolates. A shared
    RateLimiter on the client keeps the process under Entrez's cap
    (3 req/s without an API key, 10 req/s with one), so extra threads
    wait on the limiter rather than 429'ing NCBI.
    """

    if strains is None:
        if input_csv is None:
            raise ValueError("run() needs either strains= or input_csv=")
        strains = load_align_strains(input_csv)
    if output_csv is None:
        raise ValueError("run() needs output_csv=")
    if failures_csv is None:
        raise ValueError("run() needs failures_csv=")
    if fasta_dir is None or gbk_dir is None:
        raise ValueError("run() needs fasta_dir= and gbk_dir=")

    if download_sequences:
        fasta_dir.mkdir(parents=True, exist_ok=True)
        gbk_dir.mkdir(parents=True, exist_ok=True)

    if limit:
        strains = strains[:limit]

    name_overrides    = load_name_overrides(name_overrides_csv) if name_overrides_csv else {}
    forced_accessions = load_forced_accessions(forced_accessions_csv) if forced_accessions_csv else {}
    fallback_hints    = load_fallback_hints(fallback_hints_csv) if fallback_hints_csv else {}

    fieldnames = list(OUTPUT_FIELDNAMES)
    if extra_output_fields:
        for name in extra_output_fields:
            if name not in fieldnames:
                fieldnames.insert(0, name)

    cached_rows: list[dict] = []
    cached_keys: set[str] = set()
    if metadata_cache_csv and not force:
        cached_rows, cached_keys = _load_cached_rows(metadata_cache_csv, cache_key_field)
    elif metadata_cache_csv and force:
        # Drop the sidecar so this run starts clean; incremental appends rebuild it.
        metadata_cache_csv.unlink(missing_ok=True)

    rows: list[dict] = list(cached_rows)
    reports: list[StrainResolutionReport] = []
    pending = [s for s in strains if s.join_key not in cached_keys]
    n_skipped = len(strains) - len(pending)
    n_ok = 0
    n_fail = 0
    workers = max(1, int(workers))
    if workers > 1:
        log.info("Resolving %d isolate(s) with %d worker thread(s)", len(pending), workers)

    def _one(strain_input: StrainInput):
        return process_one_strain(
            strain_input, name_overrides, forced_accessions, fallback_hints,
            ncbi_client, fasta_dir, gbk_dir, genus_ranges, force, debug,
            download_sequences=download_sequences,
            include_organism_search=include_organism_search,
        )

    write_lock = threading.Lock()
    pbar = tqdm(total=len(pending), desc="NCBI resolve", unit="isolate", dynamic_ncols=True)

    def _record(strain_input: StrainInput, row, report) -> None:
        nonlocal n_ok, n_fail
        with write_lock:
            reports.append(report)
            if row is not None:
                n_ok += 1
                rows.append(row)
                cached_keys.add(strain_input.join_key)
                if metadata_cache_csv:
                    _append_metadata_row(metadata_cache_csv, row, fieldnames)
            else:
                n_fail += 1
            pbar.set_postfix(ok=n_ok, fail=n_fail, refresh=False)
            pbar.update(1)

    if workers == 1:
        for strain_input in pending:
            try:
                row, report = _one(strain_input)
            except Exception as exc:
                log.error("Unexpected error processing %r: %s", strain_input.strain, exc)
                report = StrainResolutionReport(
                    strain=strain_input.strain,
                    atcc_number=strain_input.atcc_number,
                )
                row = None
            _record(strain_input, row, report)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_one, s): s for s in pending}
            for fut in as_completed(futs):
                strain_input = futs[fut]
                try:
                    row, report = fut.result()
                except Exception as exc:
                    log.error("Unexpected error processing %r: %s",
                              strain_input.strain, exc)
                    report = StrainResolutionReport(
                        strain=strain_input.strain,
                        atcc_number=strain_input.atcc_number,
                    )
                    row = None
                _record(strain_input, row, report)

    pbar.close()

    if n_skipped:
        log.info("Skipped %d isolate(s) already in %s", n_skipped, metadata_cache_csv)

    # Taxonomy lookup for rows that still lack lineage ranks (new + stale cache).
    to_enrich = [r for r in rows if _row_needs_lineage(r)]
    if to_enrich:
        try:
            _enrich_rows_with_lineage(
                to_enrich, email=ncbi_client.email, api_key=ncbi_client.api_key,
            )
        except Exception as exc:
            log.error("Taxonomy lineage lookup failed (%s) -- writing blank rank columns", exc)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_metadata_csv(output_csv, rows, fieldnames)
    log.info("Wrote %d row(s) to %s", len(rows), output_csv)

    num_failures = write_failures_csv(reports, failures_csv)
    if num_failures:
        log.info("Wrote %d failed strain(s) to %s", num_failures, failures_csv)

    summary = build_run_summary(reports)
    if n_skipped:
        summary = summary.replace(
            "NCBI RESOLUTION SUMMARY",
            f"NCBI RESOLUTION SUMMARY\n  Skipped (already cached) : {n_skipped}",
            1,
        )
    print(summary)
    log.info("Run summary:\n%s", summary)

    return len(rows), num_failures
