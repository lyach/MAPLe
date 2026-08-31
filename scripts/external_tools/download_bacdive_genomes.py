#!/usr/bin/env python3
"""Downloads genome + CDS FASTA files from NCBI for every unique
``genome_accession`` in data/external/bacdive_growth_full.parquet.

For each accession (e.g. "GCA_900699165") this:
  1. Resolves it to NCBI's current *versioned* accession (e.g.
     "GCA_900699165.1") via the Datasets API's dataset_report endpoint --
     the download endpoint silently returns an empty package for an
     unversioned accession, so this step is required, not optional.
  2. Downloads the NCBI Datasets zip package for that accession, requesting
     both the genome FASTA and the CDS-from-genomic FASTA.
  3. Extracts the two FASTA files into
     data/external/bacdive_genomes/<accession>/.

Genomes with no CDS annotation (common for draft/scaffold-level assemblies)
are recorded as "unavailable", not treated as an error.

USAGE:
    python scripts/download_bacdive_genomes.py
    python scripts/download_bacdive_genomes.py --workers 8 --force
    python scripts/download_bacdive_genomes.py --limit 20   # smoke test

An NCBI API key is optional but raises the rate limit; set it via the
NCBI_API_KEY environment variable (same variable used elsewhere in this
project, see src/strain_metadata/credentials.py) or --api-key.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.strain_metadata.logging_setup import setup_logging

log = logging.getLogger("download_bacdive_genomes")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "external" / "bacdive_growth_full.parquet"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "external" / "bacdive_genomes"
DEFAULT_MANIFEST = ROOT / "data" / "external" / "bacdive_genome_manifest.csv"

DATASETS_API_ROOT = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession"
REQUEST_TIMEOUT = 300  # seconds; genome+CDS packages can be tens of MB
MAX_ATTEMPTS = 3
NO_CDS_MARKER = "NO_CDS_AVAILABLE"


class RateLimiter:
    """Caps the global rate of NCBI requests across every worker thread.

    NCBI's Datasets API reports a per-second limit in its response headers
    (5 req/s anonymous, 10 req/s with an API key) -- we stay comfortably
    under that with a shared minimum interval between request *starts*,
    rather than trying to parse the headers live."""

    def __init__(self, min_interval: float):
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last = time.monotonic()


@dataclass
class AccessionResult:
    accession: str
    resolved_accession: str = ""
    assembly_level: str = ""        # e.g. Complete Genome | Chromosome | Scaffold | Contig
    genome_status: str = "failed"   # ok | skipped | unavailable | failed
    cds_status: str = "failed"      # ok | skipped | unavailable | failed
    genome_path: str = ""
    cds_path: str = ""
    error: str = ""


ACCESSION_RE = re.compile(r"^GC[AF]_\d+(\.\d+)?$")


def load_accessions(input_path: Path, column: Optional[str] = None) -> list[str]:
    """Unique NCBI assembly accessions from a parquet or CSV input.

    A ``.parquet`` is read as the BacDive dataset (``genome_accession``); a
    ``.csv`` is read as a resolver manifest such as
    ``genome_strain_matches.csv`` (``accession`` by default). Either way,
    anything that is not a ``GCA_``/``GCF_`` accession is dropped -- the
    Datasets API only understands those, and ``genome_accession`` may now hold
    an IMG or BV-BRC id (see ``genome_accession_source``).
    """
    if input_path.suffix == ".parquet":
        column = column or "genome_accession"
        df = pd.read_parquet(input_path, columns=[column])
    else:
        column = column or "accession"
        df = pd.read_csv(input_path, usecols=[column], dtype=str)

    values = df[column].dropna().astype(str).str.strip()
    values = values[values != ""]
    accessions = sorted(set(values))
    usable = [a for a in accessions if ACCESSION_RE.match(a)]
    dropped = len(accessions) - len(usable)
    if dropped:
        log.warning(
            "Dropped %d non-NCBI accession(s) from %s -- not GCA_/GCF_ "
            "(IMG and BV-BRC ids cannot be fetched from the Datasets API)",
            dropped, input_path.name,
        )
    return usable


def _http_get(url: str, headers: dict, rate_limiter: RateLimiter) -> bytes:
    """GET a URL with retries + exponential backoff on transient failures.
    Raises on a non-retryable HTTP error (e.g. 404) or after exhausting
    retries on a retryable one."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        rate_limiter.wait()
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in (429, 500, 502, 503, 504) or attempt == MAX_ATTEMPTS:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_exc = exc
            if attempt == MAX_ATTEMPTS:
                raise
        wait_seconds = 2 ** attempt
        log.debug("Request to %s failed (attempt %d/%d): %s -- retrying in %ds",
                   url, attempt, MAX_ATTEMPTS, last_exc, wait_seconds)
        time.sleep(wait_seconds)
    raise last_exc  # unreachable, but keeps type checkers happy


def resolve_current_accession(
    accession: str, headers: dict, rate_limiter: RateLimiter
) -> tuple[Optional[str], str]:
    """Look up the current versioned accession NCBI has on file for a
    (possibly unversioned) accession, plus its reported assembly level
    (Complete Genome / Chromosome / Scaffold / Contig). Returns
    (None, "") if NCBI has no record at all for it (suppressed, replaced
    with no successor, or typo'd).

    This is a single lightweight JSON call (no sequence data), so it's
    safe to make even for accessions whose FASTA files are already on
    disk -- it's how we backfill assembly_level without re-downloading
    anything."""
    url = f"{DATASETS_API_ROOT}/{urllib.parse.quote(accession)}/dataset_report"
    try:
        raw = _http_get(url, headers, rate_limiter)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None, ""
        raise
    data = json.loads(raw)
    reports = data.get("reports") or []
    if not reports:
        return None, ""
    report = reports[0]
    resolved = report.get("current_accession") or report.get("accession") or accession
    assembly_level = (report.get("assembly_info") or {}).get("assembly_level", "")
    return resolved, assembly_level


def download_dataset_zip(
    accession: str, headers: dict, rate_limiter: RateLimiter, include_cds: bool
) -> bytes:
    include = "GENOME_FASTA,CDS_FASTA" if include_cds else "GENOME_FASTA"
    query = urllib.parse.urlencode({"include_annotation_type": include})
    url = f"{DATASETS_API_ROOT}/{urllib.parse.quote(accession)}/download?{query}"
    return _http_get(url, headers, rate_limiter)


def extract_genome_and_cds(zip_bytes: bytes) -> tuple[Optional[bytes], Optional[bytes]]:
    """Pulls the genomic FASTA and (if present) the CDS-from-genomic FASTA
    out of an NCBI Datasets zip package. Matched by filename suffix rather
    than parsed from dataset_catalog.json, since the suffixes are stable
    across NCBI Datasets versions and this avoids depending on the exact
    shape of that catalog file."""
    genome_bytes: Optional[bytes] = None
    cds_bytes: Optional[bytes] = None
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if name.endswith("cds_from_genomic.fna"):
                cds_bytes = zf.read(name)
            elif name.endswith("_genomic.fna"):
                genome_bytes = zf.read(name)
    return genome_bytes, cds_bytes


def already_done(genome_path: Path, cds_path: Path, no_cds_marker: Path) -> bool:
    if not genome_path.exists():
        return False
    return cds_path.exists() or no_cds_marker.exists()


def process_accession(
    accession: str, output_dir: Path, headers: dict, rate_limiter: RateLimiter, force: bool
) -> AccessionResult:
    result = AccessionResult(accession=accession)
    acc_dir = output_dir / accession
    genome_path = acc_dir / f"{accession}_genomic.fna"
    cds_path = acc_dir / f"{accession}_cds_from_genomic.fna"
    no_cds_marker = acc_dir / NO_CDS_MARKER

    # Always resolve, even for accessions we won't re-download -- it's one
    # small JSON call and it's the only way to backfill assembly_level for
    # files that already exist on disk.
    resolved, assembly_level = resolve_current_accession(accession, headers, rate_limiter)
    if resolved is None:
        result.error = "NCBI has no assembly record for this accession"
        return result
    result.resolved_accession = resolved
    result.assembly_level = assembly_level

    if not force and already_done(genome_path, cds_path, no_cds_marker):
        result.genome_status = "skipped"
        result.cds_status = "skipped" if cds_path.exists() else "unavailable"
        result.genome_path = str(genome_path)
        result.cds_path = str(cds_path) if cds_path.exists() else ""
        return result

    acc_dir.mkdir(parents=True, exist_ok=True)

    combined_error = ""
    try:
        zip_bytes = download_dataset_zip(resolved, headers, rate_limiter, include_cds=True)
        genome_bytes, cds_bytes = extract_genome_and_cds(zip_bytes)
    except Exception as exc:
        combined_error = str(exc)
        # Fall back to genome-only in case the combined request failed for
        # a reason specific to the CDS/annotation request.
        try:
            zip_bytes = download_dataset_zip(resolved, headers, rate_limiter, include_cds=False)
            genome_bytes, cds_bytes = extract_genome_and_cds(zip_bytes)
        except Exception as exc2:
            result.error = f"genome+CDS request failed ({combined_error}); genome-only retry also failed ({exc2})"
            return result

    if genome_bytes:
        genome_path.write_bytes(genome_bytes)
        result.genome_status = "ok"
        result.genome_path = str(genome_path)
    else:
        result.genome_status = "failed"
        result.error = combined_error or "no genomic FASTA in NCBI package"

    if cds_bytes:
        cds_path.write_bytes(cds_bytes)
        result.cds_status = "ok"
        result.cds_path = str(cds_path)
    else:
        result.cds_status = "unavailable"
        no_cds_marker.touch()
        if combined_error and not result.error:
            result.error = f"CDS unavailable (combined request failed: {combined_error})"

    return result


def print_summary(results: list[AccessionResult], manifest_path: Path) -> None:
    total = len(results)
    genome_ok = sum(r.genome_status == "ok" for r in results)
    genome_skipped = sum(r.genome_status == "skipped" for r in results)
    genome_failed = sum(r.genome_status == "failed" for r in results)

    cds_ok = sum(r.cds_status == "ok" for r in results)
    cds_unavailable = sum(r.cds_status == "unavailable" for r in results)
    cds_skipped = sum(r.cds_status == "skipped" for r in results)
    cds_failed = sum(r.cds_status == "failed" for r in results)

    print()
    print("=" * 72)
    print("BacDive genome/CDS download summary")
    print("=" * 72)
    print(f"Unique genome_accession values requested: {total}")
    print()
    print("Genome FASTA:")
    print(f"  downloaded OK      : {genome_ok}")
    print(f"  already on disk    : {genome_skipped}")
    print(f"  FAILED             : {genome_failed}")
    print()
    print("CDS FASTA:")
    print(f"  downloaded OK      : {cds_ok}")
    print(f"  not available (no annotation for this assembly): {cds_unavailable}")
    print(f"  already on disk    : {cds_skipped}")
    print(f"  FAILED (other reason): {cds_failed}")
    print()
    total_available = genome_ok + genome_skipped
    total_missing = genome_failed
    print(f"Total genomes available (downloaded or cached): {total_available}/{total}")
    print(f"Total genomes NOT available: {total_missing}/{total}")

    level_counts: dict[str, int] = {}
    for r in results:
        if r.assembly_level:
            level_counts[r.assembly_level] = level_counts.get(r.assembly_level, 0) + 1
    if level_counts:
        print()
        print("Assembly level breakdown:")
        for level, count in sorted(level_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {level:<20}: {count}")

    failures = [r for r in results if r.genome_status == "failed"]
    if failures:
        print()
        print(f"First {min(20, len(failures))} genome failures:")
        for r in failures[:20]:
            print(f"  {r.accession}: {r.error}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more (see manifest)")

    print()
    print(f"Full manifest written to: {manifest_path}")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                         help="Path to the BacDive growth parquet, or a "
                              "resolver manifest CSV such as "
                              "genome_strain_matches.csv")
    parser.add_argument("--accession-col", default=None,
                         help="Column holding the accession (default: "
                              "genome_accession for parquet, accession for CSV)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                         help="Directory to write per-accession genome/CDS FASTA files")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                         help="Path to write the per-accession result manifest CSV")
    parser.add_argument("--workers", type=int, default=6,
                         help="Number of concurrent download threads")
    parser.add_argument("--force", action="store_true",
                         help="Re-download even if files already exist on disk")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N accessions (for smoke testing)")
    parser.add_argument("--api-key", default=None,
                         help="NCBI API key (defaults to NCBI_API_KEY env var if set)")
    args = parser.parse_args()

    setup_logging(ROOT)

    if not args.input.exists():
        log.error("Input file not found: %s", args.input)
        return 1

    import os
    api_key = args.api_key or os.environ.get("NCBI_API_KEY", "").strip() or None
    headers = {"User-Agent": "MicrobeNet/download_bacdive_genomes", "Accept": "application/json"}
    if api_key:
        headers["api-key"] = api_key
    rate_limiter = RateLimiter(min_interval=0.12 if api_key else 0.35)

    accessions = load_accessions(args.input, args.accession_col)
    log.info("Loaded %d unique genome_accession value(s) from %s", len(accessions), args.input)
    if args.limit:
        accessions = accessions[: args.limit]
        log.info("Limiting to first %d accession(s) for this run", len(accessions))

    if not api_key:
        log.warning(
            "No NCBI API key set -- downloads will be rate-limited more "
            "aggressively. Set NCBI_API_KEY to speed this up."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    results: list[AccessionResult] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(process_accession, acc, args.output_dir, headers, rate_limiter, args.force): acc
            for acc in accessions
        }
        progress = tqdm(as_completed(futures), total=len(futures), desc="BacDive genomes", unit="genome")
        ok = failed = no_cds = 0
        for future in progress:
            result = future.result()
            results.append(result)
            if result.genome_status == "failed":
                failed += 1
                tqdm.write(f"[FAILED] {result.accession}: {result.error}")
            elif result.genome_status in ("ok", "skipped"):
                ok += 1
            if result.cds_status == "unavailable":
                no_cds += 1
            progress.set_postfix(ok=ok, failed=failed, no_cds=no_cds)

    manifest_df = pd.DataFrame([r.__dict__ for r in results])
    manifest_df.to_csv(args.manifest, index=False)

    print_summary(results, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
