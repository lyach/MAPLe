"""
config.py
=========
All the paths and tunable constants for the strain-metadata pipeline live
here. Keeping them in one place (instead of scattered across modules) is
what makes the rest of the package "modular" -- every other file imports
from here rather than redefining its own copy of a path or a magic number.

Defaults defined here can be overridden at runtime by an optional
settings.yaml at the project root (loaded by load_settings() below) or by
command-line flags, without editing this file.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("config")

# --------------------------------------------------------------------------
# PROJECT LAYOUT
# --------------------------------------------------------------------------

# config.py -> strain_metadata/ -> scripts/ -> <project root>
ROOT = Path(__file__).resolve().parents[2]

# --- inputs (data/metadata/) ---
INPUT_CSV          = ROOT / "data" / "metadata" / "align_strains.csv"
NAME_OVERRIDES_CSV = ROOT / "data" / "metadata" / "strain_name_overrides.csv"

# Two-tier manual override system (see overrides.py):
#   FORCED   -- pinned accessions that always win, bypassing all search
#   FALLBACK -- last-resort hints, tried only after all search strategies fail
FORCED_ACCESSIONS_CSV = ROOT / "data" / "metadata" / "forced_accessions.csv"
FALLBACK_HINTS_CSV    = ROOT / "data" / "metadata" / "fallback_hints.csv"

# --- outputs (data/external/) ---
EXTERNAL_DIR = ROOT / "data" / "external"
FASTA_DIR    = EXTERNAL_DIR / "fasta"                     # one .fna per assembly
GBK_DIR      = EXTERNAL_DIR / "gbk"                       # one .gbk per assembly
OUTPUT_CSV   = EXTERNAL_DIR / "strains_metadata.csv"      # successful rows
FAILURES_CSV = EXTERNAL_DIR / "failures.csv"              # diagnostic failure report
PHYLUM_CSV   = EXTERNAL_DIR / "strain_phylum.csv"          # ncbi_taxon_id -> phylum lookup

# --- logs ---
LOGS_DIR = ROOT / "logs"                                  # timestamped run logs

# --- credentials (gitignored) ---
CREDENTIALS_FILE = ROOT / ".ncbi_credentials.json"

# --- optional settings override file ---
SETTINGS_YAML = ROOT / "settings.yaml"

# --------------------------------------------------------------------------
# NCBI BEHAVIOR TUNABLES (overridable via settings.yaml / CLI)
# --------------------------------------------------------------------------

# NCBI rate limits: ~3 req/s without an API key, ~10 req/s with one.
# These are pause-between-requests values (seconds) that stay under each.
REQUEST_DELAY_NO_API_KEY   = 0.34
REQUEST_DELAY_WITH_API_KEY = 0.11

# How many times to retry a failed NCBI call before giving up (exponential
# backoff between attempts).
RETRY_ATTEMPTS = 3

# Download timeout for FASTA/GBK files from NCBI's FTP, in seconds.
DOWNLOAD_TIMEOUT_SECONDS = 120

# --------------------------------------------------------------------------
# OUTPUT SCHEMA
# --------------------------------------------------------------------------

# Column order for strains_metadata.csv. The last two columns are the
# confidence/provenance flags added in the redesign:
#   resolution_method     -- which strategy matched this strain
#   requires_verification -- "yes" if the match wasn't strain-confirmed, or
#                            if a validation flag fired
OUTPUT_FIELDNAMES = [
    "strain",
    "atcc_number",
    "ncbi_refseq",
    "ncbi_assembly_accession",
    "ncbi_taxon_id",
    "assembly_level",
    "genome_size_bp",
    "gc_content_pct",
    "num_contigs",
    "n50",
    "sequence_hash",
    "date_retrieved",
    "resolution_method",
    "requires_verification",
]

# Columns for the failures.csv diagnostic report.
FAILURE_FIELDNAMES = [
    "strain",
    "atcc_number",
    "strategies_tried",
    "rejected_candidates",
    "ncbi_search_url",
]

# NCBI's free-text completeness labels mapped to short lowercase forms.
ASSEMBLY_LEVEL_MAP = {
    "complete genome": "complete",
    "chromosome":      "chromosome",
    "scaffold":        "scaffold",
    "contig":          "contig",
}

# Ordering used for tie-breaking when multiple assemblies match the same
# strain. Lower number = higher priority (preferred).
ASSEMBLY_LEVEL_PRIORITY = {
    "Complete Genome": 0,
    "Chromosome":      1,
    "Scaffold":        2,
    "Contig":          3,
}

# --------------------------------------------------------------------------
# settings.yaml loader
# --------------------------------------------------------------------------

def load_settings(path: Path = None) -> dict:
    """Load optional overrides from settings.yaml if it exists. Returns a
    dict of any settings found (empty if no file). Silently returns empty
    if PyYAML isn't installed, so the dependency stays optional.

    Recognised keys (all optional):
        request_delay_no_api_key: float
        request_delay_with_api_key: float
        retry_attempts: int
        download_timeout_seconds: int
        genus_ranges: { Genus: [min_mb, max_mb, min_gc, max_gc], ... }
    """
    path = path or SETTINGS_YAML
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        log.warning(
            "settings.yaml found at %s but PyYAML isn't installed "
            "(pip install pyyaml) -- ignoring it and using code defaults.",
            path,
        )
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        log.info("Loaded settings overrides from %s: %s", path, list(data.keys()))
        return data
    except Exception as exc:
        log.warning("Failed to parse %s (%s) -- using code defaults.", path, exc)
        return {}
