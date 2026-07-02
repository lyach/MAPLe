"""
overrides.py
============
Two-tier manual override system for strain → accession resolution.

TIER 1 -- FORCED (forced_accessions.csv)
    Pinned accessions that ALWAYS win. When a strain appears here, the
    pipeline skips all NCBI search entirely and uses this accession. Use
    this when you have manually verified the correct assembly and want to
    guarantee it is used every time, with no ambiguity.

TIER 2 -- FALLBACK HINTS (fallback_hints.csv)
    Last-resort hints, tried only AFTER all automated NCBI search strategies
    have failed -- but BEFORE the strain is declared a failure. Use this for
    strains where you have a reasonable guess (e.g. an accession you think is
    right but haven't fully verified) and would rather the pipeline try it
    than give up. A strain resolved via a fallback hint is flagged
    requires_verification = yes in the output, since the hint is unverified.

Both files share the same format:

    strain,accession,note
    Caulobacter vibrioides,GCF_000006905.2,CB15 reference — verified on NCBI

Only `strain` and `accession` are required; `note` is free text for your own
record-keeping and is echoed into the run log.

Keeping these as two separate files (rather than one file with a priority
column) makes the intent unambiguous at a glance: anything in
forced_accessions.csv is trusted, anything in fallback_hints.csv is a guess.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("overrides")

_STRAIN_COLS    = ["strain", "strain_name", "name"]
_ACCESSION_COLS = ["accession", "ncbi_accession", "assembly_accession"]
_NOTE_COLS      = ["note", "notes", "reason"]


def _find_col(fieldnames: list[str], candidates: list[str]) -> Optional[str]:
    """Case-insensitive lookup of the first matching column header."""
    lowered = {fn.strip().lower(): fn for fn in fieldnames}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _load_override_file(path: Path, tier_label: str) -> dict[str, str]:
    """Shared loader for both forced and fallback files. Returns a dict
    mapping strain name → accession. Validates accession format and logs
    each entry for the audit trail."""
    if not path.exists():
        log.debug("No %s file at %s -- skipping this tier.", tier_label, path)
        return {}

    result: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        strain_col    = _find_col(fieldnames, _STRAIN_COLS)
        accession_col = _find_col(fieldnames, _ACCESSION_COLS)
        note_col      = _find_col(fieldnames, _NOTE_COLS)

        if not strain_col or not accession_col:
            log.warning(
                "%s (%s) is missing required columns (need 'strain' and "
                "'accession', found: %s) -- ignoring it.",
                tier_label, path, fieldnames,
            )
            return {}

        for row in reader:
            strain    = (row.get(strain_col)    or "").strip()
            accession = (row.get(accession_col) or "").strip()
            note      = (row.get(note_col)      or "").strip() if note_col else ""

            if not strain or not accession:
                continue

            # NCBI assembly accessions always start GCF_ (RefSeq) or GCA_
            # (GenBank). Reject obviously malformed values rather than letting
            # them fail mysteriously deeper in the pipeline.
            if not (accession.startswith("GCF_") or accession.startswith("GCA_")):
                log.warning(
                    "%s entry for %r has unexpected accession format %r "
                    "(expected GCF_... or GCA_...) -- skipping this row.",
                    tier_label, strain, accession,
                )
                continue

            result[strain] = accession
            suffix = f"  ({note})" if note else ""
            log.info("%s: %r -> %s%s", tier_label, strain, accession, suffix)

    log.info("Loaded %d %s entry(ies) from %s", len(result), tier_label, path)
    return result


def load_forced_accessions(path: Path) -> dict[str, str]:
    """Tier 1: accessions that bypass all search and always win."""
    return _load_override_file(path, "forced accession")


def load_fallback_hints(path: Path) -> dict[str, str]:
    """Tier 2: accessions tried only after all search strategies fail."""
    return _load_override_file(path, "fallback hint")


def resolve_via_accession(accession: str, ncbi_client) -> Optional[object]:
    """Fetch a ResolvedAssembly for a specific accession directly, bypassing
    search. Shared by both override tiers and used by the pipeline. Returns
    None if NCBI has no record for the accession (e.g. it was mistyped or has
    been suppressed)."""
    from Bio import Entrez

    log.debug("Resolving pinned accession via NCBI: %s", accession)
    try:
        search_result = ncbi_client._retry(
            Entrez.esearch, db="assembly", term=accession, retmax=5,
        )
    except Exception as exc:
        log.error("esearch failed for accession %s: %s", accession, exc)
        return None

    id_list = search_result.get("IdList", [])
    if not id_list:
        log.error(
            "Accession %s returned no NCBI results -- check it's correct in "
            "your override file (it may be suppressed, withdrawn, or mistyped).",
            accession,
        )
        return None

    try:
        summary_result = ncbi_client._retry(Entrez.esummary, db="assembly", id=id_list[0])
    except Exception as exc:
        log.error("esummary failed for accession %s: %s", accession, exc)
        return None

    summaries = summary_result.get("DocumentSummarySet", {}).get("DocumentSummary", [])
    if not summaries:
        log.error("No DocumentSummary returned for accession %s.", accession)
        return None

    return ncbi_client._parse_summary_to_resolved(summaries[0])
