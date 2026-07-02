"""
resolution_report.py
=====================
Tracks every NCBI resolution attempt made for each strain during a run, and
generates both a console summary and the failures.csv diagnostic file.

Two consumers:
  * build_run_summary()  -> human-readable end-of-run summary (console + log)
  * write_failures_csv() -> machine-readable diagnostics for failed strains,
                            with everything you need to resolve them manually
"""

from __future__ import annotations

import csv
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class Strategy(str, Enum):
    """The resolution strategy that was attempted, in priority order."""
    FORCED_ACCESSION    = "forced_accession"     # forced_accessions.csv (always wins)
    NCBI_ATCC_ASSEMBLY  = "ncbi_atcc_assembly"   # "ATCC nnnnn" in assembly db
    NCBI_BIOSAMPLE_LINK = "ncbi_biosample_link"  # ATCC -> BioSample -> assembly
    NCBI_ORGANISM       = "ncbi_organism"         # "[Organism]" field (broad)
    NCBI_FREETEXT       = "ncbi_freetext"         # plain free-text (broadest)
    FALLBACK_HINT       = "fallback_hint"          # fallback_hints.csv (last resort)

    def label(self) -> str:
        return {
            Strategy.FORCED_ACCESSION:    "Forced accession (override file)",
            Strategy.NCBI_ATCC_ASSEMBLY:  "NCBI assembly search by ATCC number",
            Strategy.NCBI_BIOSAMPLE_LINK: "NCBI BioSample -> assembly cross-reference",
            Strategy.NCBI_ORGANISM:       "NCBI assembly search by organism name",
            Strategy.NCBI_FREETEXT:       "NCBI assembly free-text search",
            Strategy.FALLBACK_HINT:       "Fallback hint (override file)",
        }[self]


# Strategies that confirm strain identity (vs. organism-level guesses).
_STRAIN_CONFIRMED_STRATEGIES = (
    Strategy.FORCED_ACCESSION,
    Strategy.NCBI_ATCC_ASSEMBLY,
    Strategy.NCBI_BIOSAMPLE_LINK,
)


@dataclass
class Attempt:
    """One resolution attempt -- what was tried, what came back."""
    strategy: Strategy
    query: str
    result: Optional[str]   # accession resolved, or None if this attempt failed
    note: str = ""


@dataclass
class StrainResolutionReport:
    """Full resolution history + validation outcome for one strain."""
    strain: str
    atcc_number: str
    attempts: list[Attempt] = field(default_factory=list)
    final_accession: Optional[str] = None
    winning_strategy: Optional[Strategy] = None
    validation_warnings: list[str] = field(default_factory=list)
    organism_mismatch: bool = False     # set if BioSample organism != expected

    def add_attempt(self, strategy, query, result=None, note=""):
        self.attempts.append(Attempt(strategy, query, result, note))

    def record_success(self, strategy, accession):
        self.winning_strategy = strategy
        self.final_accession  = accession

    @property
    def resolved(self) -> bool:
        return self.final_accession is not None

    @property
    def is_strain_confirmed(self) -> bool:
        """True only if matched via a strain-specific identifier AND no
        organism mismatch was detected."""
        return (
            self.winning_strategy in _STRAIN_CONFIRMED_STRATEGIES
            and not self.organism_mismatch
        )

    @property
    def requires_verification(self) -> bool:
        """The flag written into strains_metadata.csv. True (=needs a human
        to check) when the match wasn't strain-confirmed, when a fallback
        hint was used, when the organism didn't match, or when a genome-stat
        validation flag fired."""
        if not self.is_strain_confirmed:
            return True
        if self.validation_warnings:
            return True
        return False

    @property
    def rejected_candidates(self) -> list[str]:
        """Accessions that were found during search but not chosen as the
        final result -- useful when manually resolving a failure."""
        seen = []
        for a in self.attempts:
            if a.result and a.result != self.final_accession and a.result not in seen:
                seen.append(a.result)
        return seen


def _ncbi_search_url(strain: str, atcc_number: str) -> str:
    """Build a ready-to-paste NCBI assembly search URL using the strain's
    best identifiers, so resolving a failure is one click away."""
    if atcc_number:
        term = f'"{strain}" OR "ATCC {atcc_number}"'
    else:
        term = f'"{strain}"'
    return "https://www.ncbi.nlm.nih.gov/datasets/genome/?search=" + urllib.parse.quote(term)


def build_run_summary(reports: list[StrainResolutionReport]) -> str:
    """Human-readable end-of-run summary for console + log."""
    resolved   = [r for r in reports if r.resolved]
    failed     = [r for r in reports if not r.resolved]
    confirmed  = [r for r in resolved if r.is_strain_confirmed]
    needs_ver  = [r for r in resolved if r.requires_verification]
    flagged    = [r for r in resolved if r.validation_warnings]

    lines = [
        "",
        "=" * 70,
        "  NCBI RESOLUTION SUMMARY",
        "=" * 70,
        f"  Total strains processed   : {len(reports)}",
        f"  Successfully resolved     : {len(resolved)}",
        f"    - Strain-confirmed      : {len(confirmed)}",
        f"    - Needs verification    : {len(needs_ver)}",
        f"  Failed all strategies     : {len(failed)}",
        f"  Validation flags raised   : {len(flagged)}",
        "-" * 70,
    ]

    from collections import Counter
    method_counts = Counter(r.winning_strategy.label() for r in resolved if r.winning_strategy)
    if method_counts:
        lines.append("  Resolution method breakdown:")
        for label, count in method_counts.most_common():
            lines.append(f"    {count:>3}x  {label}")
        lines.append("-" * 70)

    if needs_ver:
        lines.append(f"  STRAINS NEEDING VERIFICATION ({len(needs_ver)}):")
        for r in needs_ver:
            reason = []
            if not r.is_strain_confirmed:
                reason.append("organism-level match" if r.winning_strategy else "unconfirmed")
            if r.organism_mismatch:
                reason.append("ORGANISM MISMATCH")
            if r.validation_warnings:
                reason.append(f"{len(r.validation_warnings)} validation flag(s)")
            lines.append(f"    - {r.strain}  ->  {r.final_accession}  [{'; '.join(reason)}]")
            for w in r.validation_warnings:
                lines.append(f"        ! {w}")
        lines.append("-" * 70)

    if failed:
        lines.append(f"  FAILED STRAINS ({len(failed)}) -- see failures.csv for details:")
        for r in failed:
            lines.append(f"    - {r.strain} (ATCC {r.atcc_number})")
        lines.append("=" * 70)
    else:
        lines.append("  No failures.")
        lines.append("=" * 70)

    return "\n".join(lines)


def write_failures_csv(reports: list[StrainResolutionReport], path: Path) -> int:
    """Write the diagnostic failures.csv for every strain that failed all
    strategies. Returns the number of failed strains written. Always writes
    the file (with header) even if there are zero failures, so a stale
    failures.csv from a previous run never lingers."""
    from .config import FAILURE_FIELDNAMES

    failed = [r for r in reports if not r.resolved]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FAILURE_FIELDNAMES)
        writer.writeheader()
        for r in failed:
            # Summarise every attempt as "strategy: reason"
            tried = "; ".join(
                f"{a.strategy.label()} ({a.note or 'no result'})"
                for a in r.attempts
            )
            writer.writerow({
                "strain":              r.strain,
                "atcc_number":         r.atcc_number,
                "strategies_tried":    tried,
                "rejected_candidates": ", ".join(r.rejected_candidates) or "none",
                "ncbi_search_url":     _ncbi_search_url(r.strain, r.atcc_number),
            })
    return len(failed)
