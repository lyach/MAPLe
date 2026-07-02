"""
validation.py
=============
A second, independent accuracy layer. After a strain is resolved and its
genome statistics computed, we sanity-check those numbers against expected
ranges for that organism's genus. This catches the case where the resolution
chain picked the WRONG assembly but everything else looked fine -- e.g. if a
Bacillus strain resolves to an assembly that's only 1.2 Mb when Bacillus
genomes are ~4-5 Mb, that's a strong signal something went wrong.

This is deliberately independent of the resolution logic: resolution checks
"did we find the right NCBI record?", validation checks "do the numbers look
biologically plausible?". Two different questions, two different failure modes
caught.

The genus ranges below are compiled from NCBI genome statistics and published
genome reports for the genera in the project's strain panel. They are
intentionally a little wide (covering known strain-to-strain variation) so
that only genuine outliers are flagged, not normal biological diversity.

Ranges can be extended or overridden via settings.yaml (see config.py) without
editing this file.
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger("validation")

# genus -> (min_genome_Mb, max_genome_Mb, min_gc_pct, max_gc_pct)
# Sizes in megabases (Mb); the pipeline works in base pairs, so we convert.
GENUS_RANGES: dict[str, tuple[float, float, float, float]] = {
    "Acetobacter":         (2.5, 4.5, 50, 65),
    "Gluconobacter":       (2.5, 4.0, 54, 62),
    "Komagataeibacter":    (3.0, 4.5, 58, 64),
    "Caulobacter":         (3.5, 5.5, 62, 68),
    "Paracoccus":          (4.0, 5.5, 62, 68),
    "Cupriavidus":         (5.5, 8.5, 62, 68),
    "Acinetobacter":       (3.0, 4.5, 38, 44),
    "Azotobacter":         (4.5, 5.5, 64, 68),
    "Escherichia":         (4.5, 5.8, 49, 52),
    "Pseudomonas":         (5.5, 7.5, 58, 67),
    "Arthrobacter":        (3.5, 5.5, 59, 67),
    "Bifidobacterium":     (1.8, 2.9, 56, 64),
    "Corynebacterium":     (2.5, 3.5, 52, 56),
    "Kocuria":             (2.5, 3.5, 68, 74),
    "Micrococcus":         (2.2, 3.0, 65, 73),
    "Bacillus":            (3.8, 5.8, 42, 47),
    "Lactiplantibacillus": (3.0, 3.5, 44, 47),
    "Lactobacillus":       (1.8, 3.5, 33, 51),
    "Pediococcus":         (1.7, 2.2, 36, 42),
    "Priestia":            (4.8, 5.8, 36, 40),   # formerly Bacillus megaterium
    "Streptococcus":       (1.7, 2.2, 36, 42),
    "Clostridium":         (3.5, 4.5, 26, 32),
    "Deinococcus":         (2.8, 3.5, 65, 70),
}

# Used when a strain's genus isn't in the table above. Very wide -- only
# flags truly absurd values (e.g. a near-empty or impossibly huge genome).
DEFAULT_RANGE = (1.0, 15.0, 20.0, 75.0)


def extract_genus(strain_name: str) -> str:
    """Pull the genus (first word) from a strain name. Handles the project's
    'Genus species; substrain' convention and parenthetical authorities like
    'Micrococcus luteus (Schroeter) Cohn' -- the genus is always the first
    whitespace-delimited token."""
    # First token before any space, semicolon, or parenthesis.
    token = strain_name.strip().split()[0] if strain_name.strip() else ""
    # Strip any trailing punctuation just in case.
    return token.strip(";,.()")


def validate_genome_stats(
    strain_name: str,
    genome_size_bp,
    gc_content_pct,
    genus_ranges: Optional[dict] = None,
) -> list[str]:
    """Check genome size and GC% against the expected range for the strain's
    genus. Returns a list of human-readable warning strings (empty if all
    checks pass). Does NOT raise -- validation flags issues, it doesn't stop
    the pipeline.

    Missing/blank values (e.g. metadata-only strains with no computed stats)
    are skipped rather than flagged, since "no data" isn't the same as "bad
    data"."""
    ranges = genus_ranges or GENUS_RANGES
    warnings: list[str] = []

    genus = extract_genus(strain_name)
    min_mb, max_mb, min_gc, max_gc = ranges.get(genus, DEFAULT_RANGE)
    using_default = genus not in ranges

    # --- genome size check ---
    if genome_size_bp not in ("", None):
        try:
            size_mb = float(genome_size_bp) / 1_000_000
            if size_mb < min_mb or size_mb > max_mb:
                range_note = " (default range — genus not in table)" if using_default else ""
                warnings.append(
                    f"genome size {size_mb:.2f} Mb is outside expected "
                    f"{min_mb}-{max_mb} Mb for {genus}{range_note}"
                )
        except (TypeError, ValueError):
            pass  # non-numeric -- nothing to validate

    # --- GC content check ---
    if gc_content_pct not in ("", None):
        try:
            gc = float(gc_content_pct)
            if gc < min_gc or gc > max_gc:
                range_note = " (default range — genus not in table)" if using_default else ""
                warnings.append(
                    f"GC content {gc:.1f}% is outside expected "
                    f"{min_gc}-{max_gc}% for {genus}{range_note}"
                )
        except (TypeError, ValueError):
            pass

    if warnings:
        for w in warnings:
            log.warning("Validation flag for %r: %s", strain_name, w)

    return warnings
