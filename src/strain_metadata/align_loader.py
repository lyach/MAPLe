"""
align_loader.py
================
Everything related to reading the project's *input* files: the raw strain
list (align_strains.csv) and the optional manual name-override table.

This is split out from the NCBI-querying code on purpose: parsing your lab's
spreadsheet export has nothing to do with talking to NCBI, and keeping them
separate means either one can change without touching the other.
"""

from __future__ import annotations

# csv: standard-library module for reading comma-separated files.
import csv
# re: used to collapse "Genus species; strain-tag" into a clean search string.
import re
# logging: structured, leveled console output instead of bare print().
import logging
# dataclass: a lightweight class that just holds a few named fields.
from dataclasses import dataclass
# Path: filesystem paths.
from pathlib import Path
# Optional: type hint meaning "this value or None".
from typing import Optional

# All log messages from this module are tagged "align_loader" so you can
# tell at a glance which part of the pipeline produced them.
log = logging.getLogger("align_loader")


@dataclass
class StrainInput:
    """One row from align_strains.csv, after parsing. `strain` is kept
    exactly as written in the source file (including things like the
    "Genus species; substrain" convention) so it round-trips cleanly into
    the output CSV unless a name override says otherwise."""
    strain: str
    atcc_number: str = ""


# Column-header text we'll accept for the strain-name column, tried in
# order, compared case-insensitively. Add more variants here if a future
# export uses a different header.
STRAIN_COLUMN_CANDIDATES = ["strain", "strain name", "strain_name", "name", "isolate"]

# Same idea, for the ATCC-number column. Note this matches the real header
# text "ATCC number" (lowercased) from your align file.
ATCC_COLUMN_CANDIDATES = [
    "atcc number", "atcc_number", "atcc no", "atcc_no", "atcc num", "atcc_num", "atcc",
]


def _find_column_index(header: list[str], candidates: list[str]) -> Optional[int]:
    """Return the position of the first header cell (case-insensitively)
    matching any of `candidates`, or None if none match.

    We deliberately work with the header as a plain list and match by
    *position* rather than using csv.DictReader's dict-of-columns approach.
    Your align file has multiple unnamed columns (blank header text) sitting
    between the named ones -- DictReader would silently collapse those
    duplicate "" keys into one, quietly dropping data. Matching by position
    avoids that trap entirely, since we never rely on header text being
    unique."""
    # Build a lowercase, whitespace-trimmed copy of the header once, so we
    # can compare candidates against it cheaply.
    lowered = [h.strip().lower() for h in header]
    for cand in candidates:
        if cand in lowered:
            return lowered.index(cand)
    return None


def clean_strain_name_for_search(raw_name: str) -> str:
    """Your align file encodes sub-strains as "Genus species; tag", e.g.
    "Escherichia coli; K-12" or "Bifidobacterium animalis; subsp. lactis".
    NCBI's organism/strain text doesn't use semicolons like that, so for
    *search purposes only* we turn "; " (or a bare ";") into a plain space,
    giving "Escherichia coli K-12" / "Bifidobacterium animalis subsp.
    lactis". This does NOT affect what gets written to the output CSV --
    only the text we send to NCBI as a search query."""
    # Replace one-or-more semicolons (optionally followed by whitespace)
    # with a single space.
    collapsed = re.sub(r";\s*", " ", raw_name)
    # Collapse any resulting double spaces (e.g. from "X;  Y") down to one,
    # and strip leading/trailing whitespace.
    return re.sub(r"\s+", " ", collapsed).strip()


def load_align_strains(path: Path) -> list[StrainInput]:
    """Read align_strains.csv and return one StrainInput per non-empty row.

    Raises FileNotFoundError with an actionable message if the file is
    missing, and ValueError if a strain-name column can't be located at all
    (an ATCC-number column is treated as optional -- some labs don't track
    it for every strain)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find input file: {path}\n"
            f"Expected the strain list at data/metadata/ncbi/align_strains.csv "
            f"relative to the project root."
        )

    # newline="" is the standard-library-recommended way to open files for
    # the csv module (it handles \r\n itself; letting Python's normal text
    # mode touch newlines too can corrupt quoted multi-line fields).
    # encoding="utf-8-sig" transparently strips a leading byte-order-mark,
    # which is common in CSVs exported from Excel/Google Sheets.
    with path.open(newline="", encoding="utf-8-sig") as f:
        # Plain csv.reader yields each row as a list of strings -- no
        # column-name dictionary, which is exactly what we want given the
        # duplicate blank headers (see _find_column_index's docstring).
        reader = csv.reader(f)

        # The first row is the header. next() pulls it off so the loop
        # below only sees data rows.
        header = next(reader)

        strain_idx = _find_column_index(header, STRAIN_COLUMN_CANDIDATES)
        atcc_idx = _find_column_index(header, ATCC_COLUMN_CANDIDATES)

        if strain_idx is None:
            raise ValueError(
                f"Couldn't find a strain-name column in {path}. "
                f"Found header: {header}. "
                f"Add your actual header text to STRAIN_COLUMN_CANDIDATES "
                f"in align_loader.py."
            )
        if atcc_idx is None:
            log.warning(
                "No ATCC-number column found in %s (looked for: %s). "
                "Will search NCBI by strain name only.",
                path, ATCC_COLUMN_CANDIDATES,
            )

        rows: list[StrainInput] = []
        for raw_row in reader:
            # Skip completely blank lines (trailing newline at EOF, etc.)
            if not raw_row:
                continue

            # strain_idx is guaranteed to be a valid index into `header`,
            # but a malformed data row could be shorter than the header --
            # guard against an IndexError rather than crashing the whole run.
            strain = raw_row[strain_idx].strip() if strain_idx < len(raw_row) else ""
            if not strain:
                # A row with no strain name isn't useful -- skip it rather
                # than writing a half-empty output row.
                continue

            atcc = ""
            if atcc_idx is not None and atcc_idx < len(raw_row):
                atcc = raw_row[atcc_idx].strip()

            rows.append(StrainInput(strain=strain, atcc_number=atcc))

    log.info("Loaded %d strain(s) from %s", len(rows), path)
    return rows


# Column-header candidates for the (optional) manual override file. Unlike
# ncbi/align_strains.csv, this is a file *we* control the format of, so a plain
# DictReader is fine here -- no duplicate-blank-header problem to work
# around.
_OVERRIDE_RAW_CANDIDATES = ["raw_name", "raw", "align_strains_name"]
_OVERRIDE_CANONICAL_CANDIDATES = ["canonical_name", "canonical", "layout_name"]


def load_name_overrides(path: Path) -> dict[str, str]:
    """Read the optional strain_name_overrides.csv (columns: raw_name,
    canonical_name) used to align strain spelling with `_layout.csv`.
    Returns an empty dict (with a log message, not an error) if the file
    doesn't exist -- this file is opt-in."""
    if not path.exists():
        log.info(
            "No name-override file at %s -- strain names will be used "
            "as-is. If they don't match _layout.csv yet, create this file "
            "with columns raw_name,canonical_name.",
            path,
        )
        return {}

    overrides: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        raw_idx = _find_column_index(fieldnames, _OVERRIDE_RAW_CANDIDATES)
        canon_idx = _find_column_index(fieldnames, _OVERRIDE_CANONICAL_CANDIDATES)

        if raw_idx is None or canon_idx is None:
            log.warning(
                "%s exists but doesn't have raw_name/canonical_name "
                "columns (found: %s) -- ignoring it.",
                path, fieldnames,
            )
            return {}

        raw_col = fieldnames[raw_idx]
        canon_col = fieldnames[canon_idx]
        for row in reader:
            raw = (row.get(raw_col) or "").strip()
            canon = (row.get(canon_col) or "").strip()
            if raw and canon:
                overrides[raw] = canon

    log.info("Loaded %d strain-name override(s) from %s", len(overrides), path)
    return overrides


def normalize_strain_name(raw_name: str, overrides: dict[str, str]) -> str:
    """Apply a manual override if one exists for this exact raw name;
    otherwise pass the name through unchanged."""
    return overrides.get(raw_name, raw_name)
