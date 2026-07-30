"""
extract.py
==========
Flatten a single nested BacDive record into one or more
``(strain x culture medium)`` rows for the modeling table.

The natural datapoint in BacDive is a culture-medium entry with a binary
``growth`` flag. Temperature, pH and oxygen tolerance are recorded in separate
blocks keyed by publication ``@ref`` (not by medium), so we summarise them at
the *strain* level and attach the same value to every medium row for that
strain. This is documented as a caveat in ``bacdive/README.md``.
"""

from __future__ import annotations

import logging
import re
import statistics
from typing import Any, Optional

from src.bacdive.media import media_features

log = logging.getLogger("bacdive.extract")

# growth-string -> binary label. Anything else -> dropped.
_GROWTH_MAP = {
    "yes": 1,
    "positive": 1,
    "no": 0,
    "negative": 0,
}

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def as_list(value: Any) -> list:
    """Normalise a BacDive field that may be a dict, list, or missing.

    BacDive returns a bare dict when a block has a single entry and a list when
    it has several. This always yields a list of dicts.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def _first(value: Any) -> dict:
    """First entry of a possibly dict/list/missing block, or ``{}``."""
    items = as_list(value)
    return items[0] if items else {}


def _parse_number(text: Any) -> Optional[float]:
    """Parse a numeric value from a string that may be a range ("10-41").

    Returns the midpoint for ranges, the single value otherwise, or None.
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    nums = [float(m) for m in _NUM_RE.findall(str(text))]
    if not nums:
        return None
    if len(nums) == 1:
        return nums[0]
    # range or multiple values -> midpoint of min/max
    return (min(nums) + max(nums)) / 2.0


def _positive_median(entries: list[dict], value_key: str, flag_key: str) -> str:
    """Median of numeric ``value_key`` across entries flagged as growth.

    ``flag_key`` is the per-entry key holding "positive"/"negative"
    (``growth`` for temperature, ``ability`` for pH). Falls back to all
    entries when none are explicitly positive. Returns "" if nothing numeric.
    """
    vals: list[float] = []
    fallback: list[float] = []
    for e in entries:
        num = _parse_number(e.get(value_key))
        if num is None:
            continue
        fallback.append(num)
        if str(e.get(flag_key, "")).strip().lower() == "positive":
            vals.append(num)
    chosen = vals or fallback
    if not chosen:
        return ""
    return f"{statistics.median(chosen):g}"


# --------------------------------------------------------------------------
# strain-level features
# --------------------------------------------------------------------------

def _taxonomy(record: dict) -> dict:
    tax = record.get("Name and taxonomic classification", {})
    if isinstance(tax, list):
        tax = tax[0] if tax else {}
    if not isinstance(tax, dict):
        tax = {}
    name = (
        tax.get("full scientific name")
        or tax.get("species")
        or ""
    )
    # strip the HTML italics tags BacDive sometimes embeds in scientific names
    name = re.sub(r"</?I>", "", str(name)).strip()
    return {
        "strain_name": name or _species_from_general(record),
        "genus": tax.get("genus", ""),
        "family": tax.get("family", ""),
        "order": tax.get("order", ""),
        "phylum": tax.get("phylum", ""),
    }


def _species_from_general(record: dict) -> str:
    general = record.get("General", {})
    if isinstance(general, dict):
        return str(general.get("species", "")).strip()
    return ""


def _morphology(record: dict) -> dict:
    cell = _first(record.get("Morphology", {}).get("cell morphology")
                  if isinstance(record.get("Morphology"), dict) else None)
    return {
        "gram_stain": cell.get("gram stain", ""),
        "cell_shape": cell.get("cell shape", ""),
        "motility": cell.get("motility", ""),
    }


def _physiology(record: dict) -> dict:
    phys = record.get("Physiology and metabolism", {})
    if not isinstance(phys, dict):
        phys = {}
    oxygen = _first(phys.get("oxygen tolerance")).get("oxygen tolerance", "")
    spore = _first(phys.get("spore formation")).get("spore formation", "")
    nutrition = _first(phys.get("nutrition type")).get("type", "") or \
        _first(phys.get("nutrition type")).get("nutrition type", "")
    return {
        "oxygen_tolerance": oxygen,
        "spore_formation": spore,
        "nutrition_type": nutrition,
    }


def _sequence(record: dict) -> dict:
    seq = record.get("Sequence information", {})
    if not isinstance(seq, dict):
        seq = {}

    # best genome assembly by BacDive score
    genomes = as_list(seq.get("Genome sequences"))
    genome_accession = ""
    if genomes:
        best = max(
            genomes,
            key=lambda g: _parse_number(g.get("score")) or -1.0,
        )
        genome_accession = best.get("INSDC accession", "") or ""

    n16s = ""
    seqs_16s = as_list(seq.get("16S sequences"))
    if seqs_16s:
        n16s = seqs_16s[0].get("accession", "") or ""

    return {
        "genome_accession": genome_accession,
        "n16s_accession": n16s,
        "gc_content": _gc_content(record, seq),
    }


def _gc_content(record: dict, seq: dict) -> str:
    """GC content can live in a couple of places; take the first numeric hit."""
    for entry in as_list(seq.get("GC content")):
        num = _parse_number(entry.get("GC-content") or entry.get("GC content"))
        if num is not None:
            return f"{num:g}"
    phys = record.get("Physiology and metabolism", {})
    if isinstance(phys, dict):
        for entry in as_list(phys.get("GC content")):
            num = _parse_number(entry.get("GC-content") or entry.get("GC content"))
            if num is not None:
                return f"{num:g}"
    return ""


def _conditions(record: dict) -> dict:
    cg = record.get("Culture and growth conditions", {})
    if not isinstance(cg, dict):
        cg = {}
    temp = _positive_median(as_list(cg.get("culture temp")), "temperature", "growth")
    ph = _positive_median(as_list(cg.get("culture pH")), "pH", "ability")
    return {"temperature": temp, "ph": ph}


def strain_features(record: dict) -> dict:
    """Assemble all strain-level (per-record) feature columns."""
    features: dict[str, Any] = {}
    features.update(_taxonomy(record))
    features.update(_morphology(record))
    features.update(_physiology(record))
    features.update(_sequence(record))
    features.update(_conditions(record))
    features["bacdive_id"] = _bacdive_id(record)
    return features


def _bacdive_id(record: dict) -> str:
    general = record.get("General", {})
    if isinstance(general, dict):
        bid = general.get("BacDive-ID")
        return "" if bid is None else str(bid)
    return ""


# --------------------------------------------------------------------------
# row assembly
# --------------------------------------------------------------------------

def record_to_rows(record: dict) -> list[dict]:
    """Flatten one BacDive record into (strain x medium) rows.

    Only culture-medium entries with a mappable growth value are kept. Returns
    an empty list when the record has no usable culture-medium data.
    """
    cg = record.get("Culture and growth conditions", {})
    if not isinstance(cg, dict):
        return []
    media = as_list(cg.get("culture medium"))
    if not media:
        return []

    base = strain_features(record)
    rows: list[dict] = []

    for medium in media:
        growth_raw = str(medium.get("growth", "")).strip().lower()
        label = _GROWTH_MAP.get(growth_raw)
        if label is None:
            continue  # e.g. "no data" / blank -> not a usable datapoint

        name = medium.get("name", "") or ""
        feats = media_features(
            name=name,
            link=medium.get("link"),
            composition=medium.get("composition"),
        )

        row = dict(base)
        row["media_name"] = name
        row["growth_class"] = label
        row["refs"] = str(medium.get("@ref", ""))
        row.update(feats)
        rows.append(row)

    return rows
