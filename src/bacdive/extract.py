"""
extract.py
==========
Flatten a single nested BacDive record into modeling rows, for one of two
targets (see ``record_to_rows``):

* ``medium_growth`` -- one row per ``(isolate, culture medium, temperature)``
  with a binary ``growth`` flag. Dedup identity is
  ``(bacdive_id, media_name, temperature, oxygen_tolerance)``.
* ``metabolite_utilization`` -- one row per ``(strain, metabolite)``, restricted
  to entries where ``kind of utilization tested == "growth"``, with a real
  ``+``/``-`` label and the source publication's ``@ref`` id + full citation.

Temperature, pH and oxygen tolerance are recorded in separate blocks keyed by
publication ``@ref`` (not by medium/metabolite). pH and oxygen are summarised
once at the isolate level. Temperature does the same *unless* more than one
distinct parsed temperature is reported, in which case each culture-medium
entry is matched to the temperature entry(ies) sharing its ``@ref`` -- see
``_group_by_ref`` and ``record_to_medium_rows``.

By default a reported range (e.g. ``"30-37"``) is one observation (its
midpoint). Set ``temperature_range_expansion=True`` to emit three points
instead (min, max, mean).
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

# metabolite "utilization activity" -> binary label
_UTILIZATION_MAP = {
    "+": 1,
    "-": 0,
}

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
# A "min-max" range, e.g. "10-41" or "-2 - 4". Matched before the generic
# _NUM_RE scan so the separating hyphen isn't mistaken for the second
# number's sign (a bare findall on "10-41" yields [10, -41], not [10, 41]).
_RANGE_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)")

# BacDive's "Genome sequences" entries carry an INSDC accession but no direct
# URL, unlike culture-medium entries. Build the NCBI Datasets genome page.
_NCBI_GENOME_URL = "https://www.ncbi.nlm.nih.gov/datasets/genome/{accession}/"


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
    s = str(text)
    m = _RANGE_RE.search(s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (min(lo, hi) + max(lo, hi)) / 2.0
    nums = [float(n) for n in _NUM_RE.findall(s)]
    if not nums:
        return None
    if len(nums) == 1:
        return nums[0]
    # multiple standalone values -> midpoint of min/max
    return (min(nums) + max(nums)) / 2.0


def _parse_temp_points(text: Any) -> list[float]:
    """Parse a temperature field into one or more numeric points.

    A single value yields one point. A "min-max" range (e.g. "10-41")
    yields three points (min, max, mean). Used for the isolate-level
    median and, when ``temperature_range_expansion`` is on, for emitting
    one modeling row per point.
    """
    if text is None:
        return []
    if isinstance(text, (int, float)):
        return [float(text)]
    s = str(text)
    m = _RANGE_RE.search(s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        lo, hi = min(lo, hi), max(lo, hi)
        return [lo, hi, (lo + hi) / 2.0]
    nums = [float(n) for n in _NUM_RE.findall(s)]
    if not nums:
        return []
    if len(nums) == 1:
        return [nums[0]]
    lo, hi = min(nums), max(nums)
    return [lo, hi, (lo + hi) / 2.0]


def _positive_median(entries: list[dict], value_key: str, flag_key: str) -> str:
    """Median of numeric ``value_key`` across entries flagged as growth.

    ``flag_key`` is the per-entry key holding "positive"/"negative"
    (``ability`` for pH). Falls back to all entries when none are explicitly
    positive. Returns "" if nothing numeric.
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


def _median_all(entries: list[dict], value_key: str) -> str:
    """Median of numeric ``value_key`` across *all* entries, regardless of
    any growth/ability flag. Ranges (e.g. "10-41") are expanded into three
    points -- min, max, mean -- by ``_parse_temp_points`` before the median
    is taken. Returns "" if nothing numeric.
    """
    vals = [num for e in entries for num in _parse_temp_points(e.get(value_key))]
    if not vals:
        return ""
    return f"{statistics.median(vals):g}"


def _group_by_ref(
    entries: list[dict],
    value_key: str,
    flag_key: Optional[str] = None,
    *,
    expand_ranges: bool = False,
) -> dict[str, list[tuple[float, Optional[int]]]]:
    """Group numeric ``value_key`` values by the entry's publication ``@ref``.

    Returns ``{ref: [(value, growth_label), ...]}`` (distinct pairs, in
    first-seen order), dropping entries with no parseable number.

    Each source entry contributes one midpoint by default. If
    ``expand_ranges`` is true, a range such as ``"10-41"`` becomes three
    points (min, max, mean), each carrying that entry's growth label.

    ``growth_label`` is the entry's own ``flag_key`` mapped through
    ``_GROWTH_MAP`` (e.g. "negative" -> 0), or ``None`` if ``flag_key`` is
    omitted or unrecognised -- a *negative*-flagged temperature reading means
    growth was NOT observed at that temperature, which should override the
    medium's own growth label when the two are matched by shared ``@ref``.
    """
    grouped: dict[str, list[tuple[float, Optional[int]]]] = {}
    for e in entries:
        if expand_ranges:
            points = _parse_temp_points(e.get(value_key))
        else:
            num = _parse_number(e.get(value_key))
            points = [] if num is None else [num]
        if not points:
            continue
        ref = str(e.get("@ref", "")).strip()
        label = None
        if flag_key is not None:
            label = _GROWTH_MAP.get(str(e.get(flag_key, "")).strip().lower())
        vals = grouped.setdefault(ref, [])
        for num in points:
            pair = (num, label)
            if pair not in vals:
                vals.append(pair)
    return grouped


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
        "strain_name_raw": name or _species_from_general(record),
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
        "genome_link": _NCBI_GENOME_URL.format(accession=genome_accession) if genome_accession else "",
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
    temp = _median_all(as_list(cg.get("culture temp")), "temperature")
    ph = _positive_median(as_list(cg.get("culture pH")), "pH", "ability")
    return {"temperature": temp, "ph": ph}


def strain_features(record: dict) -> dict:
    """Assemble all strain-level (per-record) feature columns.

    ``strain_name_raw`` is BacDive's scientific name (species-level).
    ``strain_name`` is the isolate-unique key ``{bacdive_id}_{strain_name_raw}``
    used by the modeling pipeline for grouping / joins.
    """
    features: dict[str, Any] = {}
    features.update(_taxonomy(record))
    features.update(_morphology(record))
    features.update(_physiology(record))
    features.update(_sequence(record))
    features.update(_conditions(record))
    features["bacdive_id"] = _bacdive_id(record)
    raw = str(features.get("strain_name_raw", "") or "").strip()
    bid = str(features.get("bacdive_id", "") or "").strip()
    if bid and raw:
        features["strain_name"] = f"{bid}_{raw}"
    else:
        features["strain_name"] = bid or raw
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

def record_to_medium_rows(
    record: dict, *, temperature_range_expansion: bool = False,
) -> list[dict]:
    """Flatten one BacDive record into (isolate x medium x temperature) rows.

    Only culture-medium entries with a mappable growth value are kept. Returns
    an empty list when the record has no usable culture-medium data.

    Temperature: if the record reports more than one *distinct* parsed
    temperature, a single isolate-wide value is no longer precise enough, so
    each medium entry is matched against ``culture temp`` entries sharing its
    ``@ref``. A medium whose ref has no matching temperature falls back to
    ``base["temperature"]`` (the isolate-wide median) and keeps its own
    ``growth`` label; a medium whose ref matches more than one temperature is
    expanded into one row per (medium, temperature) pair. A matched
    temperature entry's own ``growth`` flag overrides the medium's growth
    label for that row.

    ``temperature_range_expansion``: if false (default), a range such as
    ``"30-37"`` is one midpoint. If true, it emits min, max, and mean as
    three rows (and those extra points can themselves trigger @ref matching).
    """
    cg = record.get("Culture and growth conditions", {})
    if not isinstance(cg, dict):
        return []
    media = as_list(cg.get("culture medium"))
    if not media:
        return []

    base = strain_features(record)

    temps_by_ref = _group_by_ref(
        as_list(cg.get("culture temp")),
        "temperature",
        flag_key="growth",
        expand_ranges=temperature_range_expansion,
    )
    distinct_temps = {v for vals in temps_by_ref.values() for v, _ in vals}
    match_temp_by_ref = len(distinct_temps) > 1

    rows: list[dict] = []

    for medium in media:
        growth_raw = str(medium.get("growth", "")).strip().lower()
        label = _GROWTH_MAP.get(growth_raw)
        if label is None:
            continue  # e.g. "no data" / blank -> not a usable datapoint

        name = medium.get("name", "") or ""
        ref = str(medium.get("@ref", ""))
        feats = media_features(
            name=name,
            link=medium.get("link"),
            composition=medium.get("composition"),
        )

        matched_temps = temps_by_ref.get(ref.strip()) if match_temp_by_ref else None
        # (None, None) -> keep base["temperature"] and the medium's own label
        temp_overrides = matched_temps or [(None, None)]

        for temp_val, temp_label in temp_overrides:
            row = dict(base)
            row["media_name"] = name
            row["medium_link"] = medium.get("link") or ""
            row["growth_class"] = label if temp_label is None else temp_label
            row["refs"] = ref
            row.update(feats)
            if temp_val is not None:
                row["temperature"] = f"{temp_val:g}"
            rows.append(row)

    return rows


def _reference_lookup(record: dict) -> dict[str, dict]:
    """Map ``@ref`` id (as a string) -> that publication's ``Reference`` entry."""
    return {
        str(ref["@id"]): ref
        for ref in as_list(record.get("Reference"))
        if ref.get("@id") is not None
    }


def _format_reference(ref: dict) -> str:
    """Render a ``Reference`` entry as a single human-readable citation string."""
    parts = [
        str(ref.get("authors", "") or "").strip(),
        str(ref.get("title", "") or "").strip(),
        str(ref.get("journal", "") or "").strip(),
        str(ref.get("doi/url", "") or "").strip(),
    ]
    return ". ".join(p for p in parts if p)


def record_to_metabolite_rows(record: dict) -> list[dict]:
    """Flatten one BacDive record into (strain x metabolite) rows.

    Reads ``Physiology and metabolism -> metabolite utilization``, keeping
    only entries whose ``kind of utilization tested`` is ``"growth"`` (as
    opposed to e.g. "carbon source", "fermentation", "builds acid from") with
    a mappable ``utilization activity`` (``+``/``-``). Each row also carries
    the publication ``@ref`` id and the full citation text resolved from the
    record's ``Reference`` block. Returns an empty list when the record has
    no usable growth-utilization data.
    """
    phys = record.get("Physiology and metabolism", {})
    if not isinstance(phys, dict):
        return []
    entries = as_list(phys.get("metabolite utilization"))
    if not entries:
        return []

    base = strain_features(record)
    references = _reference_lookup(record)
    rows: list[dict] = []

    for entry in entries:
        kind = str(entry.get("kind of utilization tested", "")).strip().lower()
        if kind != "growth":
            continue

        activity = str(entry.get("utilization activity", "")).strip()
        label = _UTILIZATION_MAP.get(activity)
        if label is None:
            continue  # e.g. "+/-" or blank -> not a usable datapoint

        substrate = entry.get("metabolite", "") or ""
        if not substrate:
            continue

        chebi = entry.get("Chebi-ID")
        ref_id = str(entry.get("@ref", ""))
        row = dict(base)
        row["substrate_name"] = substrate
        row["chebi_id"] = "" if chebi is None else str(chebi)
        row["kind_of_utilization"] = entry.get("kind of utilization tested", "") or ""
        row["growth_class"] = label
        row["refs"] = ref_id
        row["reference"] = _format_reference(references.get(ref_id, {}))
        rows.append(row)

    return rows


_EXTRACTORS = {
    "medium_growth": record_to_medium_rows,
    "metabolite_utilization": record_to_metabolite_rows,
}


def record_to_rows(
    record: dict,
    target: str = "medium_growth",
    *,
    temperature_range_expansion: bool = False,
) -> list[dict]:
    """Flatten one BacDive record into rows for the requested ``target``."""
    try:
        extractor = _EXTRACTORS[target]
    except KeyError:
        raise ValueError(
            f"unknown target {target!r}; choose one of {sorted(_EXTRACTORS)}"
        ) from None
    if target == "medium_growth":
        return extractor(
            record, temperature_range_expansion=temperature_range_expansion,
        )
    return extractor(record)
