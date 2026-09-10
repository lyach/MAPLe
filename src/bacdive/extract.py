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

from src.bacdive.media import (
    canonical_media_name,
    get_media_alias_map,
    media_canon_status,
    media_features,
)

log = logging.getLogger("bacdive.extract")

# growth-string -> binary label. Anything else -> dropped.
_GROWTH_MAP = {
    "yes": 1,
    "positive": 1,
    "no": 0,
    "negative": 0,
}

# Provenance of ``growth_class`` on a medium_growth row.
_GROWTH_SOURCE_MEDIUM = "culture medium"
_GROWTH_SOURCE_TEMP = "culture temp"
_TEMP_JOIN_MATCHED = "matched"
_TEMP_JOIN_UNMATCHED = "unmatched"
_TEMP_JOIN_SKIPPED = "skipped_single_temp"

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

# BacDive's "Genome sequences" entries carry an accession but no direct URL,
# unlike culture-medium entries. Build the browse page for whichever registry
# the accession came from.
_NCBI_GENOME_URL = "https://www.ncbi.nlm.nih.gov/datasets/genome/{accession}/"
_BVBRC_GENOME_URL = "https://www.bv-brc.org/view/Genome/{accession}"
_IMG_GENOME_URL = (
    "https://img.jgi.doe.gov/cgi-bin/m/main.cgi"
    "?section=TaxonDetail&page=taxonDetail&taxon_oid={accession}"
)

# Accession registries on a "Genome sequences" entry, best first. BacDive fills
# these inconsistently: an entry may carry any subset, so the same entry can
# yield an INSDC accession for one strain and only an IMG id for another.
_GENOME_ACCESSION_FIELDS = (
    ("insdc", "INSDC accession", _NCBI_GENOME_URL),
    ("bvbrc", "BV-BRC accession", _BVBRC_GENOME_URL),
    ("img", "IMG accession", _IMG_GENOME_URL),
)


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


# Culture-collection accession, e.g. "DSM 1313", "CIP 103.894", "ATCC 700813".
_COLLECTION_RE = re.compile(
    r"\b(DSM|DSMZ|ATCC|CIP|LMG|NCTC|JCM|KCTC|NBRC|CCUG|CECT|IAM|VKM|BCRC"
    r"|NRRL|CGMCC|NCIMB|KACC)\b[ ._-]?(\d[\d.]*)",
    re.I,
)


def _strain_identity(record: dict) -> dict:
    """Strain-level identifiers usable to resolve a genome outside BacDive.

    None of these are derivable from the species name, and every one of them
    is a handle the genome-gap audit found we were discarding:

    * ``strain_designation`` / ``is_type_strain`` -- pins the isolate, and type
      strains match NCBI's ``relation_to_type_material``.
    * ``dsm_number`` / ``culture_collection_ids`` -- the deposit accessions to
      match against NCBI's ``assembly_summary`` files.
    * ``ncbi_taxid`` / ``ncbi_taxid_level`` -- the level matters: a
      species-level taxid identifies a *set* of assemblies, not this strain's.
    """
    tax = record.get("Name and taxonomic classification", {})
    if isinstance(tax, list):
        tax = tax[0] if tax else {}
    if not isinstance(tax, dict):
        tax = {}
    general = record.get("General", {})
    if not isinstance(general, dict):
        general = {}

    taxid, taxid_level = "", ""
    entry = general.get("NCBI tax id")
    entries = as_list(entry)
    if entries:
        # prefer a strain-level assignment over the species-level default
        best = next(
            (e for e in entries
             if str(e.get("Matching level", "")).strip().lower() == "strain"),
            entries[0],
        )
        taxid = str(best.get("NCBI tax id", "") or "").strip()
        taxid_level = str(best.get("Matching level", "") or "").strip()

    dsm = str(general.get("DSM-Number", "") or "").strip()

    # Collection ids are scattered across strain history, external links and
    # free text, so scan the record rather than guessing at a field name.
    collection_ids = sorted({
        f"{m.group(1).upper()} {m.group(2)}"
        for m in _COLLECTION_RE.finditer(_json_text(record))
    })
    if dsm and not any(c.startswith("DSM ") for c in collection_ids):
        collection_ids.insert(0, f"DSM {dsm}")

    type_strain = str(tax.get("type strain", "") or "").strip().lower()

    return {
        "strain_designation": str(tax.get("strain designation", "") or "").strip(),
        "is_type_strain": {"yes": "1", "no": "0"}.get(type_strain, ""),
        "dsm_number": dsm,
        "culture_collection_ids": ";".join(collection_ids),
        "ncbi_taxid": taxid,
        "ncbi_taxid_level": taxid_level,
    }


def _json_text(record: dict) -> str:
    """Whole record as one string, for scanning free-text fields."""
    import json

    try:
        return json.dumps(record, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(record)


# A CIP catalogue medium, recorded either as the cross-reference entry
# ("CIP Medium 6", carries the Pasteur link) or as the experimental entry
# ("MEDIUM 6 - Columbia agar with 10 % horse blood", carries the recipe).
_CIP_MEDIUM_RE = re.compile(r"^\s*(CIP\s+)?MEDIUM\s*(\d+)", re.I)
_CIP_LINK_RE = re.compile(r"crbip=(\d+)", re.I)


def _cip_medium_number(name: str) -> str:
    """CIP catalogue number for a medium name, or "" if it isn't a CIP medium.

    Both spellings map to the same number: BacDive splits one CIP medium into
    two entries, the cross-reference holding the catalogue URL and the
    experimental entry holding the composition. A number identifies the recipe
    globally -- across 66 numbers seen in the cache, none maps to two different
    recipes -- so it is safe as the medium's identity.
    """
    match = _CIP_MEDIUM_RE.match(str(name or ""))
    return match.group(2) if match else ""


def _is_cip_crossref(name: str) -> bool:
    """True for the "CIP Medium 6" spelling (the catalogue cross-reference)."""
    match = _CIP_MEDIUM_RE.match(str(name or ""))
    return bool(match and match.group(1))


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

    genome_accession, genome_source, genome_link = _best_genome_accession(
        as_list(seq.get("Genome sequences"))
    )

    n16s = ""
    seqs_16s = as_list(seq.get("16S sequences"))
    if seqs_16s:
        n16s = seqs_16s[0].get("accession", "") or ""

    return {
        "genome_accession": genome_accession,
        "genome_accession_source": genome_source,
        "genome_link": genome_link,
        "n16s_accession": n16s,
        "gc_content": _gc_content(record, seq),
    }


def _best_genome_accession(genomes: list[dict]) -> tuple[str, str, str]:
    """Best accession across a strain's genome entries, INSDC first.

    Registries are tried in ``_GENOME_ACCESSION_FIELDS`` order and score only
    breaks ties *within* a registry: the highest-scoring INSDC accession wins
    if any entry has one, and only when no entry anywhere carries an INSDC
    accession do BV-BRC and then IMG get a turn.

    Ranking strictly by score instead would be misleading. BacDive's
    top-scoring entry frequently carries only a BV-BRC/IMG id while a
    lower-scoring sibling holds a perfectly good ``GCA_`` accession -- true for
    430 of the 533 strains this function was blanking. Preferring score there
    yields an id that ``scripts/download_bacdive_genomes.py`` cannot fetch,
    and neither alternative is a usable substitute: IMG sits behind a bot
    challenge, and BV-BRC ids resolve to an NCBI assembly for only a small
    minority of the strains that have nothing else.

    Previously this took ``max(...)`` by score and then read ``INSDC
    accession`` off that one entry, returning "" whenever that entry had none.

    Entries with no parseable score sort last, rather than being scored -1.0 --
    a value a legitimate ``score: 0`` also collapsed to, demoting it below
    entries carrying no score at all.

    Returns ``(accession, source, url)``; all "" when nothing is available.
    """
    def rank(entry: dict) -> tuple[int, float]:
        num = _parse_number(entry.get("score"))
        # (0, -score) sorts ahead of (1, 0.0) -> scored entries beat unscored
        return (0, -num) if num is not None else (1, 0.0)

    ranked = sorted(genomes, key=rank)
    for source, field, url_template in _GENOME_ACCESSION_FIELDS:
        for entry in ranked:
            accession = str(entry.get(field, "") or "").strip()
            if accession:
                return accession, source, url_template.format(accession=accession)
    return "", "", ""


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
    features.update(_strain_identity(record))
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
    record: dict, *, temperature_range_expansion: bool = True,
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
    label for that row. Provenance columns (always emitted):

    * ``medium_growth_class`` -- the culture-medium entry's own mapped flag.
    * ``temp_growth_class`` -- the matched culture-temp flag, or blank.
    * ``growth_source`` -- which block supplied the *emitted* ``growth_class``
      (``"culture temp"`` whenever a temp flag was applied, including when it
      *agrees* with the medium).
    * ``growth_override`` -- 1 iff the temp flag was applied *and* differs
      from the medium. This is the unbiased count of labels that culture temp
      actually changed.
    * ``temp_join`` -- ``matched`` / ``unmatched`` / ``skipped_single_temp``.
      Matching is skipped when the record has only one distinct temperature,
      so those rows stay ``growth_source="culture medium"`` even if a temp
      entry shares the medium's ``@ref``.

    ``temperature_range_expansion``: If true (default), it emits min, max, and mean as
    three rows (and those extra points can themselves trigger @ref matching).
    if false, a range such as ``"30-37"`` is one midpoint.

    CIP media: BacDive records one CIP medium as two entries -- a
    cross-reference ("CIP Medium 6") holding the catalogue URL, and an
    experimental entry ("MEDIUM 6 - ...") holding the recipe and the
    temperature. Both are folded onto the canonical name ``CIP Medium <n>``
    and share the recipe, so they collapse to one row per
    ``(medium, temperature)``. A cross-reference that contributes no
    independent measurement -- no temperature of its own that the experimental
    entry doesn't already report -- is dropped rather than emitted with the
    isolate-wide median, which would invent a datapoint. A CIP medium with no
    experimental entry at all is kept (it is the only record of that medium)
    but with an *empty* temperature unless its own ref supplies one.
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

    cip = _resolve_cip_media(media, temps_by_ref, match_temp_by_ref)

    rows: list[dict] = []

    for medium in media:
        growth_raw = str(medium.get("growth", "")).strip().lower()
        label = _GROWTH_MAP.get(growth_raw)
        if label is None:
            continue  # e.g. "no data" / blank -> not a usable datapoint

        name = medium.get("name", "") or ""
        ref = str(medium.get("@ref", ""))
        link = medium.get("link")
        composition = medium.get("composition")

        cip_number = _cip_medium_number(name)
        cip_info = cip.get(cip_number) if cip_number else None
        if cip_info is not None:
            if id(medium) in cip_info["drop"]:
                continue  # cross-reference adding no independent measurement
            # fold both spellings onto one identity, sharing recipe and URL
            name = cip_info["canonical_name"]
            composition = composition or cip_info["composition"]
            link = link or cip_info["link"]

        feats = media_features(
            name=name,
            link=link,
            composition=composition,
        )
        media_alias_overrides = get_media_alias_map()
        media_canon = canonical_media_name(name, media_alias_overrides)
        media_canon_state = media_canon_status(name, media_alias_overrides)

        ref_key = ref.strip()
        ref_temps = temps_by_ref.get(ref_key)
        if match_temp_by_ref:
            temp_overrides = ref_temps or [(None, None)]
        else:
            temp_overrides = [(None, None)]

        skipped_temp_label: Optional[int] = None
        if not match_temp_by_ref and ref_temps:
            labs = {lab for _, lab in ref_temps if lab is not None}
            if len(labs) == 1:
                skipped_temp_label = next(iter(labs))

        for temp_val, temp_label in temp_overrides:
            used_temp_label = temp_label if match_temp_by_ref else None
            emitted = label if used_temp_label is None else used_temp_label
            if match_temp_by_ref:
                temp_join = (
                    _TEMP_JOIN_MATCHED if temp_val is not None
                    else _TEMP_JOIN_UNMATCHED
                )
                documented_temp = temp_label
            else:
                temp_join = _TEMP_JOIN_SKIPPED
                documented_temp = skipped_temp_label

            row = dict(base)
            row["media_name"] = name
            row["media_name_full"] = medium.get("name", "") or ""
            row["media_canon"] = media_canon
            row["media_canon_status"] = media_canon_state
            row["medium_link"] = link or ""
            row["growth_class"] = emitted
            row["medium_growth_class"] = label
            row["growth_source"] = (
                _GROWTH_SOURCE_MEDIUM if used_temp_label is None
                else _GROWTH_SOURCE_TEMP
            )
            row["growth_override"] = int(
                used_temp_label is not None and used_temp_label != label
            )
            row["temp_join"] = temp_join
            row["temp_growth_class"] = (
                "" if documented_temp is None else str(int(documented_temp))
            )
            row["refs"] = ref
            row.update(feats)
            # always set, so non-CIP rows carry "" like every other blank
            # column rather than a None that only appears for some rows
            row["cip_medium_number"] = cip_number if cip_info else ""
            row["cip_medium_link"] = cip_info["link"] if cip_info else ""
            if temp_val is not None:
                row["temperature"] = f"{temp_val:g}"
            elif cip_info is not None and not cip_info["temperature_measured"]:
                # This CIP medium has no temperature reported against any of
                # its refs. The isolate-wide median was never measured for it,
                # so leave it empty rather than fabricate an observation.
                row["temperature"] = ""
            rows.append(row)

    return rows


def _resolve_cip_media(
    media: list[dict],
    temps_by_ref: dict[str, list],
    match_temp_by_ref: bool,
) -> dict[str, dict]:
    """Group CIP medium entries by catalogue number and decide what to keep.

    For each number this returns the shared recipe and catalogue URL (pooled
    across the cross-reference and experimental spellings), whether *any* of
    its entries has a measured temperature, and the ``id()``s of entries to
    drop.

    An entry is dropped when it is a cross-reference ("CIP Medium 6") that
    contributes no independent measurement: either it has no temperature of
    its own, or every temperature it reports is already reported by a sibling
    entry for the same medium. Keeping it would duplicate a row, or -- worse,
    when it has no temperature -- mint one at the isolate-wide median, a value
    never measured against that reference.

    Entries that do report their own distinct temperatures are kept: the CIP
    catalogue tests growth across a range (e.g. 10, 33.5, 55 C) and those
    no-growth points are real, and are the dataset's main source of negatives.
    """
    by_number: dict[str, dict] = {}

    for medium in media:
        number = _cip_medium_number(medium.get("name", "") or "")
        if not number:
            continue
        info = by_number.setdefault(number, {
            "canonical_name": f"CIP Medium {number}",
            "composition": "",
            "link": "",
            "temperature_measured": False,
            "drop": set(),
            "entries": [],
        })
        info["entries"].append(medium)
        if not info["composition"] and (medium.get("composition") or "").strip():
            info["composition"] = medium["composition"]
        link = str(medium.get("link", "") or "").strip()
        if not info["link"] and _CIP_LINK_RE.search(link):
            info["link"] = link

    for number, info in by_number.items():
        # temperatures each entry reports against its own @ref
        own: dict[int, set] = {}
        for medium in info["entries"]:
            ref = str(medium.get("@ref", "")).strip()
            values = temps_by_ref.get(ref) if match_temp_by_ref else None
            own[id(medium)] = {v for v, _ in (values or [])}
        # When every reported temperature is the same value, the isolate-wide
        # figure *is* that measured value -- there is nothing to fabricate, so
        # it applies to this medium like any other. Only when temperatures
        # differ between refs can the isolate-wide median be a number no one
        # measured for this medium, and only then is blanking warranted.
        info["temperature_measured"] = not match_temp_by_ref or any(own.values())

        if len(info["entries"]) < 2:
            continue  # sole record of this medium -> always kept

        # temperatures contributed by the experimental ("MEDIUM n") entries
        experimental = {
            v
            for medium in info["entries"]
            if not _is_cip_crossref(medium.get("name", "") or "")
            for v in own[id(medium)]
        }
        for medium in info["entries"]:
            if not _is_cip_crossref(medium.get("name", "") or ""):
                continue
            if not own[id(medium)] - experimental:
                info["drop"].add(id(medium))

    return by_number


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
