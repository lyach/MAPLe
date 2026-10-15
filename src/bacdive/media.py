"""Turn a BacDive ``culture medium`` entry into modeling features.

BacDive's open vocabulary of medium names does not transfer under
leave-one-medium-out CV, so the free-text ``composition`` is parsed into
presence flags (mirroring ``local/media_features.csv``) plus an ingredient
count, and a stable MediaDive id is pulled from the ``link``. Medium names are
also folded onto a canonical grouping key (``media_canon``).
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

from loguru import logger


# ---------------------------------------------------------------------------
# Composition markers
# ---------------------------------------------------------------------------

# Lowercased substring markers: each feature column maps to the tokens that,
# if found in the composition text, set it.
_MARKERS: dict[str, tuple[str, ...]] = {
    "has_yeast_extract": ("yeast extract",),
    "has_peptone": ("peptone", "tryptone", "trypto casein", "casein"),
    "has_agar": ("agar",),
    "has_blood_components": ("blood", "hemin", "haemin", "serum", "haematin", "hematin"),
    "has_nacl": ("nacl", "sodium chloride"),
}

# Complex (undefined) ingredients: their presence means the medium contains
# at least one nutritionally undefined component (biological extract, hydrolysate, etc.).
_COMPLEX_MARKERS: tuple[str, ...] = (
    "yeast extract",
    "peptone",
    "tryptone",
    "trypto casein",
    "casein",
    "beef extract",
    "meat extract",
    "blood",
    "serum",
    "brain heart",
    "malt extract",
    "soy",
    "soja",
    "tryptic",
    "lysate",
    "digest",
    # branded/commercial complex media (named as a single recipe ingredient,
    # so they never contain a literal marker like "peptone" above)
    "nutrient broth",
    "nutrient agar",
    "columbia agar",
    "r2a agar",
    "r2a broth",
    "marine agar",
    "marine broth",
    "middlebrook",
    "man rogosa sharp",
    "todd hewitt",
    "mueller hinton",
    "bcye",
    "cye-agar",
    "legionella",
    "rcm medium",
    "dehydrated rcm",
    "fastidious anaerobe",
    "gam broth",
    "gifu anaerobic",
    "bsk-h",
    "leptospira medium",
    "emjh",
    "v-8 juice",
    "v8 juice",
    "oatmeal",
    "flour medium",
    "plain flour",
    "liver broth",
    "soil extract",
    "garden soil",
    "cow manure",
    "cow-manure",
    "wilkins chalgrene",
    "casitone",
    "casamino acids",
    "skim milk",
    "oadc",
    "fildes enrichment",
    "mycobactin",
    "albumin",
    "chocolate agar",
    "trypticase soy",
    "sabouraud",
    "macconkey",
    "brucella",
    "sea salt",
    "artificial seawater",
    "potato",
    "carrot",
    "mucin",
)

# Chemically defined (pure-compound) ingredients: their presence is evidence
# that at least some components are compositionally known.
_DEFINED_MARKERS: tuple[str, ...] = (
    # simple sugars
    "glucose",
    "sucrose",
    "fructose",
    "galactose",
    "maltose",
    "lactose",
    "ribose",
    "xylose",
    "mannose",
    # inorganic / mineral salts
    "nacl",
    "kcl",
    "k2hpo4",
    "kh2po4",
    "na2hpo4",
    "mgso4",
    "cacl2",
    "feso4",
    "mncl2",
    "znso4",
    "mgcl2",
    "nh4cl",
    "k2so4",
    "na2so4",
    # nitrogen sources
    "nh4",
    "ammonium",
    "nitrate",
    "urea",
    # vitamins
    "thiamine",
    "biotin",
    "riboflavin",
    "vitamin",
    "niacin",
    "pantothen",
    "folic acid",
    "cobalamin",
    # nucleobases / nucleosides
    "thymine",
    "thymidine",
    "adenine",
    "guanine",
    # organic acids
    "pyruvate",
    "citrate",
    "acetate",
    "fumarate",
    "succinate",
    # pure amino acids
    "cysteine",
    "glycine",
    "glutamate",
    "glutamine",
    "alanine",
    # generic mineral-salt terms: catches spelled-out salt names regardless
    # of hydrate form/spelling (e.g. "Magnesium sulphate heptahydrate" never
    # matches the "mgso4" formula abbreviation above)
    "chloride",
    "sulphate",
    "sulfate",
    "carbonate",
    "phosphate",
    "molybdate",
    "bicarbonate",
    # other pure compounds
    "dextrose",
    "tween",
)

_MEDIADIVE_RE = re.compile(r"mediadive\.dsmz\.de/medium/(\d+)", re.IGNORECASE)

# A real recipe clause names a mass/volume for at least one ingredient
# ("39 g", "1000 ml"). Short free-text notes attached to a standard medium's
# name ("supplemented with 2.5% NaCl (w/v)") describe a modifier, not a
# recipe, and should never be mistaken for one when backfilling.
_QUANTITY_UNIT_RE = re.compile(r"\d+(\.\d+)?\s*(g|mg|kg|ml|mL|l|L)\b")


def looks_like_recipe(composition: str | None) -> bool:
    """Return True if ``composition`` names at least one ingredient's mass/volume.

    Gates backfilling ``composition_raw`` so only recipe-shaped text is reused.
    """
    return bool(composition) and bool(_QUANTITY_UNIT_RE.search(str(composition)))


def parse_mediadive_id(link: str | None) -> str:
    """Extract the MediaDive medium id from a ``link`` URL, or "" if not a MediaDive link."""
    if not link:
        return ""
    match = _MEDIADIVE_RE.search(str(link))
    return match.group(1) if match else ""


# ---------------------------------------------------------------------------
# Canonical medium name (grouping key)
# ---------------------------------------------------------------------------

# Filler words
_GENERIC_MEDIA_TOKENS = frozenset({
    "broth", "medium", "media", "agar", "agars", "minimal", "modified",
    "plates", "plate", "ahar", "ahars", "ager",
    "slant", "slants", "base", "s",
    "and", "with",
    "extract",
    "test",
    "dox",
    "microaerobic", "microaerophilic", "aerobic", "aerobically",
    "anaerobically",
    "defibrinated",
})

# Supplier tokens
_VENDOR_TOKENS = frozenset({
    "oxoid", "difco", "bbl", "becton", "dickinson", "bd", "merck", "sigma",
    "himedia", "fluka", "biomerieux", "biolife", "acumedia", "condalab",
    "thermo", "remel", "hardy", "scharlau", "liofilchem", "conda",
})

# Preparation qualifiers that describe how a medium was made up
_SKIP_MEDIA_TOKENS = frozenset({
    "x", "modified", "modi", "half", "strength", "bacto", "diluted",
    "liquid", "solid", "pre", "reduced", "anaerobe", "anaerobic",
})

# The subset of stripped tokens that CAN change a formulation rather than just
# its spelling: "half strength marine broth" really is not marine broth, and
# "modified GAM" is a distinct recipe. 
RISKY_STRIPPED_TOKENS = frozenset({
    "minimal", "modified", "modi", "half", "strength", "diluted", "reduced",
    "base", "pre",
})

# Spelling variants BacDive uses for one identity.
_MEDIA_TOKEN_CANON = {
    "trypticase": "tryptic",
    "tryptone": "tryptic",
    "tryptose": "tryptic",
    "tripticase": "tryptic",
    "trypicase": "tryptic",
    "soya": "soy",
    "soybean": "soy",
    # MacConkey misspellings
    "macconcey": "macconkey",
    "mcconkey": "macconkey",
    "maconkey": "macconkey",
    "conkey": "macconkey",
    "reasoners": "r2a",
    "reasoner": "r2a",
    # typos BacDive carries
    "nutritient": "nutrient",
    "infuison": "infusion",
    "infusio": "infusion",
    "chalgrene": "chalgren",
    # "soybean-casein digest agar" is the formal name for tryptic soy
    # agar, so "Trypticase Casein Soy Agar" is TSA under another name
    "casein": "tryptic",
    # 2216 is the ZoBell formulation number for marine agar/broth
    "2216": "marine",
    # LB is Luria-Bertani and lysogeny broth -- the same medium
    "lysogeny": "luria bertani",
    "chrysoidine": "chrysoidin",
}

# Acronym -> expansion, applied so "TSA" and "Trypticase Soy Agar" converge.
_MEDIA_ACRONYMS = {
    "tsa": "tryptic soy",
    "tsb": "tryptic soy",
    "na": "nutrient",
    "nb": "nutrient",
    "lb": "luria bertani",
    "la": "luria bertani",
    "isp": "international streptomyces project",
    "mrs": "de man rogosa sharpe",
    "r2a": "r2a",
    "bhi": "brain heart infusion",
    "tcbs": "thiosulfate citrate bile salts sucrose",
    "ma": "marine",
    "mb": "marine",
    "yma": "yeast mannitol",
    "pca": "plate count",
    "pda": "potato dextrose",
    "mca": "macconkey",
    "gam": "gifu anaerobic",
    "thy": "todd hewitt",
    "pye": "peptone yeast extract",
    "nu": "nutrient",
    "gym": "glucose yeast malt",
    "ym": "yeast malt",
}

_SHORT_MEDIA_CODES = frozenset(_MEDIA_ACRONYMS)

# Multi-word spellings that tokenising would break apart, collapsed before
# tokenisation.
_PHRASE_CANON: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"ma?c[\s'’-]*conkey", re.IGNORECASE), "macconkey"),
    (re.compile(r"reasoner'?’?s?[\s-]*2\s*a", re.IGNORECASE), "r2a"),
    (re.compile(r"luria[\s-]*bertani", re.IGNORECASE), "luria bertani"),
    (re.compile(r"mueller[\s-]*hinton", re.IGNORECASE), "mueller hinton"),
    # TCBS is spelled out with and without "salts" and "sucrose"; fold every
    # spelling onto the acronym so they cannot form separate groups.
    (re.compile(r"thiosulfate[\s,-]*citrate[\s,-]*bile(?:[\s,-]*salts?)?"
                r"(?:[\s,-]*sucrose)?", re.IGNORECASE), "tcbs"),
    # A blood percentage is the standard supplement level (5% sheep blood), not
    # an identity: "Columbia agar with sheep blood" and "Columbia Agar with 5%
    # Sheep Blood" are one medium. NaCl percentages are NOT stripped -- for a
    # halophile, salinity IS the medium ("MA + 3% NaCl" vs "MA + 10% NaCl").
    (re.compile(r"\d+(?:\.\d+)?\s*%?\s*(?=(?:defibrinated\s+)?"
                r"(?:sheep|horse|human|rabbit)?\s*blood)", re.IGNORECASE), " "),
    (re.compile(r"brain[\s-]*heart[\s-]*infusion", re.IGNORECASE), "brain heart infusion"),
    (re.compile(r"de\s*man[\s,-]*rogosa[\s,-]*(?:and\s*)?sharpe", re.IGNORECASE),
     "de man rogosa sharpe"),
    # bare "Man-Rogosa-Sharpe" drops the "de"
    (re.compile(r"(?<!de )man[\s,-]+rogosa[\s,-]+(?:and\s*)?sharpe", re.IGNORECASE),
     "de man rogosa sharpe"),
)

# Atmosphere annotations. A gas phase is a culture condition, not part of the
# medium's identity, so "WILKINS-CHALGREN ANAEROBE BROTH (N2/CO2)" must reach
# the same key as the same broth without the annotation.
_GAS_TOKENS = frozenset({"n2", "co2", "h2", "o2", "air"})

# Parenthetical content that is a vendor or an acronym restatement, not a
# distinct formulation: "(Oxoid)", "(Difco)", "(BD)", "(TSA)", "(DSMZ Medium 830)".
_VENDOR_PAREN_RE = re.compile(
    r"\((?:\s*(?:oxoid|difco|bd|merck|sigma|himedia|becton[^)]*|dsmz[^)]*|"
    r"[a-z0-9\s.-]{1,6})\s*)\)",
    re.IGNORECASE,
)
_ANY_PAREN_RE = re.compile(r"\([^)]*\)")
_DSMZ_TAIL_RE = re.compile(r"\b(?:dsmz|atcc|jcm|nbrc)\s*medium\s*[\w.-]*", re.IGNORECASE)

# Separators that can join two medium names in one field.
_LIST_SPLIT_RE = re.compile(r"\s*(?:,|;|\bor\b|\band\b|\+)\s*", re.IGNORECASE)
# A gas mixture or a dilution ratio -- never a medium list.
_GAS_MIX_RE = re.compile(r"\b(?:n2|co2|h2|o2|air)\b\s*/\s*\b(?:n2|co2|h2|o2|air)\b",
                         re.IGNORECASE)
_RATIO_RE = re.compile(r"/\s*\d")


def _media_tokens(name: str) -> list[str]:
    """Identity tokens for a medium name.

    Splits a short medium code from digits glued to either side ("2xLB" ->
    ["2", "lb"], "ISP2" -> ["isp", "2"]), then folds spelling variants via
    ``_MEDIA_TOKEN_CANON``. The trailing split matters: BacDive writes both
    "ISP 2" and "ISP2", which would otherwise land in different groups.
    """
    text = str(name or "").lower()
    for pattern, replacement in _PHRASE_CANON:
        text = pattern.sub(replacement, text)
    raw = [
        _MEDIA_TOKEN_CANON.get(t, t)
        for t in re.findall(r"[a-z0-9]+", text)
    ]
    out: list[str] = []
    for tok in raw:
        split = False
        for code in sorted(_SHORT_MEDIA_CODES, key=len, reverse=True):
            if len(tok) <= len(code):
                continue
            if tok.endswith(code):
                prefix = tok[: -len(code)]
                if prefix and all(c.isdigit() or c == "x" for c in prefix):
                    out.extend([c for c in prefix if c != "x"] + [code])
                    split = True
                    break
            if tok.startswith(code):
                suffix = tok[len(code):]
                if suffix and suffix.isdigit():
                    out.extend([code, suffix])
                    split = True
                    break
        if not split:
            out.append(tok)
    return out


def _medium_cores(name: str) -> set[str]:
    """Recognised medium identities named anywhere in ``name``.

    Lets :func:`is_multi_medium` tell a list of media from one name that
    merely contains a separator.
    """
    tokens = set(_media_tokens(name))
    cores = {t for t in tokens if t in _MEDIA_ACRONYMS}
    for core, phrase in (("macconkey", "macconkey"), ("r2a", "r2a")):
        if core in tokens:
            cores.add(phrase)
    return cores


def is_multi_medium(name: str) -> bool:
    """Return True if ``name`` lists two or more alternative media.

    BacDive sometimes records every medium tried in one field ("NA, 2xYT agar,
    TSA"), so the row cannot be assigned to a held-out medium group. Must not
    fire on gas mixtures ("PY + X MEDIUM (N2/CO2)") or dilution ratios
    ("BHI/2 MEDIUM").
    """
    text = str(name or "").strip()
    if not text:
        return False
    stripped = _GAS_MIX_RE.sub(" ", text)
    stripped = _RATIO_RE.sub(" ", stripped)
    stripped = _ANY_PAREN_RE.sub(" ", stripped)
    parts = [p for p in _LIST_SPLIT_RE.split(stripped) if p.strip()]
    if len(parts) < 2:
        return False
    # Two or more distinct recognised cores means a genuine list
    cores: set[str] = set()
    for part in parts:
        cores |= _medium_cores(part)
    return len(cores) >= 2


def canonical_media_name(name: str, overrides: dict[str, str] | None = None) -> str:
    """Return the canonical medium identity for ``name`` (the S6 grouping key).

    Casefolds, strips vendor/acronym parentheticals and DSMZ catalogue tails,
    drops generic form words, expands acronyms, folds spelling variants, then
    sorts the surviving tokens so word order cannot split a group.
    ``overrides`` (see :func:`load_media_alias_map`) is consulted first and
    wins outright.

    Multi-medium strings get an ``ambiguous:<slug>`` key and a blank name
    becomes ``unspecified``, so neither collides with a real group or trips NA
    handling in the split machinery. See :func:`media_canon_status`.
    """
    raw = str(name or "").strip()
    if not raw:
        return "unspecified"

    if overrides:
        hit = overrides.get(raw) or overrides.get(raw.lower())
        if hit:
            return hit

    if is_multi_medium(raw):
        slug = "_".join(_media_tokens(raw))[:60]
        return f"ambiguous:{slug}"

    text = _DSMZ_TAIL_RE.sub(" ", raw)
    text = _VENDOR_PAREN_RE.sub(" ", text)
    # Any surviving parenthetical is usually an acronym restatement of the
    # name itself ("Trypticase Soy Agar (TSA)"); keep its content so a
    # bare-acronym spelling still converges on the same tokens.
    text = text.replace("(", " ").replace(")", " ")

    tokens = _media_tokens(text)

    # Drop form words, preparation qualifiers and vendor names.
    tokens = [
        t for t in tokens
        if t not in _GENERIC_MEDIA_TOKENS
        and t not in _SKIP_MEDIA_TOKENS
        and t not in _VENDOR_TOKENS
    ]

    # Expand acronyms so short and long spellings converge.
    expanded: list[str] = []
    for tok in tokens:
        expanded.extend(_MEDIA_ACRONYMS.get(tok, tok).split())
    expanded = [_MEDIA_TOKEN_CANON.get(t, t) for t in expanded]
    expanded = [
        t for t in expanded
        if t not in _GENERIC_MEDIA_TOKENS
        and t not in _SKIP_MEDIA_TOKENS
        and t not in _GAS_TOKENS
        and t not in _VENDOR_TOKENS
    ]

    if not expanded:
        # Nothing but form words -- fall back to the casefolded raw name so
        # the row keeps a stable identity instead of collapsing into "".
        return re.sub(r"\s+", " ", raw.lower()).strip()

    return " ".join(sorted(set(expanded)))


def media_canon_status(name: str, overrides: dict[str, str] | None = None) -> str:
    """Return ``ok``, ``ambiguous`` or ``unspecified`` for a medium name.

    Emitted as its own column so a split can drop non-identifiable rows with a
    plain value list (``filters.exclude: {media_canon_status: [...]}``).
    """
    canon = canonical_media_name(name, overrides)
    if canon == "unspecified":
        return "unspecified"
    if canon.startswith("ambiguous:"):
        return "ambiguous"
    return "ok"


MEDIA_ALIAS_MAP_CSV = (
    Path(__file__).resolve().parents[2] / "data" / "external" / "media_alias_map.csv"
)


def load_media_alias_map(path: Path | str | None = None) -> dict[str, str]:
    """Load the curated ``media_name -> canonical_key`` override map.

    A missing file is normal and returns ``{}``. Produced by
    ``scripts/bacdive/apply_media_alias_review.py``. Both the raw and the
    casefolded spelling are registered.
    """
    csv_path = Path(path) if path is not None else MEDIA_ALIAS_MAP_CSV
    if not csv_path.exists():
        logger.info("no media alias map at {}; using deterministic rules only", csv_path)
        return {}

    overrides: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("media_name") or "").strip()
            key = (row.get("canonical_key") or "").strip()
            if not raw or not key:
                continue
            overrides[raw] = key
            overrides.setdefault(raw.lower(), key)
    logger.info("loaded {} media alias overrides from {}", len(overrides), csv_path)
    return overrides


@lru_cache(maxsize=1)
def get_media_alias_map() -> dict[str, str]:
    """Return the cached :func:`load_media_alias_map` for the default path.

    Fetched here because ``record_to_medium_rows`` shares a signature with the
    metabolite target. Call ``get_media_alias_map.cache_clear()`` after
    rewriting the CSV in the same process.
    """
    return load_media_alias_map()


# ---------------------------------------------------------------------------
# Composition features
# ---------------------------------------------------------------------------

def _count_ingredients(composition: str) -> int:
    """Rough ingredient count from the free-text composition.

    Handles newline-separated (DSMZ) and semicolon-separated (CIP) layouts,
    dropping header and filler lines ("Name:", "Distilled water", empties).
    """
    if not composition:
        return 0

    text = composition.strip()
    parts = re.split(r";|\n", text)

    skip_prefixes = ("name:", "composition:")
    count = 0
    for part in parts:
        token = part.strip().lower()
        if not token:
            continue
        if token.startswith(skip_prefixes):
            continue
        if "distilled water" in token or token.startswith("water"):
            continue
        count += 1
    return count


def media_features(name: str, link: str | None, composition: str | None) -> dict:
    """Return the media feature columns for one culture-medium entry.

    Keys match the medium feature columns in ``config.OUTPUT_FIELDNAMES``.
    """
    comp = (composition or "").strip()
    comp_lower = comp.lower()
    # Fall back to the medium name when composition is blank (many CIP/Pasteur
    # entries have a link but no composition text)
    haystack = comp_lower or (name or "").lower()

    flags = {
        col: int(any(tok in haystack for tok in tokens))
        for col, tokens in _MARKERS.items()
    }

    # Complexity is three-way:
    #   defined      -- composition present, a defined marker matched and no
    #                   complex marker did
    #   semi_defined -- composition present, both kinds of marker matched
    #   undefined    -- no composition, or no defined marker matched (e.g. a
    #                   modifier note like "at 37 C" or an unresolvable
    #                   solution code); "defined" must be positively evidenced
    if not comp_lower:
        flags["complexity"] = "undefined"
    else:
        has_complex = any(tok in comp_lower for tok in _COMPLEX_MARKERS)
        has_defined = any(tok in comp_lower for tok in _DEFINED_MARKERS)
        if has_complex and has_defined:
            flags["complexity"] = "semi_defined"
        elif has_defined:
            flags["complexity"] = "defined"
        else:
            flags["complexity"] = "undefined"

    return {
        "mediadive_id": parse_mediadive_id(link),
        "n_ingredients": _count_ingredients(comp),
        **flags,
        "composition_raw": comp,
    }
