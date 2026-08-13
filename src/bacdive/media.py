"""
media.py
========
Turn a BacDive ``culture medium`` entry into modeling features.

BacDive uses a large, open vocabulary of media names (hundreds of DSMZ / CIP
recipes), so ``media_name`` as an identity one-hot does not transfer across
media under leave-one-medium-out CV. Instead we parse the free-text
``composition`` into a handful of presence flags (mirroring the columns in
``local/media_features.csv``) plus an ingredient count, and pull a stable
MediaDive medium id out of the ``link`` when present.
"""

from __future__ import annotations

import re

# Substring markers (lowercased) used to derive presence flags. Each maps a
# feature column to the tokens that, if found in the composition text, set it.
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
)

_MEDIADIVE_RE = re.compile(r"mediadive\.dsmz\.de/medium/(\d+)", re.IGNORECASE)


def parse_mediadive_id(link: str | None) -> str:
    """Extract the numeric MediaDive medium id from a ``link`` URL.

    Returns "" for missing links or non-MediaDive links (e.g. CIP/Pasteur).
    """
    if not link:
        return ""
    match = _MEDIADIVE_RE.search(str(link))
    return match.group(1) if match else ""


def _count_ingredients(composition: str) -> int:
    """Rough ingredient count from the free-text composition.

    BacDive compositions come in two layouts:
      * newline-separated "Ingredient 10.0 g/l" lines (DSMZ style), and
      * semicolon-separated "Ingredient (10.0 g)" clauses (CIP style).
    We split on whichever separator dominates and drop obvious header/filler
    lines ("Name:", "Composition:", "Distilled water", empties).
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

    Keys match the ``media`` feature columns in the output schema
    (``config.OUTPUT_FIELDNAMES``).
    """
    comp = (composition or "").strip()
    comp_lower = comp.lower()
    # Fall back to the medium name for flag detection when composition is blank
    # (many CIP/Pasteur entries have a link but no composition text).
    haystack = comp_lower or (name or "").lower()

    flags = {
        col: int(any(tok in haystack for tok in tokens))
        for col, tokens in _MARKERS.items()
    }

    # complexity: three-way classification of medium chemical definition.
    #   defined     — composition present, no complex markers detected
    #   semi_defined — composition present, both complex and defined markers present
    #   undefined   — no composition, or complex markers with no defined markers
    if not comp_lower:
        flags["complexity"] = "undefined"
    else:
        has_complex = any(tok in comp_lower for tok in _COMPLEX_MARKERS)
        has_defined = any(tok in comp_lower for tok in _DEFINED_MARKERS)
        if has_complex and has_defined:
            flags["complexity"] = "semi_defined"
        elif has_complex:
            flags["complexity"] = "undefined"
        else:
            flags["complexity"] = "defined"

    return {
        "mediadive_id": parse_mediadive_id(link),
        "n_ingredients": _count_ingredients(comp),
        **flags,
        "composition_raw": comp,
    }
