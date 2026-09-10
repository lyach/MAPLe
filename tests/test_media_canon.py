#!/usr/bin/env python3
"""Regression tests for the canonicalised media name (`media_canon` column).

Every case here came from auditing real BacDive medium names (see
``scripts/bacdive/prepare_media_alias_review.py``), and the two directions
matter differently:

* A missing merge is a leak: the "held-out" medium is still in the
  training set under another spelling.
* A wrong merge only costs group cardinality (both variants are held out
  together, so nothing leaks), but it collapses genuinely different recipes.

Usage:
    python tests/test_media_canon.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bacdive.media import (
    canonical_media_name,
    is_multi_medium,
    media_canon_status,
)

# Groups of raw BacDive names that MUST collapse to one key.
MERGE_GROUPS: list[tuple[str, list[str]]] = [
    ("MacConkey: case, typo, hyphen, possessive, plural, vendor, acronym", [
        "MacConkey agar", "MacConkey", "MacConkey medium", "MacConkey Agar",
        "MacConkey agar (Oxoid)", "MacConcey agar", "MacConkey agar (Difco)",
        "MacConkey's agar", "MacConkey agar (BD)", "Mac-Conkey agar",
        "MacConkey agar (MCA)", "McConkey agar", "MacConkey agars",
    ]),
    ("Tryptic/trypticase soy, agar vs broth, casein synonym, vendor", [
        "TSA (Trypticase soy agar)", "Tryptic Soy Agar", "Trypticase Soy Agar",
        "Trypticase Soy Agar (TSA)", "tryptic soy agar", "trypticase soy agar (TSA)",
        "Trypticase soy broth", "tryptic soy broth (TSB)",
        "trypticase soy agar (TSA; BBL)", "Trypticase Casein Soy Agar",
        "TRYPTONE SOYA BROTH (TSB) (DSMZ Medium 545)",
    ]),
    ("LB: acronym, Luria-Bertani, lysogeny broth", [
        "LB (Luria-Bertani) MEDIUM", "LB medium", "LB broth",
        "Luria-Bertani broth", "LA", "LB (lysogeny broth) agar",
        "lysogeny broth agar",
    ]),
    ("Nutrient agar/broth: acronym, typo, vendor forms", [
        "NA agar", "Nutrient agar (NA)", "nutrient agar (NA)", "nutrient broth",
        "nutritient agar (NA)", "nutrient agar (Difco)", "nutrient agar (NA; Difco)",
        "nutrient agar (Nu, Oxoid)", "NB", "nutrient agar plates",
    ]),
    ("Marine agar: acronym, 2216 formulation number, vendors", [
        "MA agar", "Marine agar (MA)", "Marine agar (MA) 2216",
        "marine agar 2216", "marine agar (MA, Becton-Dickinson)",
        "marine agar 2216 (Difco)", "MA plates", "MB",
    ]),
    ("R2A: acronym vs Reasoner's 2A", [
        "Reasoner's 2A agar (R2A)", "R2A MEDIUM (DSMZ Medium 830)", "R2A agar",
        "r2a",
    ]),
    ("MRS: acronym, De Man/DeMan/Man-Rogosa-Sharpe", [
        "MRS agar", "De Man, Rogosa and Sharpe Agar", "Man-Rogosa-Sharpe agar",
        "DeMan, Rogosa and Sharpe agar",
    ]),
    ("Brain heart infusion: acronym and 'infuison' typo", [
        "BHI", "brain heart infusion agar", "Brain-Heart infuison agar",
        "Brain Heart Infusion (BHI) agar", "BHI MEDIUM (DSMZ Medium 215)",
    ]),
    ("Blood agar: standard 5% level and 'base' are not identities", [
        "blood agar", "Blood agar", "Blood-agar", "Blood Agar", "5% blood agar",
        "blood agar base",
    ]),
    ("Columbia + sheep blood: percentage, defibrinated, atmosphere", [
        "Columbia agar with sheep blood", "Columbia Agar with 5% Sheep Blood",
        "Columbia agar with 5 % sheep blood (microaerobic)",
        "Columbia agar with 5 % defibrinated sheep blood agar",
    ]),
    ("YMA: 'extract' is implied by the ingredient", [
        "Yeast Mannitol Agar", "yeast extract-mannitol agar", "YMA",
        "YMA (yeast mannitol agar)", "YMA (yeast extract mannitol agar)",
        "yeast extract-mannitol agar (YMA)",
    ]),
    ("TCBS: spelled out with and without 'salts'/'sucrose'", [
        "TCBS", "TCBS agar", "Thiosulfate Citrate Bile Sucrose (TCBS) agar",
        "thiosulfate citrate bile salts sucrose agar medium",
        "thiosulfate-citrate-bile salts agar (TCBS; Oxoid)",
    ]),
    ("Czapek / Czapek-Dox", ["Czapek's agar", "Czapek’s-Dox agar"]),
    ("Dnase 'test' is generic", ["Dnase agar", "Dnase test agar"]),
    ("Peptone yeast extract, spelled and acronym", [
        "PYE agar (Peptone Yeast Extract Agar)", "peptone-yeast-extract agar",
        "Peptone yeast extract agar",
    ]),
    ("Chrysoidin/chrysoidine spelling", [
        "bile-chrysoidin-glycerol agar", "Bile-chrysoidine-glycerol agar",
    ]),
    ("Gas phase is a condition, not a medium identity", [
        "WILKINS-CHALGREN ANAEROBE BROTH (N2/CO2)",
        "WILKINS-CHALGREN ANAEROBE BROTH",
    ]),
    # A digit glued to the RIGHT of a code, not just the left. BacDive writes
    # this medium both ways and the two spellings carried 46 and 14 no-growth
    # rows in separate groups before the trailing split was added.
    ("ISP 2 vs ISP2: digit glued to the acronym", [
        "ISP 2", "ISP2", "ISP 2 agar", "ISP 2 medium", "ISP 2 Agar",
        "ISP2 agar", "ISP2 MEDIUM (DSMZ Medium 987)", "ISP2 medium",
        "ISP2 agar medium", "Modified ISP2 Medium (DSMZ Medium 1832)",
    ]),
]

# Pairs that MUST stay in different groups: genuinely different recipes.
SPLIT_PAIRS: list[tuple[str, str, str]] = [
    ("MacConkey agar", "sorbitol MacConkey agar",
     "sorbitol MacConkey is a distinct selective medium"),
    ("MacConkey agar", "MacConkey agar without crystal violet",
     "omitting crystal violet changes the selectivity"),
    ("MA supplemented with 3 % (w/v) NaCl", "MA supplemented with 10 % (w/v) NaCl",
     "for a halophile, salinity IS the medium"),
    ("STARCH - MINERAL SALT - AGAR (STMS) (DSMZ Medium 252)",
     "STARCH-MINERAL SALT-AGAR (STMS) + 10%NACL (DSMZ Medium 1240)",
     "added NaCl changes the medium"),
    ("STARCH-MINERAL SALT-AGAR (STMS) + 10%NACL (DSMZ Medium 1240)",
     "STARCH-MINERAL SALT-AGAR (STMS) + 15% NACL (DSMZ Medium 1241)",
     "different salinity"),
    ("Columbia agar", "Columbia blood agar", "blood is a real supplement"),
    ("Brucella agar", "Brucella blood agar", "blood is a real supplement"),
    ("chocolate agar", "Chocolate blood agar", "blood is a real supplement"),
    ("Columbia Agar with 5% Sheep Blood", "Columbia Agar with 5% Horse Blood",
     "blood source differs"),
    ("SKIM MILK 1 AGAR", "SKIM MILK 3 AGAR", "numbered DSMZ media are distinct"),
    ("ISP 2", "ISP 4", "numbered ISP media are distinct recipes"),
    ("ISP 2 agar", "ISP 5 agar", "numbered ISP media are distinct recipes"),
    ("Yeast extract-malt extract agar", "glucose-yeast extract-malt extract agar",
     "YM and GYM are different media"),
    ("BHI MEDIUM (DSMZ Medium 215)", "BHI/2 MEDIUM (DSMZ Medium 217)",
     "half-strength BHI is a different formulation"),
    ("Peptone yeast extract agar", "Tryptone yeast extract agar",
     "peptone and tryptone are different reagents"),
]

# Names that list several alternative media -- medium undefined for that row.
MULTI_MEDIUM = [
    "NA, 2xYT agar, TSA, R2A agar, MacConkey agar",
    "PCA, TSA, NA, R2A, MA and MacConkey agar",
]

# Single media whose names merely contain a separator: the false positives a
# naive comma/slash split produces (a bare heuristic flagged 350 rows, almost
# all of them these).
NOT_MULTI_MEDIUM = [
    "PY + X MEDIUM (N2/CO2) (DSMZ Medium 104c)",
    "BHI/2 MEDIUM (DSMZ Medium 217)",
    "VY/2 AGAR (DSMZ Medium 9)",
    "METHANOGENIUM MEDIUM (H2/CO2) (DSMZ Medium 141)",
    "WILKINS-CHALGREN ANAEROBE BROTH (N2/CO2) (DSMZ Medium 339a)",
    "SSE/HD1:10 (DSMZ Medium 1426)",
    "FERROUS SULFATE/YEAST EXTRACT MEDIUM (DSMZ Medium 1190)",
    "GAM Broth, modified (Gifu Anaerobic Medium Broth, modified) (DSMZ Medium 1715)",
]


def test_merge_groups() -> None:
    for label, names in MERGE_GROUPS:
        keys = {canonical_media_name(n) for n in names}
        assert len(keys) == 1, (
            f"{label}: expected one canonical key, got {len(keys)}: "
            f"{sorted(keys)}\n  "
            + "\n  ".join(f"{n!r} -> {canonical_media_name(n)!r}" for n in names)
        )


def test_split_pairs() -> None:
    for left, right, why in SPLIT_PAIRS:
        lk, rk = canonical_media_name(left), canonical_media_name(right)
        assert lk != rk, (
            f"{left!r} and {right!r} both canonicalised to {lk!r} "
            f"but must stay separate ({why})"
        )


def test_multi_medium_detection() -> None:
    for name in MULTI_MEDIUM:
        assert is_multi_medium(name), f"{name!r} lists several media"
        assert canonical_media_name(name).startswith("ambiguous:"), name
        assert media_canon_status(name) == "ambiguous", name
    for name in NOT_MULTI_MEDIUM:
        assert not is_multi_medium(name), (
            f"{name!r} is one medium; a separator in its name is not a list"
        )
        assert media_canon_status(name) == "ok", name


def test_blank_name_is_named_not_empty() -> None:
    # An empty grouping value risks NA-comparison surprises in the split
    # machinery, so a missing medium gets an explicit token instead.
    for blank in ("", "   ", None):
        assert canonical_media_name(blank) == "unspecified", repr(blank)
        assert media_canon_status(blank) == "unspecified", repr(blank)


def test_overrides_win_over_rules() -> None:
    overrides = {"MacConkey agar": "hand_picked_key"}
    assert canonical_media_name("MacConkey agar", overrides) == "hand_picked_key"
    # ... and do not disturb names they do not mention
    assert canonical_media_name("MacConkey", overrides) == "macconkey"


def test_is_idempotent() -> None:
    # Re-canonicalising an already-canonical key must be a no-op, so that
    # re-running add_media_canon.py cannot drift the grouping.
    for _, names in MERGE_GROUPS:
        for name in names:
            once = canonical_media_name(name)
            assert canonical_media_name(once) == once, f"{name!r} -> {once!r}"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {test.__name__}\n  {exc}\n")
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
