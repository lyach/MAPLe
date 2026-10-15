#!/usr/bin/env python3
"""
prepare_media_name_review.py
=============================
Builds a human-review worksheet for BacDive "category D" media: entries with
a blank ``composition_raw``, no link of any kind (``medium_link``,
``cip_medium_number``/``cip_medium_link``), and no exact-name match against
MediaDive (``fetch_media_recipes.py``'s ``mediadive_by_name`` source already
handles the exact matches; those are excluded here).

These are common, well-known media (LB, MacConkey, MRS, ISP2, ...) where an
automatic match is too risky to make alone -- MediaDive often carries several
near-duplicate variants of one name (e.g. "ISP 2 MEDIUM" vs "ISP 2 MEDIUM
WITH 10% NACL" vs "Modified ISP2 Medium"), and picking the wrong one would
silently attach an incorrect recipe. Instead this script surfaces the top
candidate MediaDive entries per name for a person (or an LLM reasoning pass,
see apply_media_name_review.py's sibling reasoning step) to review and pick
from, or reject outright.

MediaDive's REST API and web UI both ignore search/query parameters (checked
directly: ``/rest/media?search=X`` and ``/media?q=X`` both just return the
full unfiltered catalogue) -- there is no server-side search to call. All
matching here is local, against the one cached full listing
(``fetch_media_list``), scored by a blend of word (Jaccard) overlap,
character-level similarity (``difflib``), and a small abbreviation/synonym
table (TSA, LB, ISP, NA, MRS, R2A, BHI, ...) expanded on both sides before
scoring -- catches things pure word-overlap misses (e.g. "Bennetts Agar" vs
"BENNETT'S AGAR") without over-fuzzing unrelated names into false positives.

Writes:
    <--out, default data/external/media_name_review.csv>
        One row per distinct media_name, its row count in the dataset, and
        up to --top-k (default 5, max 6) candidate MediaDive entries (id,
        name, score, and a composition preview) plus empty decision/notes
        columns to fill in.
    data/external/media_recipe_cache/mediadive/<id>.json
        Every candidate's full recipe, fetched and cached ahead of time (the
        same cache fetch_media_recipes.py uses) so reviewing doesn't require
        any network access.

After filling in the ``chosen_mediadive_id`` column (leave blank for "no
good candidate"), feed the completed sheet back with
``apply_media_name_review.py`` to add those matches to
data/external/media_recipe_manifest.csv as source=manual_review.

USAGE:
    python scripts/bacdive/prepare_media_name_review.py
    python scripts/bacdive/prepare_media_name_review.py --top-k 6
    # a follow-up round after a reasoning pass on the first: skip names it
    # already matched, write a separate worksheet for what's left
    python scripts/bacdive/prepare_media_name_review.py \\
        --reasoned data/external/media_name_review_reasoned.csv \\
        --out data/external/media_name_review_round2.csv
    python scripts/bacdive/prepare_media_name_review.py --limit 20   # smoke test
"""

from __future__ import annotations

import argparse
import csv
import difflib
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.bacdive.fetch_media_recipes import (
    CACHE_DIR, DEFAULT_INPUT, MANIFEST_CSV, REQUEST_DELAY,
    _needs_external_source, _present_columns,
    fetch_media_list, fetch_mediadive,
)
from src.strain_metadata.logging_setup import setup_logging

log = logging.getLogger("prepare_media_name_review")

REVIEW_CSV = Path(__file__).resolve().parents[2] / "data" / "external" / "media_name_review.csv"
PREVIEW_LEN = 150
MAX_CANDIDATE_SLOTS = 6

# Common medium-name abbreviations, expanded on both the query and every
# MediaDive name before scoring -- pure word-overlap treats "TSA" and
# "Trypticase Soy Agar" as unrelated tokens.
SYNONYMS = {
    "tsa": "trypticase soy agar", "tsb": "trypticase soy broth",
    "na": "nutrient agar", "lb": "luria bertani",
    "isp": "international streptomyces project",
    "mrs": "de man rogosa sharpe", "r2a": "reasoners 2a agar",
    "bhi": "brain heart infusion", "tcbs": "thiosulfate citrate bile salts sucrose",
    "ma": "marine agar", "yma": "yeast mannitol agar",
}

REVIEW_FIELDNAMES = ["media_name", "row_count"]
for i in range(1, MAX_CANDIDATE_SLOTS + 1):
    REVIEW_FIELDNAMES += [
        f"candidate{i}_id", f"candidate{i}_name", f"candidate{i}_score",
        f"candidate{i}_preview",
    ]
REVIEW_FIELDNAMES += ["chosen_mediadive_id", "notes"]


def _expand_tokens(name: str) -> tuple[set[str], list[str]]:
    """Raw tokens plus a synonym-expanded token set for scoring."""
    raw = re.findall(r"[a-z0-9]+", str(name or "").lower())
    expanded: list[str] = []
    for t in raw:
        expanded.extend(SYNONYMS.get(t, t).split())
    return set(expanded), raw


def _score(query_name: str, candidate_name: str) -> float:
    """Blend of synonym-expanded Jaccard overlap and character-level
    similarity (``difflib``). Neither alone is enough: pure word-overlap
    misses close spelling variants (e.g. apostrophes: "Bennetts Agar" vs
    "BENNETT'S AGAR"), pure character similarity misses abbreviation
    expansions (e.g. "TSA" vs "Trypticase Soy Agar" share almost no
    characters). Weighted 60/40 toward word overlap since a shared
    *word* is a stronger identity signal than shared characters.
    """
    q_exp, q_raw = _expand_tokens(query_name)
    c_exp, c_raw = _expand_tokens(candidate_name)
    if not q_exp and not c_exp:
        return 0.0
    jaccard = len(q_exp & c_exp) / len(q_exp | c_exp) if (q_exp | c_exp) else 0.0
    ratio = difflib.SequenceMatcher(None, " ".join(q_raw), " ".join(c_raw)).ratio()
    return 0.6 * jaccard + 0.4 * ratio


def top_candidates(query_name: str, media_list: list[dict], k: int) -> list[tuple[float, dict]]:
    """Top-``k`` MediaDive entries by ``_score`` against ``query_name``."""
    if not str(query_name or "").strip():
        return []
    scored = [(_score(query_name, m.get("name")), m) for m in media_list]
    scored.sort(key=lambda x: -x[0])
    return scored[:k]


def already_name_matched(manifest_path: Path) -> set[str]:
    """``query_name``s the exact-match step already resolved -- exclude them."""
    if not manifest_path.exists():
        return set()
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    resolved = manifest[
        (manifest["source"] == "mediadive_by_name") & (manifest["status"] == "ok")
    ]
    return set(resolved["query_name"].astype(str).str.strip())


def already_reasoned_matched(reasoned_csv: Optional[Path]) -> set[str]:
    """``media_name``s a prior reasoning pass already confidently matched --
    exclude them from a follow-up round rather than re-reviewing."""
    if reasoned_csv is None or not reasoned_csv.exists():
        return set()
    reasoned = pd.read_csv(reasoned_csv, dtype=str, keep_default_na=False)
    matched = reasoned[reasoned["chosen_mediadive_id"].astype(str).str.strip() != ""]
    return set(matched["media_name"].astype(str).str.strip())


def load_category_d(input_path: Path, exclude_names: set[str]) -> pd.Series:
    """``media_name`` -> row count for entries with no link and no composition,
    excluding names already resolved by the exact-match step."""
    columns = ["composition_raw", "media_name", "medium_link",
               "cip_medium_number", "cip_medium_link"]
    df = pd.read_parquet(input_path, columns=_present_columns(input_path, columns))

    def blank(s):
        return s.isna() | (s.astype(str).str.strip() == "")

    no_link = (
        _needs_external_source(df)
        & blank(df["medium_link"])
        & blank(df["cip_medium_number"])
        & blank(df["cip_medium_link"])
    )
    names = df.loc[no_link, "media_name"].astype(str).str.strip()
    names = names[~names.isin(exclude_names) & (names != "")]
    return names.value_counts()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_CSV,
                         help="Recipe manifest to exclude already-resolved names from")
    parser.add_argument("--reasoned", type=Path, default=None,
                         help="A prior reasoning pass's output CSV (media_name, "
                              "chosen_mediadive_id, ...) -- names it already "
                              "confidently matched are excluded from this round")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--out", type=Path, default=REVIEW_CSV)
    parser.add_argument("--top-k", type=int, default=5,
                         help=f"Candidates per name (max {MAX_CANDIDATE_SLOTS} columns wide)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N names by row count (smoke test)")
    parser.add_argument("--force", action="store_true",
                         help="Re-fetch cached candidate recipes")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY)
    args = parser.parse_args()

    setup_logging(Path(__file__).resolve().parents[2])

    if not args.input.exists():
        log.error("input file not found: %s", args.input)
        return 1
    top_k = min(args.top_k, MAX_CANDIDATE_SLOTS)

    exclude_names = already_name_matched(args.manifest) | already_reasoned_matched(args.reasoned)
    counts = load_category_d(args.input, exclude_names)
    if args.limit is not None:
        counts = counts.head(args.limit)
    log.info("%d distinct category-D media name(s) to review (%d rows)", len(counts), counts.sum())

    media_list = fetch_media_list(args.cache_dir, args.force)
    log.info("MediaDive catalogue: %d entries", len(media_list))

    # Resolve candidates for every name first, so we know the deduplicated
    # set of ids to fetch/cache before making any requests.
    per_name_candidates: dict[str, list[tuple[float, dict]]] = {}
    for name in counts.index:
        per_name_candidates[name] = top_candidates(name, media_list, top_k)

    candidate_ids = sorted({
        str(m["id"]) for cands in per_name_candidates.values() for _, m in cands
    })
    log.info("%d distinct candidate MediaDive id(s) to fetch/cache", len(candidate_ids))

    previews: dict[str, str] = {}
    for i, medium_id in enumerate(tqdm(candidate_ids, desc="Caching candidates", unit="medium")):
        result = fetch_mediadive(medium_id, "", args.cache_dir, args.force)
        previews[medium_id] = (result.composition_raw or "")[:PREVIEW_LEN]
        if i < len(candidate_ids) - 1:
            time.sleep(args.delay)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDNAMES)
        writer.writeheader()
        for name, row_count in counts.items():
            row = {"media_name": name, "row_count": row_count,
                   "chosen_mediadive_id": "", "notes": ""}
            for i, (score, m) in enumerate(per_name_candidates[name], start=1):
                medium_id = str(m["id"])
                row[f"candidate{i}_id"] = medium_id
                row[f"candidate{i}_name"] = m.get("name", "")
                row[f"candidate{i}_score"] = f"{score:.2f}"
                row[f"candidate{i}_preview"] = previews.get(medium_id, "")
            writer.writerow(row)

    with_candidates = sum(1 for c in per_name_candidates.values() if c)
    print()
    print("=" * 60)
    print("Media name review worksheet")
    print("=" * 60)
    print(f"Names to review          : {len(counts)}")
    print(f"  with >=1 candidate     : {with_candidates}")
    print(f"  with no candidate      : {len(counts) - with_candidates}")
    print(f"Candidate recipes cached : {len(candidate_ids)}")
    print(f"Worksheet written to     : {args.out}")
    print(f"Cache dir                : {args.cache_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
