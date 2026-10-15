#!/usr/bin/env python3
"""
apply_media_name_review.py
============================
Feeds confirmed matches from a completed media-name review sheet (see
``prepare_media_name_review.py`` and the reasoning passes over its output)
into ``data/external/media_recipe_manifest.csv``, so
``src.bacdive.pipeline._backfill_composition`` picks them up as
``composition_source == "manual_review"`` on the next rebuild.

Input: a review CSV with (at least) ``media_name``, ``chosen_mediadive_id``,
``chosen_mediadive_name`` columns -- either a human's filled-in worksheet, or
the output of a reasoning pass such as ``media_name_review_reasoned.csv``.
Rows with a blank ``chosen_mediadive_id`` are skipped (no confident match).

For each confirmed row, the chosen MediaDive id's recipe is read from the
existing cache (``media_recipe_cache/mediadive/<id>.json`` -- already fetched
by prepare_media_name_review.py; nothing new is fetched here) and appended to
the manifest with ``source=manual_review``, keyed by ``query_name`` (the
BacDive ``media_name``) for the pipeline's by-name join.

USAGE:
    python scripts/bacdive/apply_media_name_review.py data/external/media_name_review_reasoned.csv
    python scripts/bacdive/apply_media_name_review.py data/external/media_name_review_reasoned.csv --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.bacdive.fetch_media_recipes import (
    CACHE_DIR, MANIFEST_CSV, MANIFEST_FIELDNAMES, RecipeResult, fetch_mediadive,
)
from src.strain_metadata.logging_setup import setup_logging

log = logging.getLogger("apply_media_name_review")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reviewed", type=Path, help="Completed review CSV (media_name, chosen_mediadive_id, ...)")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_CSV,
                         help="Manifest to append to (default: %(default)s)")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--dry-run", action="store_true",
                         help="Report what would be added without writing the manifest")
    args = parser.parse_args()

    setup_logging(Path(__file__).resolve().parents[2])

    if not args.reviewed.exists():
        log.error("reviewed file not found: %s", args.reviewed)
        return 1

    reviewed = pd.read_csv(args.reviewed, dtype=str, keep_default_na=False)
    for col in ("media_name", "chosen_mediadive_id"):
        if col not in reviewed.columns:
            log.error("%s is missing required column %r", args.reviewed, col)
            return 1

    confirmed = reviewed[reviewed["chosen_mediadive_id"].str.strip() != ""]
    log.info("%d confirmed match(es) in %s", len(confirmed), args.reviewed)

    new_rows: list[RecipeResult] = []
    missing_cache = 0
    for _, row in confirmed.iterrows():
        query_name = row["media_name"].strip()
        medium_id = row["chosen_mediadive_id"].strip()
        cache_file = args.cache_dir / "mediadive" / f"{medium_id}.json"
        if not cache_file.exists():
            log.warning("no cached recipe for id %s (media_name=%r) -- skipping; "
                        "run prepare_media_name_review.py again to cache it first",
                        medium_id, query_name)
            missing_cache += 1
            continue
        result = fetch_mediadive(medium_id, "", args.cache_dir, force=False,
                                  source="manual_review", query_name=query_name)
        if result.status != "ok":
            log.warning("cached recipe for id %s (media_name=%r) has status=%s, not ok -- skipping",
                        medium_id, query_name, result.status)
            continue
        new_rows.append(result)

    log.info("%d row(s) ready to add (%d skipped: no cache)", len(new_rows), missing_cache)

    if args.dry_run:
        for r in new_rows[:10]:
            print(f"  {r.query_name!r} -> id={r.key} ({r.name}), "
                  f"{r.n_ingredients} ingredient(s)")
        if len(new_rows) > 10:
            print(f"  ... and {len(new_rows) - 10} more")
        print(f"\n[dry run] would append {len(new_rows)} row(s) to {args.manifest}")
        return 0

    new_df = pd.DataFrame([asdict(r) for r in new_rows], columns=MANIFEST_FIELDNAMES)

    if args.manifest.exists():
        existing = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
        # replace any prior manual_review row for the same query_name (re-running
        # this script after a corrected worksheet shouldn't duplicate/stale-out)
        existing = existing[
            ~((existing["source"] == "manual_review")
              & existing["query_name"].isin(new_df["query_name"]))
        ]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.manifest, index=False)

    print()
    print("=" * 60)
    print("Applied media-name review to manifest")
    print("=" * 60)
    print(f"Confirmed matches in review file : {len(confirmed)}")
    print(f"Skipped (no cached recipe)       : {missing_cache}")
    print(f"Added as source=manual_review    : {len(new_rows)}")
    print(f"Manifest                         : {args.manifest} ({len(combined)} total rows)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
