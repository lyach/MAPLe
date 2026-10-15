#!/usr/bin/env python3
"""Turn a completed medium-name alias review sheet into the curated alias map.

Reads the reviewed CSV (from ``prepare_media_alias_review.py``, or a
reasoning pass over it) and writes ``data/external/media_alias_map.csv``,
which ``src.bacdive.media.canonical_media_name`` consults before its rules --
so a reviewed decision is never overturned by a later rule change. Rows with
a blank ``canonical_key`` mean "the rules already got this one right" and are
skipped, not treated as an error.

Identity sibling of ``apply_media_name_review.py``, which instead resolves a
medium to a MediaDive recipe -- that decides what it's made of, this decides
which medium it is.

After running this, re-run ``add_media_canon.py`` and regenerate any
held-out-medium split maps -- their group values change.

Usage:
    python scripts/bacdive/apply_media_alias_review.py data/external/media_alias_review_reasoned.csv
    python scripts/bacdive/apply_media_alias_review.py <sheet> --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.bacdive.media import MEDIA_ALIAS_MAP_CSV, canonical_media_name

OUT_FIELDNAMES = ["media_name", "canonical_key", "source", "review_kind", "notes"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("reviewed", type=Path,
                   help="Completed review CSV (media_name, canonical_key, ...)")
    p.add_argument("--map", type=Path, default=MEDIA_ALIAS_MAP_CSV,
                   help="Alias map to write/update (default: %(default)s)")
    p.add_argument("--source", default="manual_review",
                   help="Provenance tag for the added rows (default: %(default)s)")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would change without writing")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.reviewed.exists():
        print(f"error: reviewed file not found: {args.reviewed}", file=sys.stderr)
        return 1

    reviewed = pd.read_csv(args.reviewed, dtype=str, keep_default_na=False)
    for col in ("media_name", "canonical_key"):
        if col not in reviewed.columns:
            print(f"error: {args.reviewed} is missing required column {col!r}",
                  file=sys.stderr)
            return 1

    reviewed["media_name"] = reviewed["media_name"].str.strip()
    reviewed["canonical_key"] = reviewed["canonical_key"].str.strip()
    confirmed = reviewed[
        (reviewed["canonical_key"] != "") & (reviewed["media_name"] != "")
    ].copy()
    print(f"{len(reviewed)} reviewed row(s), {len(confirmed)} with a canonical_key")

    # One decision per media_name. A name given two different keys is a
    # contradiction in the sheet, not something to resolve silently.
    clashes = (
        confirmed.groupby("media_name")["canonical_key"].nunique().pipe(lambda s: s[s > 1])
    )
    if len(clashes):
        print(f"error: {len(clashes)} media_name(s) were assigned more than one "
              f"canonical_key -- fix the sheet:", file=sys.stderr)
        for name in clashes.index[:10]:
            keys = sorted(confirmed.loc[confirmed.media_name == name, "canonical_key"].unique())
            print(f"  {name!r}: {keys}", file=sys.stderr)
        return 1

    new = confirmed.drop_duplicates("media_name").copy()
    new["source"] = args.source
    for col in ("review_kind", "notes"):
        if col not in new.columns:
            new[col] = ""
    # A reasoning pass writes its rationale in `reasoning`; keep it in `notes`.
    if "reasoning" in new.columns:
        new["notes"] = new.apply(
            lambda r: r["reasoning"] or r["notes"], axis=1
        )
    new = new[OUT_FIELDNAMES]

    # Report only the decisions that actually change something.
    changes = [
        (r.media_name, canonical_media_name(r.media_name), r.canonical_key)
        for r in new.itertuples()
        if canonical_media_name(r.media_name) != r.canonical_key
    ]
    print(f"{len(changes)} decision(s) differ from the deterministic rules "
          f"({len(new) - len(changes)} confirm the rules)")
    for name, was, now in changes[:15]:
        print(f"  {name!r}: {was!r} -> {now!r}")
    if len(changes) > 15:
        print(f"  ... and {len(changes) - 15} more")

    if args.map.exists():
        existing = pd.read_csv(args.map, dtype=str, keep_default_na=False)
        # Replace any prior row for the same media_name, so re-running after a
        # corrected sheet cannot leave a stale decision behind.
        existing = existing[~existing["media_name"].isin(new["media_name"])]
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new

    if args.dry_run:
        print(f"\n[dry run] would write {len(combined)} row(s) to {args.map}")
        return 0

    args.map.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.map, index=False)
    print(f"\nwrote {args.map} ({len(combined)} total override(s))")
    print("Next: python scripts/bacdive/add_media_canon.py, then regenerate the "
          "held-out-medium split maps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
