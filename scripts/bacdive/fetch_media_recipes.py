#!/usr/bin/env python3
"""
fetch_media_recipes.py
=======================
Recover ``composition_raw`` text for BacDive culture-medium entries whose
record has no composition of its own but does carry a link to an external
catalogue:

  * CIP media (``cip_medium_link``) -- link to the Institut Pasteur CRBIP
    catalogue (``catalogue-crbip.pasteur.fr``). The recipe is scraped
    straight off the catalogue page's ingredient list.
  * DSMZ media whose ``medium_link`` is a direct PDF
    (``dsmz.de/.../DSMZ_Medium<id>.pdf``) rather than a MediaDive page --
    BacDive's own link-parsing only recognises ``mediadive.dsmz.de`` URLs
    (see ``src.bacdive.media.parse_mediadive_id``), so these never got a
    ``mediadive_id`` and their composition was never resolved. The same
    numeric/lettered id used in the PDF filename is usually also a valid
    MediaDive record id, queryable structured (JSON) at
    ``mediadive.dsmz.de/rest/medium/<id>`` -- no PDF parsing needed.

  * Media with no link at all (a plain name like "MacConkey agar") whose
    ``media_name`` matches a MediaDive entry name EXACTLY (after
    lowercasing/whitespace normalisation only -- see ``_norm_name``).
    Deliberately no fuzzy matching: MediaDive frequently carries several
    near-duplicate variants of a common name (e.g. "ISP 2 MEDIUM" vs "ISP 2
    MEDIUM WITH 10% NACL" vs "Modified ISP2 Medium"), and guessing among
    them risks attaching the wrong recipe -- worse than leaving it blank.
    Only ~120 of ~1300 such names clear this exact-match bar; the rest are
    left for ``pipeline._backfill_composition``'s in-corpus name-match step
    or manual curation.

Writes a per-item cache (raw HTML / JSON, one file per medium) plus one
manifest CSV row per item recording what was resolved and where it came
from -- the same cache + manifest pattern used by ``bacdive_cache`` and
``bacdive_genome_manifest.csv``:

    data/external/media_recipe_cache/cip/<number>.html
    data/external/media_recipe_cache/mediadive/<id>.json
    data/external/media_recipe_cache/mediadive_media_list.json
    data/external/media_recipe_manifest.csv

``src.bacdive.pipeline`` reads the manifest to backfill ``composition_raw``
on the next build; this script only populates the cache/manifest, it does
not touch any parquet file itself.

Both sources are queried gently (one request at a time, a fixed delay
between requests, capped retries) since these are small third-party
catalogues, not APIs meant for bulk access.

USAGE:
    python scripts/bacdive/fetch_media_recipes.py
    python scripts/bacdive/fetch_media_recipes.py --input data/external/bacdive_growth_full.parquet
    python scripts/bacdive/fetch_media_recipes.py --only cip --limit 10   # smoke test
    python scripts/bacdive/fetch_media_recipes.py --force                # re-fetch cached items
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.bacdive.media import looks_like_recipe
from src.strain_metadata.logging_setup import setup_logging

log = logging.getLogger("fetch_media_recipes")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "external" / "bacdive_growth_full.parquet"
CACHE_DIR = ROOT / "data" / "external" / "media_recipe_cache"
MANIFEST_CSV = ROOT / "data" / "external" / "media_recipe_manifest.csv"

USER_AGENT = "MicrobeNet/fetch_media_recipes (research use; contact: pavan.krishnan@mail.utoronto.ca)"
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 0.5  # seconds between requests -- both are small third-party sites
MAX_ATTEMPTS = 3

MANIFEST_FIELDNAMES = [
    "source", "key", "link", "name", "query_name", "composition_raw",
    "n_ingredients", "status", "error", "cache_file",
]
MEDIADIVE_MEDIA_LIST_URL = "https://mediadive.dsmz.de/rest/media"

_DSMZ_ID_RE = re.compile(r"DSMZ_Medium([0-9]+[a-z]?)\.pdf", re.IGNORECASE)
_LABEL_RE = re.compile(
    r'<label[^>]*class="ui-outputlabel ui-widget (pGrisG|pGris)"[^>]*>(.*?)</label>',
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class RecipeResult:
    source: str          # cip | mediadive | mediadive_by_name
    key: str              # cip number, or mediadive id
    link: str = ""
    name: str = ""              # MediaDive/CIP's own name for the medium
    query_name: str = ""        # our media_name this was matched from (mediadive_by_name only)
    composition_raw: str = ""
    n_ingredients: int = 0
    status: str = "failed"   # ok | empty | not_found | http_error | error
    error: str = ""
    cache_file: str = ""


def _norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip().lower())


def _present_columns(input_path: Path, columns: list[str]) -> list[str]:
    """``columns`` plus ``composition_source`` if that column exists in the
    file -- lets ``pd.read_parquet(..., columns=...)`` select it without
    erroring on a pre-backfill file that doesn't have it yet."""
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(input_path).schema_arrow.names)
    extra = ["composition_source"] if "composition_source" in available else []
    return [*columns, *extra]


def _needs_external_source(df: pd.DataFrame) -> pd.Series:
    """True for rows BacDive's own record does not answer for composition.

    Deliberately NOT just "composition_raw is blank": once a rebuild has
    applied a backfill, composition_raw is non-blank for rows this same
    external-fetch pipeline filled in, but they must still count as targets
    -- otherwise re-running against an already-improved parquet finds only
    the leftover gaps and, since the manifest is rewritten whole each run,
    silently drops every previously-resolved CIP/MediaDive entry from it.
    ``composition_source`` (added by pipeline._backfill_composition)
    distinguishes "record" (BacDive's own) from everything else, including
    still-blank (""); fall back to a blank check when that column isn't
    present (e.g. a pre-backfill extraction).
    """
    def blank(s):
        return s.isna() | (s.astype(str).str.strip() == "")

    if "composition_source" in df.columns:
        return df["composition_source"].astype(str).str.strip() != "record"
    return blank(df["composition_raw"])


def _http_get(url: str) -> bytes:
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/json"}
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            last_exc = exc
            if attempt == MAX_ATTEMPTS:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_exc = exc
            if attempt == MAX_ATTEMPTS:
                raise
        time.sleep(2 ** attempt)
    raise last_exc  # unreachable


def _clean_text(raw_html: str) -> str:
    """Strip inner tags (e.g. ``<SUP>brand</SUP>``) and normalise whitespace."""
    text = _TAG_RE.sub(" ", raw_html)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def parse_cip_page(raw_html: str) -> tuple[str, str]:
    """Extract ``(name, composition_raw)`` from a CRBIP catalogue page.

    Ingredient lines are single-line ``pGris`` labels carrying a mass/volume
    unit (``"Columbia agar 39 g (Bio-Rad 279240)"``); the free-text
    preparation protocol is a separate multi-line ``pGris`` label and is
    intentionally excluded -- it is not part of the recipe's composition.
    """
    name = ""
    ingredients: list[str] = []
    for css_class, inner in _LABEL_RE.findall(raw_html):
        if css_class == "pGrisG":
            name = _clean_text(inner)
            continue
        if "\n" in inner or "\r" in inner:
            continue  # multi-line -> preparation protocol, not an ingredient
        text = _clean_text(inner)
        if not text or not looks_like_recipe(text):
            continue
        # drop a trailing supplier/catalogue-number parenthetical, e.g.
        # "Columbia agar 39 g (Bio-Rad 279240)" -> "Columbia agar 39 g"
        text = re.sub(r"\s*\([^()]*\)\s*$", "", text).strip()
        ingredients.append(text)
    return name, ";".join(ingredients)


def parse_mediadive_json(data: dict) -> tuple[str, str]:
    """Extract ``(name, composition_raw)`` from a MediaDive ``/rest/medium/<id>`` payload.

    Formatted to match BacDive's own DSMZ-sourced ``composition_raw`` style
    (see e.g. "R2A MEDIUM (DSMZ Medium 830)" in the dataset): a
    ``Name: ...`` / ``Composition:`` header, one ``Ingredient amount unit``
    line per compound (pooled across every solution), and a trailing
    ``Distilled water`` line.
    """
    payload = data.get("data") or {}
    medium = payload.get("medium") or {}
    name = str(medium.get("name") or "").strip()

    lines: list[str] = []
    water_line = ""
    for solution in payload.get("solutions") or []:
        for item in solution.get("recipe") or []:
            compound = str(item.get("compound") or "").strip()
            if not compound:
                continue
            amount = item.get("amount")
            unit = str(item.get("unit") or "").strip()
            if compound.lower() in ("distilled water", "deionized water", "water"):
                if not water_line:
                    water_line = compound
                continue
            if amount is None:
                lines.append(compound)
            else:
                lines.append(f"{compound} {amount:g} {unit}".strip())
    if water_line:
        lines.append(water_line)

    header = f"Name: {name}\nComposition:\n" if name else "Composition:\n"
    composition_raw = header + "\n".join(lines) if lines else ""
    return name, composition_raw


def fetch_cip(number: str, link: str, cache_dir: Path, force: bool) -> RecipeResult:
    cache_file = cache_dir / "cip" / f"{number}.html"
    result = RecipeResult(source="cip", key=number, link=link, cache_file=str(cache_file))

    if cache_file.exists() and not force:
        raw = cache_file.read_text(encoding="utf-8")
    else:
        try:
            raw_bytes = _http_get(link)
        except urllib.error.HTTPError as exc:
            result.status, result.error = "http_error", f"HTTP {exc.code}"
            return result
        except Exception as exc:  # noqa: BLE001
            result.status, result.error = "error", str(exc)
            return result
        raw = raw_bytes.decode("utf-8", errors="replace")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(raw, encoding="utf-8")

    name, composition = parse_cip_page(raw)
    result.name = name
    result.composition_raw = composition
    result.n_ingredients = composition.count(";") + 1 if composition else 0
    result.status = "ok" if composition else "empty"
    return result


def fetch_mediadive(
    medium_id: str, link: str, cache_dir: Path, force: bool,
    *, source: str = "mediadive", query_name: str = "",
) -> RecipeResult:
    # Cache is keyed by MediaDive id alone -- the same record whether we
    # reached it via a DSMZ PDF link or a name match, so both share the
    # one cached fetch.
    cache_file = cache_dir / "mediadive" / f"{medium_id}.json"
    result = RecipeResult(
        source=source, key=medium_id, link=link, query_name=query_name,
        cache_file=str(cache_file),
    )

    if cache_file.exists() and not force:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        url = f"https://mediadive.dsmz.de/rest/medium/{medium_id}"
        try:
            raw_bytes = _http_get(url)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                result.status, result.error = "not_found", "not in MediaDive"
            else:
                result.status, result.error = "http_error", f"HTTP {exc.code}"
            return result
        except Exception as exc:  # noqa: BLE001
            result.status, result.error = "error", str(exc)
            return result
        data = json.loads(raw_bytes)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    if data.get("status") != 200:
        result.status, result.error = "not_found", str(data.get("msg") or "not found")
        return result

    name, composition = parse_mediadive_json(data)
    result.name = name
    result.composition_raw = composition
    result.n_ingredients = max(composition.count("\n") - 1, 0) if composition else 0
    result.status = "ok" if composition else "empty"
    return result


def load_targets(input_path: Path, only: Optional[str]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return ``(cip_targets, mediadive_targets)`` as ``(key, link)`` pairs.

    Candidates are rows BacDive's own record doesn't answer for --
    see ``_needs_external_source``.
    """
    columns = ["composition_raw", "mediadive_id", "medium_link", "cip_medium_number", "cip_medium_link"]
    df = pd.read_parquet(input_path, columns=_present_columns(input_path, columns))

    def blank(s):
        return s.isna() | (s.astype(str).str.strip() == "")

    comp_blank = _needs_external_source(df)

    cip_targets: list[tuple[str, str]] = []
    if only in (None, "cip"):
        cip_rows = df[comp_blank & ~blank(df["cip_medium_number"]) & ~blank(df["cip_medium_link"])]
        cip_targets = sorted(set(
            zip(cip_rows["cip_medium_number"].astype(str).str.strip(),
                cip_rows["cip_medium_link"].astype(str).str.strip())
        ))

    dsmz_targets: list[tuple[str, str]] = []
    if only in (None, "mediadive"):
        dsmz_rows = df[comp_blank & blank(df["mediadive_id"])
                       & df["medium_link"].astype(str).str.contains("dsmz.de", na=False)]
        for link in sorted(set(dsmz_rows["medium_link"].astype(str).str.strip())):
            m = _DSMZ_ID_RE.search(link)
            if m:
                dsmz_targets.append((m.group(1), link))

    return cip_targets, dsmz_targets


def fetch_media_list(cache_dir: Path, force: bool) -> list[dict]:
    """The full MediaDive catalogue (~3300 entries: id, name, link, ...).

    Fetched once and cached whole -- used to match a plain media name (no
    link at all) against a MediaDive id locally, rather than one search
    request per name.
    """
    cache_file = cache_dir / "mediadive_media_list.json"
    if cache_file.exists() and not force:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    raw = _http_get(MEDIADIVE_MEDIA_LIST_URL)
    data = json.loads(raw)
    media_list = data.get("data") or []
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(media_list, indent=2), encoding="utf-8")
    return media_list


def load_name_targets(
    input_path: Path, media_list: list[dict],
) -> list[tuple[str, str]]:
    """``(query_name, mediadive_id)`` pairs for exact normalized-name matches.

    Candidates: rows BacDive's own record doesn't answer for (see
    ``_needs_external_source``) with no link/CIP number of any kind -- a
    link-based fetch has nothing to go on for these. Deliberately
    exact-match only (see module docstring on why fuzzy matching is skipped).
    """
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
    query_names = sorted(set(df.loc[no_link, "media_name"].astype(str).str.strip()) - {""})

    name_index: dict[str, str] = {}
    for m in media_list:
        name_index.setdefault(_norm_name(m.get("name")), str(m.get("id")))

    targets = []
    for name in query_names:
        medium_id = name_index.get(_norm_name(name))
        if medium_id:
            targets.append((name, medium_id))
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                         help="BacDive growth parquet to scan for blank composition_raw rows")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_CSV)
    parser.add_argument("--only", choices=["cip", "mediadive", "name"], default=None,
                         help="Restrict to one source (default: all three)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N items per source (smoke test)")
    parser.add_argument("--force", action="store_true",
                         help="Re-fetch even if already cached")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY,
                         help=f"Seconds between requests (default: {REQUEST_DELAY})")
    args = parser.parse_args()

    setup_logging(ROOT)

    if not args.input.exists():
        log.error("input file not found: %s", args.input)
        return 1

    cip_targets, dsmz_targets = load_targets(args.input, args.only)

    name_targets: list[tuple[str, str]] = []
    if args.only in (None, "name"):
        media_list = fetch_media_list(args.cache_dir, args.force)
        log.info("MediaDive catalogue: %d entries", len(media_list))
        name_targets = load_name_targets(args.input, media_list)

    if args.limit is not None:
        cip_targets = cip_targets[: args.limit]
        dsmz_targets = dsmz_targets[: args.limit]
        name_targets = name_targets[: args.limit]

    log.info(
        "CIP catalogue targets: %d, MediaDive (by link) targets: %d, "
        "MediaDive (by name) targets: %d",
        len(cip_targets), len(dsmz_targets), len(name_targets),
    )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    results: list[RecipeResult] = []

    work = [("cip", number, link, "") for number, link in cip_targets]
    work += [("mediadive", medium_id, link, "") for medium_id, link in dsmz_targets]
    work += [("mediadive_by_name", medium_id, "", query_name)
             for query_name, medium_id in name_targets]

    for i, (source, key, link, query_name) in enumerate(
        tqdm(work, desc="Fetching recipes", unit="medium")
    ):
        if source == "cip":
            result = fetch_cip(key, link, args.cache_dir, args.force)
        else:
            result = fetch_mediadive(
                key, link, args.cache_dir, args.force,
                source=source, query_name=query_name,
            )
        results.append(result)
        if i < len(work) - 1:
            time.sleep(args.delay)

    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    ok = sum(r.status == "ok" for r in results)
    empty = sum(r.status == "empty" for r in results)
    not_found = sum(r.status == "not_found" for r in results)
    failed = sum(r.status in ("http_error", "error") for r in results)

    print()
    print("=" * 60)
    print("Media recipe fetch summary")
    print("=" * 60)
    print(f"CIP catalogue targets     : {len(cip_targets)}")
    print(f"MediaDive (by link) targets: {len(dsmz_targets)}")
    print(f"MediaDive (by name) targets: {len(name_targets)}")
    print(f"resolved OK               : {ok}")
    print(f"reachable but no recipe   : {empty}")
    print(f"not found                 : {not_found}")
    print(f"failed (network/parse)    : {failed}")
    print(f"Manifest written to       : {args.manifest}")
    print(f"Cache dir                 : {args.cache_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
