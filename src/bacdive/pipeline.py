"""
pipeline.py
===========
Runs the BacDive pipeline end to end for a chosen extraction target:

  1. read the taxonomic scope + target (taxa.yaml),
  2. fetch every matching record per taxon (cached; not yet cached records
     are fetched in parallel batches -- see ``client.BacDiveFetcher``),
  3. flatten records into rows for the target
     (``(strain x medium)`` or ``(strain x metabolite x assay)``),
  4. de-duplicate rows across publications (same strain + unit),
  5. write the target dataset + ``bacdive_failures.csv`` and a summary.

When more than one taxon is listed (and ``limit`` is not set), taxa are also
processed concurrently (``taxa_workers``): each worker gets its own
HTTP session.

``run_from_cache`` skips steps 1-2 and rebuilds the table from
``data/external/bacdive_cache/*.json`` only (no API).

Takes plain arguments so it can be driven from the CLI, a notebook, or a test.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.bacdive import config
from src.bacdive.config import TargetSpec
from src.bacdive.extract import record_to_rows
from src.bacdive.media import looks_like_recipe, media_features

log = logging.getLogger("bacdive.pipeline")


def _blank(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "")


def _load_recipe_manifest(path: Path) -> pd.DataFrame:
    """Successfully-resolved rows of the external recipe manifest.

    Built by ``scripts/bacdive/fetch_media_recipes.py`` from the CIP/Pasteur
    catalogue and MediaDive (by link or by exact name match); see that
    script's docstring. Returns an empty frame (not an error) when the
    manifest hasn't been generated yet.
    """
    cols = ["source", "link", "query_name", "composition_raw"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    manifest = pd.read_csv(path, dtype=str, keep_default_na=False)
    return manifest[manifest["status"] == "ok"][cols]


def _apply_composition(
    df: pd.DataFrame, mask: pd.Series, composition: pd.Series, source: str,
) -> None:
    """Fill ``composition_raw``/``composition_source`` and recompute the
    derived media features (``n_ingredients``, ``has_*``, ``complexity``)
    for the rows in ``mask``, in place.

    Only the composition-text-derived columns are touched -- ``mediadive_id``
    stays whatever this row's own ``medium_link`` produced, never borrowed
    from wherever the backfilled text came from.
    """
    for idx in df.index[mask]:
        feats = media_features(
            name=df.at[idx, "media_name_full"] or df.at[idx, "media_name"],
            link=df.at[idx, "medium_link"],
            composition=composition.loc[idx],
        )
        for col in ("n_ingredients", "has_yeast_extract", "has_peptone",
                    "has_agar", "has_blood_components", "has_nacl",
                    "complexity", "composition_raw"):
            df.at[idx, col] = feats[col]
        df.at[idx, "composition_source"] = source


def _backfill_composition(
    df: pd.DataFrame, manifest_path: Path = config.MEDIA_RECIPE_MANIFEST,
) -> pd.DataFrame:
    """Fill blank ``composition_raw`` from four sources BacDive's own record
    doesn't carry, most-authoritative first:

    1. The external recipe manifest, matched by the medium's own
       ``cip_medium_link`` / ``medium_link`` -- an authoritative recipe for
       *this exact* medium (CIP/Pasteur catalogue pages, or MediaDive
       records reached via the numeric id in a DSMZ PDF link).
    2. The same manifest, matched by ``media_name`` instead -- media with no
       link at all, resolved by an EXACT (not fuzzy) name match against
       MediaDive's catalogue; see fetch_media_recipes.py on why fuzzy
       matching is deliberately not attempted.
    3. The same manifest again, ``source=manual_review`` -- a human/LLM
       reasoning pass's confirmed choice among fuzzy-matched MediaDive
       candidates (prepare_media_name_review.py +
       apply_media_name_review.py), for names the exact-match step above
       couldn't resolve on its own.
    4. Another entry sharing the same ``media_name`` that does have a
       composition -- the same medium, reported by a different publication.
       Only text shaped like a real recipe (``media.looks_like_recipe``) is
       eligible, and the most common such text among the name's siblings is
       used, so a one-off modifier note (e.g. "supplemented with 2.5% NaCl")
       never gets picked over the medium's real recipe.

    Every row keeps a ``composition_source`` marking which of these (if any)
    supplied its text; rows BacDive's own record already answered are
    marked "record" and left untouched.
    """
    if "composition_raw" not in df.columns or df.empty:
        return df

    df = df.copy()
    comp_blank = _blank(df["composition_raw"])
    df["composition_source"] = np.where(comp_blank, "", "record")

    manifest = _load_recipe_manifest(manifest_path)
    if not manifest.empty:
        by_link = manifest[manifest["source"].isin(["cip", "mediadive"])]
        for link_col, source in (("cip_medium_link", "cip_catalogue"),
                                  ("medium_link", "mediadive")):
            if link_col not in df.columns:
                continue
            comp_blank = _blank(df["composition_raw"])
            recipe_by_link = by_link.set_index("link")["composition_raw"]
            matched_link = df[link_col].isin(recipe_by_link.index)
            mask = comp_blank & matched_link
            if mask.any():
                composition = df.loc[mask, link_col].map(recipe_by_link)
                _apply_composition(df, mask, composition, source)

        # mediadive_by_name: an exact (non-fuzzy) name match, made automatically
        # by fetch_media_recipes.py. manual_review: a human/LLM-reasoned match
        # from prepare_media_name_review.py's candidate worksheet, applied via
        # apply_media_name_review.py. Both key off media_name the same way;
        # kept as separate composition_source values for provenance.
        for source in ("mediadive_by_name", "manual_review"):
            by_name = manifest[manifest["source"] == source]
            if by_name.empty or "media_name" not in df.columns:
                continue
            comp_blank = _blank(df["composition_raw"])
            recipe_by_name = by_name.set_index("query_name")["composition_raw"]
            matched_name = df["media_name"].astype(str).str.strip().isin(recipe_by_name.index)
            mask = comp_blank & matched_name
            if mask.any():
                composition = df.loc[mask, "media_name"].astype(str).str.strip().map(recipe_by_name)
                _apply_composition(df, mask, composition, source)

    comp_blank = _blank(df["composition_raw"])
    if comp_blank.any():
        name_norm = df["media_name"].astype(str).str.strip().str.lower()
        recipe_rows = df.loc[~comp_blank, "composition_raw"]
        recipe_rows = recipe_rows[recipe_rows.map(looks_like_recipe)]
        if not recipe_rows.empty:
            # most common recipe-shaped text per name -- not just the first
            # -- so an outlier variant doesn't win over the usual recipe.
            canonical = (
                recipe_rows.groupby(name_norm.loc[recipe_rows.index])
                .agg(lambda s: s.value_counts().idxmax())
            )
            matched_name = name_norm.isin(canonical.index)
            mask = comp_blank & matched_name
            if mask.any():
                composition = name_norm.loc[mask].map(canonical)
                _apply_composition(df, mask, composition, "name_match")

    return df


@dataclass
class _TaxonResult:
    taxon: str
    rows: list[dict] = field(default_factory=list)
    n_records: int = 0
    ok: bool = False
    error: Optional[str] = None


def _unit_key(row: pd.Series, spec: TargetSpec) -> str:
    """Stable key for a row's de-dup unit: preferred id, else name, plus extras.

    e.g. medium -> media name (plus temperature, oxygen); metabolite -> ChEBI
    id or substrate name, combined with the assay kind. When ``id_col`` and
    ``name_col`` are the same (media_name), the name is lowercased so spelling
    variants of the same string collapse.
    """
    ident = str(row.get(spec.id_col, "") or "").strip()
    if ident and spec.id_col != spec.name_col:
        key = f"id:{ident}"
    else:
        key = "nm:" + str(row.get(spec.name_col, "") or "").strip().lower()
    for col in spec.extra_key_cols:
        key += "|" + str(row.get(col, "") or "").strip().lower()
    return key


def _dedupe(df: pd.DataFrame, spec: TargetSpec) -> pd.DataFrame:
    """Collapse duplicate (isolate, unit) rows coming from multiple refs.

    On a ``growth_class`` conflict the positive label wins (max); all other
    columns take the first non-null value. Conflicts are logged.
    """
    if df.empty:
        return df

    df = df.copy()
    df["_isolate_key"] = df[spec.isolate_col].astype(str).str.strip().str.lower()
    df["_unit_key"] = df.apply(lambda r: _unit_key(r, spec), axis=1)

    grp = df.groupby(["_isolate_key", "_unit_key"], sort=False)

    n_conflicts = int((grp["growth_class"].nunique() > 1).sum())
    if n_conflicts:
        log.warning(
            "%d (isolate, unit) pairs had conflicting labels -- resolved to positive.",
            n_conflicts,
        )

    agg = {col: "first" for col in df.columns if col not in
           {"growth_class", "_isolate_key", "_unit_key"}}
    agg["growth_class"] = "max"

    deduped = grp.agg(agg).reset_index(drop=True)
    return deduped.drop(columns=["_isolate_key", "_unit_key"], errors="ignore")


def _fetch_one_taxon(
    taxon: str,
    target: str,
    cache_dir: Path,
    request_delay: float,
    force: bool,
    show_progress: bool,
    max_workers: int,
    batch_size: int,
    limit: Optional[int] = None,
    n_records_so_far: int = 0,
) -> _TaxonResult:
    """Fetch + flatten every record for one taxon. Owns its own fetcher/session."""
    # Imported lazily so cache-only builds do not require the bacdive PyPI client.
    from src.bacdive.client import BacDiveFetcher

    fetcher = BacDiveFetcher(
        cache_dir=cache_dir,
        request_delay=request_delay,
        force=force,
        show_progress=show_progress,
        max_workers=max_workers,
        batch_size=batch_size,
    )
    result = _TaxonResult(taxon=taxon)
    try:
        for record in fetcher.fetch_taxon(taxon):
            result.ok = True
            result.n_records += 1
            result.rows.extend(record_to_rows(record, target))
            if limit is not None and n_records_so_far + result.n_records >= limit:
                break
        if not result.ok:
            result.error = "no records returned"
    except LookupError as exc:
        log.warning("%s", exc)
        result.error = str(exc)
    except Exception as exc:  # noqa: BLE001 - keep going on any one taxon
        log.error("error processing taxon %r: %s", taxon, exc)
        result.error = f"error: {exc}"
    return result


def run(
    taxa: list[str],
    target: str = config.DEFAULT_TARGET,
    output_file: Optional[Path] = None,
    output_writer: str = "to_parquet",
    failures_csv: Path = config.FAILURES_CSV,
    cache_dir: Path = config.CACHE_DIR,
    request_delay: float = config.REQUEST_DELAY,
    limit: Optional[int] = None,
    force: bool = False,
    show_progress: bool = True,
    max_workers: int = config.MAX_WORKERS,
    batch_size: int = config.FETCH_BATCH_SIZE,
    taxa_workers: int = config.TAXA_WORKERS,
) -> tuple[int, int, int]:
    """Run the pipeline for a list of taxonomy queries and a target.

    Parameters
    ----------
    taxa:
        Taxonomy queries (see ``taxa.yaml``).
    target:
        One of ``config.TARGETS`` (``medium_growth`` or
        ``metabolite_utilization``).
    output_file:
        Output path; defaults to the target's spec path.
    output_writer:
        Pandas DataFrame export method name (e.g. ``"to_parquet"``,
        ``"to_csv"``).
    limit:
        Optional cap on the total number of *records* processed (for quick
        tests). ``None`` = no cap. Forces sequential (``taxa_workers``
        effectively 1).
    force:
        Re-fetch and overwrite cached records even if already on disk.
    show_progress:
        Show tqdm progress bars (one per taxon, plus one per-taxon for record
        fetching). Disable for non-interactive/log-only runs.
    max_workers:
        Threads used per taxon to fetch not yet cached record batches in
        parallel (network-bound; see ``client.BacDiveFetcher``).
    batch_size:
        BacDive-IDs per fetch call within a taxon.
    taxa_workers:
        Taxa processed concurrently, each with its own fetcher/session.
        Only takes effect when >1 taxon is listed and ``limit`` is unset.

    Returns ``(n_rows, n_taxa_ok, n_failures)``.
    """
    if target not in config.TARGETS:
        raise ValueError(
            f"unknown target {target!r}; choose one of {sorted(config.TARGETS)}"
        )
    spec = config.TARGETS[target]
    output_file = output_file or spec.output_file

    log.info("target: %s -> %s (%s)", target, output_file, output_writer)

    rows: list[dict] = []
    failures: list[dict] = []
    n_taxa_ok = 0
    n_records = 0

    # `limit` needs a precise, shared running count to stop promptly, so it
    # forces the simple sequential path. Otherwise, run taxa concurrently
    # when there's more than one (each taxon already parallelizes its own
    # record fetching, so single-taxon runs get no benefit from this level).
    run_parallel = limit is None and taxa_workers > 1 and len(taxa) > 1

    if not run_parallel:
        taxa_bar = tqdm(
            taxa, desc="Taxa", unit="taxon", position=0, disable=not show_progress,
        )
        for taxon in taxa_bar:
            taxa_bar.set_postfix(records=n_records, current=taxon[:20])
            if limit is not None and n_records >= limit:
                log.info("record limit (%d) reached -- stopping.", limit)
                break
            result = _fetch_one_taxon(
                taxon, target, cache_dir, request_delay, force, show_progress,
                max_workers, batch_size, limit=limit, n_records_so_far=n_records,
            )
            n_records += result.n_records
            if result.ok:
                n_taxa_ok += 1
                rows.extend(result.rows)
            else:
                failures.append({"taxon": taxon, "reason": result.error})
        taxa_bar.close()
    else:
        log.info(
            "processing %d taxa with up to %d concurrent worker(s)",
            len(taxa), taxa_workers,
        )
        # Disable each fetcher's inner per-record bar: multiple tqdm bars
        # writing from different threads at once would garble the terminal.
        taxa_bar = tqdm(
            total=len(taxa), desc="Taxa", unit="taxon", position=0,
            disable=not show_progress,
        )
        with cf.ThreadPoolExecutor(max_workers=taxa_workers) as pool:
            futures = {
                pool.submit(
                    _fetch_one_taxon, taxon, target, cache_dir, request_delay,
                    force, False, max_workers, batch_size,
                ): taxon
                for taxon in taxa
            }
            for future in cf.as_completed(futures):
                result = future.result()
                n_records += result.n_records
                if result.ok:
                    n_taxa_ok += 1
                    rows.extend(result.rows)
                else:
                    failures.append({"taxon": result.taxon, "reason": result.error})
                taxa_bar.update(1)
                taxa_bar.set_postfix(records=n_records, current=result.taxon[:20])
        taxa_bar.close()

    df = pd.DataFrame(rows)
    if not df.empty:
        df = _dedupe(df, spec)
        df = _backfill_composition(df)
        # enforce column order / presence
        for col in spec.fieldnames:
            if col not in df.columns:
                df[col] = ""
        df = df[spec.fieldnames]
        # keep categoricals NaN-free so downstream OneHotEncoder is safe
        for col in spec.categorical_features:
            df[col] = df[col].replace("", "unknown").fillna("unknown")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    getattr(df, output_writer)(output_file, index=False)
    log.info("wrote %d row(s) -> %s", len(df), output_file)

    _write_failures(failures, failures_csv)

    _log_summary(df, spec, n_records, n_taxa_ok, failures)
    return len(df), n_taxa_ok, len(failures)


def run_from_cache(
    target: str = config.DEFAULT_TARGET,
    fieldnames: Optional[list[str]] = None,
    categorical_features: Optional[list[str]] = None,
    output_file: Optional[Path] = None,
    output_writer: str = "to_parquet",
    cache_dir: Path = config.CACHE_DIR,
    failures_csv: Optional[Path] = None,
    limit: Optional[int] = None,
    show_progress: bool = True,
    temperature_range_expansion: bool = False,
) -> tuple[int, int, int]:
    """Build a target table from cached JSON records only (no API calls).

    Walks ``cache_dir/*.json``, flattens each record with ``record_to_rows``,
    de-duplicates, and writes the selected columns.

    ``fieldnames`` / ``categorical_features`` default to the full target schema
    in ``config.TARGETS``. Pass a subset (as resolved from
    ``build_from_cache.yaml``) to drop unused columns from the output.

    Returns ``(n_rows, n_records_ok, n_failures)``.
    """
    if target not in config.TARGETS:
        raise ValueError(
            f"unknown target {target!r}; choose one of {sorted(config.TARGETS)}"
        )
    base = config.TARGETS[target]
    spec = TargetSpec(
        name=base.name,
        output_file=output_file or base.output_file,
        fieldnames=list(fieldnames or base.fieldnames),
        categorical_features=list(
            categorical_features
            if categorical_features is not None
            else base.categorical_features
        ),
        id_col=base.id_col,
        name_col=base.name_col,
        isolate_col=base.isolate_col,
        extra_key_cols=list(base.extra_key_cols),
        positive_only=base.positive_only,
    )
    output_file = spec.output_file
    failures_csv = failures_csv or (
        config.EXTERNAL_DIR / "bacdive_cache_failures.csv"
    )

    if not cache_dir.is_dir():
        raise FileNotFoundError(f"cache dir not found: {cache_dir}")

    cache_files = sorted(cache_dir.glob("*.json"))
    if limit is not None:
        cache_files = cache_files[: max(0, limit)]

    log.info(
        "cache-only build: target=%s, %d file(s) in %s -> %s "
        "(temperature_range_expansion=%s)",
        target, len(cache_files), cache_dir, output_file,
        temperature_range_expansion,
    )

    rows: list[dict] = []
    failures: list[dict] = []
    n_records = 0

    for path in tqdm(
        cache_files, desc="Cache", unit="rec", disable=not show_progress,
    ):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append({"cache_file": path.name, "reason": str(exc)})
            continue
        if not isinstance(record, dict):
            failures.append(
                {"cache_file": path.name, "reason": "record is not a JSON object"}
            )
            continue
        n_records += 1
        rows.extend(
            record_to_rows(
                record, target,
                temperature_range_expansion=temperature_range_expansion,
            )
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = _dedupe(df, spec)
        df = _backfill_composition(df)
        for col in spec.fieldnames:
            if col not in df.columns:
                df[col] = ""
        df = df[spec.fieldnames]
        for col in spec.categorical_features:
            if col in df.columns:
                df[col] = df[col].replace("", "unknown").fillna("unknown")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    getattr(df, output_writer)(output_file, index=False)
    log.info("wrote %d row(s) -> %s", len(df), output_file)

    failures_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=config.CACHE_FAILURE_FIELDNAMES).to_csv(
        failures_csv, index=False
    )
    if failures:
        log.info("wrote %d cache failure(s) -> %s", len(failures), failures_csv)

    _log_summary(
        df, spec, n_records, n_records, failures, source="cache",
    )
    return len(df), n_records, len(failures)


def _write_failures(failures: list[dict], failures_csv: Path) -> None:
    failures_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=config.FAILURE_FIELDNAMES).to_csv(
        failures_csv, index=False
    )
    if failures:
        log.info("wrote %d failure(s) -> %s", len(failures), failures_csv)


def _log_summary(
    df: pd.DataFrame,
    spec: TargetSpec,
    n_records: int,
    n_taxa_ok: int,
    failures: list[dict],
    source: str = "api",
) -> None:
    if source == "cache":
        ok_line = f"cache files ok    : {n_taxa_ok}"
        fail_line = f"cache files bad   : {len(failures)}"
    else:
        ok_line = f"taxa with hits    : {n_taxa_ok}"
        fail_line = f"taxa failed       : {len(failures)}"
    lines = [
        "=== BacDive pipeline summary ===",
        f"target            : {spec.name}",
        f"records processed : {n_records}",
        ok_line,
        fail_line,
        f"datapoints (rows) : {len(df)}",
    ]
    pos = neg = 0
    if not df.empty:
        pos = int((df["growth_class"] == 1).sum())
        neg = int((df["growth_class"] == 0).sum())
        lines.append(f"unique strains    : {df['strain_name'].nunique()}")
        if spec.isolate_col in df.columns and spec.isolate_col != "strain_name":
            lines.append(
                f"unique {spec.isolate_col:<11}: {df[spec.isolate_col].nunique()}"
            )
        lines += [
            f"unique {spec.unit_col:<11}: {df[spec.unit_col].nunique()}",
            f"growth_class      : {pos} positive / {neg} negative",
        ]
    summary = "\n".join(lines)
    print(summary)
    log.info("run summary:\n%s", summary)

    # PU warning: positive-only labels can't be used as a headline metric.
    if spec.positive_only and pos and neg == 0:
        log.warning(
            "target '%s' is positive-only (%d positive / 0 negative). This is a "
            "positive-unlabeled (PU) signal: absence of a (strain, unit) pair is "
            "NOT a negative. Use it as weak supervision / pretraining, not a "
            "reported classification metric. See bacdive/README.md.",
            spec.name, pos,
        )