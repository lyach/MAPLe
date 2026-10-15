"""Run the BacDive pipeline end to end for a chosen extraction target.

``run`` reads the taxonomic scope, fetches every matching record per taxon
(cached; see ``client.BacDiveFetcher``), flattens records into rows for the
target, de-duplicates across publications and writes the dataset plus
``bacdive_failures.csv``. With more than one taxon (and no ``limit``), taxa are
processed concurrently, each with its own HTTP session.

``run_from_cache`` skips fetching and rebuilds the table from
``data/external/bacdive_cache/*.json`` only (no API).
"""

from __future__ import annotations

import concurrent.futures as cf
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.bacdive import config
from src.bacdive.config import TargetSpec
from src.bacdive.extract import record_to_rows
from src.bacdive.media import looks_like_recipe, media_features


# ---------------------------------------------------------------------------
# Composition backfill
# ---------------------------------------------------------------------------

def _blank(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "")


def _load_recipe_manifest(path: Path) -> pd.DataFrame:
    """Return the successfully resolved rows of the external recipe manifest.

    Built by ``scripts/bacdive/fetch_media_recipes.py``; an empty frame when it
    has not been generated yet.
    """
    cols = ["source", "link", "query_name", "composition_raw"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    manifest = pd.read_csv(path, dtype=str, keep_default_na=False)
    ok = manifest[manifest["status"] == "ok"][cols]
    logger.info(
        "media recipe manifest {}: {} ok row(s) of {} ({})",
        path.name, len(ok), len(manifest),
        ok["source"].value_counts().to_dict(),
    )
    return ok


def _apply_composition(
    df: pd.DataFrame, mask: pd.Series, composition: pd.Series, source: str,
) -> None:
    """Fill ``composition_raw`` / ``composition_source`` and recompute the derived
    media features for the rows in *mask*, in place.

    ``mediadive_id`` is left as this row's own ``medium_link`` produced it.
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
    """Fill blank ``composition_raw`` from sources BacDive's record lacks.

    Sources, most authoritative first:

    1. The recipe manifest matched on ``cip_medium_link`` / ``medium_link``.
    2. The manifest matched on an exact ``media_name`` (``mediadive_by_name``;
       fuzzy matching is deliberately not attempted).
    3. The manifest's ``manual_review`` rows, for names step 2 could not resolve.
    4. Another entry with the same ``media_name`` that has a recipe-shaped
       composition (``media.looks_like_recipe``); the most common such text
       wins so a one-off modifier note never beats the real recipe.

    ``composition_source`` records which source supplied each row's text; rows
    BacDive already answered are marked "record" and left untouched.
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

        # Both sources key off media_name the same way; they stay separate
        # composition_source values for provenance
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
            # Most common recipe-shaped text per name, so an outlier variant
            # doesn't win over the usual recipe
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


# ---------------------------------------------------------------------------
# De-duplication and fetching
# ---------------------------------------------------------------------------

@dataclass
class _TaxonResult:
    taxon: str
    rows: list[dict] = field(default_factory=list)
    n_records: int = 0
    ok: bool = False
    error: str | None = None


def _unit_key(row: pd.Series, spec: TargetSpec) -> str:
    """Return a stable key for a row's de-dup unit: preferred id, else name, plus extras.

    When ``id_col`` and ``name_col`` are the same (``media_name``), the name is
    lowercased so case variants collapse.
    """
    ident = str(row.get(spec.id_col, "") or "").strip()
    if ident and spec.id_col != spec.name_col:
        key = f"id:{ident}"
    else:
        key = "nm:" + str(row.get(spec.name_col, "") or "").strip().lower()
    for col in spec.extra_key_cols:
        key += "|" + str(row.get(col, "") or "").strip().lower()
    return key


def _dedupe(
    df: pd.DataFrame,
    spec: TargetSpec,
    *,
    dedup_w_negatives: bool = False,
) -> pd.DataFrame:
    """Collapse duplicate (isolate, unit) rows coming from multiple refs.

    Same-label duplicates always collapse. On a ``growth_class`` conflict,
    positive wins by default (which deletes an explicit no-growth);
    *dedup_w_negatives* keeps both labels instead.
    """
    if df.empty:
        return df

    df = df.copy()
    df["_isolate_key"] = df[spec.isolate_col].astype(str).str.strip().str.lower()
    df["_unit_key"] = df.apply(lambda r: _unit_key(r, spec), axis=1)
    # Prefer culture-temp provenance among same-label rows
    if "growth_source" in df.columns:
        df = df.assign(
            _src_rank=(df["growth_source"] != "culture temp").astype(int)
        ).sort_values(
            ["growth_class", "_src_rank"], ascending=[False, True],
        ).drop(columns="_src_rank")

    unit_grp = df.groupby(["_isolate_key", "_unit_key"], sort=False)
    n_conflicts = int((unit_grp["growth_class"].nunique() > 1).sum())

    if dedup_w_negatives:
        if n_conflicts:
            logger.warning(
                "{} (isolate, unit) pairs had conflicting labels -- "
                "keeping both (dedup_w_negatives); same-label duplicates collapsed.",
                n_conflicts,
            )
        group_cols = ["_isolate_key", "_unit_key", "growth_class"]
        grp = df.groupby(group_cols, sort=False)
        agg = {col: "first" for col in df.columns if col not in group_cols}
        deduped = grp.agg(agg).reset_index()
        return deduped.drop(columns=["_isolate_key", "_unit_key"], errors="ignore")

    if n_conflicts:
        logger.warning(
            "{} (isolate, unit) pairs had conflicting labels -- resolved to positive.",
            n_conflicts,
        )

    agg = {col: "first" for col in df.columns if col not in
           {"growth_class", "_isolate_key", "_unit_key"}}
    agg["growth_class"] = "max"

    deduped = unit_grp.agg(agg).reset_index(drop=True)
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
    limit: int | None = None,
    n_records_so_far: int = 0,
) -> _TaxonResult:
    """Fetch and flatten every record for one taxon, with its own fetcher/session."""
    # Imported lazily so cache-only builds do not require the bacdive PyPI client
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
        logger.warning("{}", exc)
        result.error = str(exc)
    except Exception as exc:  # noqa: BLE001 -- keep going on any one taxon
        logger.error("error processing taxon {!r}: {}", taxon, exc)
        result.error = f"error: {exc}"
    return result


# ---------------------------------------------------------------------------
# Output and summary
# ---------------------------------------------------------------------------

def _merge_strain_metadata(
    df: pd.DataFrame,
    metadata_csv: Path,
    left_on: str = "strain_name",
    right_on: str = "strain",
) -> pd.DataFrame:
    """Left-merge a per-strain metadata CSV onto *df* (``left_on`` == ``right_on``).

    Every row of *df* is kept. Columns already in *df* are dropped from the CSV
    so the built table's values win, and the CSV is de-duplicated on
    ``right_on`` (first row wins) so rows are never multiplied. Unmatched rows
    and blank cells come through as ``""``, not NaN.
    """
    if not metadata_csv.is_absolute():
        metadata_csv = config.ROOT / metadata_csv
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"strain metadata CSV not found: {metadata_csv}")
    if left_on not in df.columns:
        raise KeyError(
            f"cannot merge strain metadata: {left_on!r} not in the built table"
        )

    meta = pd.read_csv(metadata_csv, dtype=str)
    if right_on not in meta.columns:
        raise KeyError(
            f"{metadata_csv} has no {right_on!r} column "
            f"(found: {list(meta.columns)})"
        )

    n_dupes = int(meta[right_on].duplicated().sum())
    if n_dupes:
        logger.warning(
            "strain metadata {} has {} duplicate {!r} value(s); keeping the first",
            metadata_csv, n_dupes, right_on,
        )
        meta = meta.drop_duplicates(subset=[right_on], keep="first")

    overlap = [c for c in meta.columns if c != right_on and c in df.columns]
    if overlap:
        logger.info(
            "strain metadata: dropping {} column(s) already in the table "
            "(left wins): {}",
            len(overlap), overlap,
        )
        meta = meta.drop(columns=overlap)

    added = [c for c in meta.columns if c != right_on]
    n_before = len(df)
    n_matched = int(df[left_on].isin(set(meta[right_on])).sum())
    merged = df.merge(
        meta, how="left", left_on=left_on, right_on=right_on, validate="m:1",
    ).drop(columns=[right_on])
    # Missing strains and blank cells merge in as NaN; fill with "" to match
    # how the rest of the table spells an empty value
    if added:
        merged[added] = merged[added].fillna("")
    logger.info(
        "strain metadata: merged {} on {}=={} -- {} column(s) added {}, "
        "{}/{} row(s) matched",
        metadata_csv.name, left_on, right_on, len(added), added,
        n_matched, n_before,
    )
    return merged


def _write_failures(failures: list[dict], failures_csv: Path) -> None:
    failures_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=config.FAILURE_FIELDNAMES).to_csv(
        failures_csv, index=False
    )
    if failures:
        logger.info("wrote {} failures -> {}", len(failures), failures_csv)


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
        if "growth_source" in df.columns:
            src = df["growth_source"].fillna("").astype(str).value_counts()
            lines.append(
                "growth_source     : "
                + ", ".join(f"{k or 'blank'} {int(v)}" for k, v in src.items())
            )
            if "growth_override" in df.columns:
                ov = pd.to_numeric(df["growth_override"], errors="coerce").fillna(0)
                n_ov = int((ov == 1).sum())
                n_temp = int((df["growth_source"] == "culture temp").sum())
                n_agreed = n_temp - n_ov
                lines.append(
                    f"growth_override   : {n_ov} temp != medium "
                    f"({n_agreed} culture-temp rows agree with the medium)"
                )
            if "temp_join" in df.columns:
                join = df["temp_join"].fillna("").astype(str).value_counts()
                lines.append(
                    "temp_join         : "
                    + ", ".join(f"{k or 'blank'} {int(v)}" for k, v in join.items())
                )
    summary = "\n".join(lines)
    logger.info("run summary:\n{}", summary)

    # Positive-only labels can't be used as a headline metric
    if spec.positive_only and pos and neg == 0:
        logger.warning(
            "target '{}' is positive-only ({} positive / 0 negative). This is a "
            "positive-unlabeled (PU) signal: absence of a (strain, unit) pair is "
            "NOT a negative. Use it as weak supervision / pretraining, not a "
            "reported classification metric. See bacdive/README.md.",
            spec.name, pos,
        )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run(
    taxa: list[str],
    target: str = config.DEFAULT_TARGET,
    output_file: Path | None = None,
    output_writer: str = "to_parquet",
    failures_csv: Path = config.FAILURES_CSV,
    cache_dir: Path = config.CACHE_DIR,
    request_delay: float = config.REQUEST_DELAY,
    limit: int | None = None,
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
        One of ``config.TARGETS``.
    output_file:
        Output path; ``None`` = the target's default path.
    output_writer:
        Pandas DataFrame export method name (e.g. ``"to_parquet"``).
    limit:
        Cap on the total number of records processed; ``None`` = no cap.
        Forces sequential processing.
    force:
        Re-fetch and overwrite cached records.
    show_progress:
        Show tqdm progress bars; disable for log-only runs.
    max_workers:
        Threads per taxon fetching uncached record batches.
    batch_size:
        BacDive-IDs per fetch call.
    taxa_workers:
        Taxa processed concurrently; only applies with more than one taxon
        and no ``limit``.

    Returns ``(n_rows, n_taxa_ok, n_failures)``.
    """
    if target not in config.TARGETS:
        raise ValueError(
            f"unknown target {target!r}; choose one of {sorted(config.TARGETS)}"
        )
    spec = config.TARGETS[target]
    output_file = output_file or spec.output_file

    logger.info("target: {} -> {} ({})", target, output_file, output_writer)

    rows: list[dict] = []
    failures: list[dict] = []
    n_taxa_ok = 0
    n_records = 0

    # ``limit`` needs a precise shared count to stop promptly, so it forces the
    # sequential path. Single-taxon runs gain nothing from taxon-level
    # concurrency (each taxon already parallelizes its own fetching).
    run_parallel = limit is None and taxa_workers > 1 and len(taxa) > 1

    if not run_parallel:
        taxa_bar = tqdm(
            taxa, desc="Taxa", unit="taxon", position=0, disable=not show_progress,
        )
        for taxon in taxa_bar:
            taxa_bar.set_postfix(records=n_records, current=taxon[:20])
            if limit is not None and n_records >= limit:
                logger.info("record limit reached: {}", limit)
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
        logger.info(
            "processing {} taxa with up to {} concurrent workers",
            len(taxa), taxa_workers,
        )
        # Inner per-record bars are disabled: several tqdm bars writing from
        # different threads would garble the terminal
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
        # Enforce column order and presence
        for col in spec.fieldnames:
            if col not in df.columns:
                df[col] = ""
        df = df[spec.fieldnames]
        # Keep categoricals NaN-free so downstream OneHotEncoder is safe
        for col in spec.categorical_features:
            df[col] = df[col].replace("", "unknown").fillna("unknown")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    getattr(df, output_writer)(output_file, index=False)
    logger.info("wrote {} rows -> {}", len(df), output_file)

    _write_failures(failures, failures_csv)

    _log_summary(df, spec, n_records, n_taxa_ok, failures)
    return len(df), n_taxa_ok, len(failures)


def run_from_cache(
    target: str = config.DEFAULT_TARGET,
    fieldnames: list[str] | None = None,
    categorical_features: list[str] | None = None,
    output_file: Path | None = None,
    output_writer: str = "to_parquet",
    cache_dir: Path = config.CACHE_DIR,
    failures_csv: Path | None = None,
    limit: int | None = None,
    show_progress: bool = True,
    temperature_range_expansion: bool = False,
    dedup_w_negatives: bool = False,
    strain_metadata: Path | None = None,
) -> tuple[int, int, int]:
    """Build a target table from cached JSON records only (no API calls).

    Parameters
    ----------
    fieldnames, categorical_features:
        Column subsets (as resolved from ``build_from_cache.yaml``); ``None`` =
        the full target schema in ``config.TARGETS``.
    limit:
        Cap on the number of cache files read; ``None`` = all.
    dedup_w_negatives:
        Keep both rows of a (isolate, unit) pair with conflicting labels.
    strain_metadata:
        Optional CSV left-merged on ``strain_name`` == ``strain``; see
        ``_merge_strain_metadata``.

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

    logger.info(
        "cache-only build: target={}, {} file(s) in {} -> {} "
        "(temperature_range_expansion={}, dedup_w_negatives={})",
        target, len(cache_files), cache_dir, output_file,
        temperature_range_expansion, dedup_w_negatives,
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
        df = _dedupe(df, spec, dedup_w_negatives=dedup_w_negatives)
        df = _backfill_composition(df)
        for col in spec.fieldnames:
            if col not in df.columns:
                df[col] = ""
        df = df[spec.fieldnames]
        for col in spec.categorical_features:
            if col in df.columns:
                df[col] = df[col].replace("", "unknown").fillna("unknown")
        if strain_metadata is not None:
            df = _merge_strain_metadata(df, strain_metadata)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    getattr(df, output_writer)(output_file, index=False)
    logger.info("wrote {} rows -> {}", len(df), output_file)

    failures_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=config.CACHE_FAILURE_FIELDNAMES).to_csv(
        failures_csv, index=False
    )
    if failures:
        logger.info("wrote {} cache failures -> {}", len(failures), failures_csv)

    _log_summary(
        df, spec, n_records, n_records, failures, source="cache",
    )
    return len(df), n_records, len(failures)
