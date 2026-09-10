"""
config.py
=========
Paths and tunable constants for the BacDive growth pipeline. Every other
module in ``src.bacdive`` imports its paths and magic numbers from here so
they live in exactly one place.

The pipeline downloads a taxonomy-scoped subset of BacDive (DSMZ) and emits a
tidy ``(strain x culture medium)`` table with a binary ``growth_class`` target
that plugs straight into the existing modeling pipeline (see
``docs/modeling_pipeline.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# --------------------------------------------------------------------------
# PROJECT LAYOUT
# --------------------------------------------------------------------------

# config.py -> src/bacdive/ -> src/ -> <project root>
ROOT = Path(__file__).resolve().parents[2]

# --- inputs ---
# User-editable taxonomic scope (list of genera / species). Lives under the
# top-level ``bacdive/`` directory so it is easy to find and edit.
TAXA_YAML = ROOT / "bacdive" / "taxa.yaml"
# Cache-only rebuild config (target + output feature subsets; no API).
BUILD_FROM_CACHE_YAML = ROOT / "bacdive" / "build_from_cache.yaml"

# --- outputs (data/external/) ---
EXTERNAL_DIR = ROOT / "data" / "external"
CACHE_DIR = EXTERNAL_DIR / "bacdive_cache"          # one <bacdive_id>.json per record
FAILURES_CSV = EXTERNAL_DIR / "bacdive_failures.csv"  # taxa that returned nothing / errored
# External recipes recovered by scripts/bacdive/fetch_media_recipes.py for
# entries BacDive's own record left composition_raw blank on (see
# pipeline._backfill_composition). Missing is fine -- backfill is skipped.
MEDIA_RECIPE_MANIFEST = EXTERNAL_DIR / "media_recipe_manifest.csv"

# --- logs ---
LOGS_DIR = ROOT / "logs"

# --------------------------------------------------------------------------
# API BEHAVIOR TUNABLES
# --------------------------------------------------------------------------

# Pause between paged API calls (seconds). The public v2 API is
# unauthenticated; keep requests gentle to stay a good citizen.
REQUEST_DELAY = 0.2

# BacDive-IDs per '/v2/fetch/<ids>' call. Matches the API's paging size.
FETCH_BATCH_SIZE = 100

# Concurrent threads used to fetch batches within one taxon.
MAX_WORKERS = 10

# Concurrent taxa processed at once (each with its own BacdiveClient/session).
# Only kicks in when >1 taxon is listed and --limit is not set.
TAXA_WORKERS = 6

# --------------------------------------------------------------------------
# OUTPUT SCHEMAS (per target)
# --------------------------------------------------------------------------

# Strain feature columns
_STRAIN_FEATURES = [
    "genus",
    "family",
    "order",
    "phylum",
    "gram_stain",
    "cell_shape",
    "motility",
    "oxygen_tolerance",
    "nutrition_type",
    "spore_formation",
    "gc_content",
    "genome_accession",
    # Registry genome_accession came from: insdc | bvbrc | img.
    # genome_accession is no longer INSDC-only, so consumers that feed it to
    # NCBI (scripts/download_bacdive_genomes.py) must filter on this.
    "genome_accession_source",
    "genome_link",
    "n16s_accession",
    # Strain identity -- handles for resolving a genome outside BacDive
    "strain_designation",
    "is_type_strain",
    "dsm_number",
    "culture_collection_ids",
    "ncbi_taxid",
    "ncbi_taxid_level",
    "temperature",
    "ph",
]

# Strain categoricals
_STRAIN_CATEGORICALS = [
    "genus",
    "family",
    "order",
    "phylum",
    "gram_stain",
    "cell_shape",
    "motility",
    "oxygen_tolerance",
    "nutrition_type",
    "spore_formation",
]

# Always-kept identifiers
_MEDIUM_ID_COLS = [
    "bacdive_id",
    "strain_name",
    "strain_name_raw",
    # Three medium-name columns, each with one job:
    #   media_name        BacDive's original
    #   media_name_full   BacDive's original, more descriptive spelling
    #   media_canon       variant canonicalised 
    "media_name",
    "media_name_full",
    "media_canon",
    # ok | ambiguous (name lists several media) | unspecified (no name at all)
    "media_canon_status",
    "mediadive_id",
    "medium_link",
    "cip_medium_number",
    "cip_medium_link",
    # provenance of growth_class -- see extract.record_to_medium_rows
    "growth_source",
    "growth_override",
    "medium_growth_class",
    "temp_growth_class",
    "temp_join",
    # target
    "growth_class",
]
_METABOLITE_ID_COLS = [
    "bacdive_id",
    "strain_name",
    "strain_name_raw",
    "substrate_name",
    "chebi_id",
    "kind_of_utilization",
    "growth_class",
]

# Optional medium columns
_MEDIUM_FEATURES = [
    "n_ingredients",
    "has_yeast_extract",
    "has_peptone",
    "has_agar",
    "has_blood_components",
    "has_nacl",
    "complexity",
    "composition_raw",
    # provenance of composition_raw: record (BacDive's own entry) |
    # cip_catalogue | mediadive (external backfill) | name_match (borrowed
    # from another entry sharing this media_name) | "" (still unresolved)
    "composition_source",
    "refs",
    "reference",
]

# Optional metabolite columns
_METABOLITE_FEATURES = [
    "refs",
    "reference",
]

# --- medium_growth: one row = (strain, culture medium) ---
_MEDIUM_GROWTH_FIELDNAMES = [
    *_MEDIUM_ID_COLS,
    *_STRAIN_FEATURES,
    *_MEDIUM_FEATURES,
]

# --- metabolite_utilization: one row = (strain, metabolite, assay kind) ---
_METABOLITE_FIELDNAMES = [
    *_METABOLITE_ID_COLS,
    *_STRAIN_FEATURES,
    *_METABOLITE_FEATURES,
]

# Columns for the failures.csv diagnostic report.
FAILURE_FIELDNAMES = [
    "taxon",
    "reason",
]

# Cache-read failures (build_from_cache.py).
CACHE_FAILURE_FIELDNAMES = [
    "cache_file",
    "reason",
]


# --------------------------------------------------------------------------
# TARGET REGISTRY
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetSpec:
    """Everything the pipeline needs to build one target's output table.

    ``isolate_col`` / ``id_col`` / ``name_col`` / ``extra_key_cols`` define the
    de-duplication key: rows sharing the same
    ``(isolate, unit, extra...)`` are collapsed. By default the positive
    ``growth_class`` wins on conflict; ``dedup_w_negatives=True`` keeps both
    labels instead. ``unit_col`` is the column counted as "unique units" in
    the run summary.
    """

    name: str
    output_file: Path
    fieldnames: list[str]
    categorical_features: list[str]
    id_col: str                         # preferred stable id for the unit key
    name_col: str                       # fallback name for the unit key
    isolate_col: str = "strain_name"    # isolate identity in the dedup key
    extra_key_cols: list[str] = field(default_factory=list)
    positive_only: bool = False         # if True, warn about PU when negatives ~0

    @property
    def unit_col(self) -> str:
        return self.name_col


TARGETS: dict[str, TargetSpec] = {
    "medium_growth": TargetSpec(
        name="medium_growth",
        # This is the FULL schema's own name -- both the live-API path
        # (get_bacdive_dataset.py, always full schema, no --out) and
        # build_bacdive_from_cache.py's --full-schema fall back to it. A
        # cache-only build using a YAML-restricted column subset instead
        # (e.g. bacdive/build_from_cache.yaml) always sets its OWN
        # output_file explicitly, so it never resolves to this default --
        # the two schemas can't collide on one name.
        output_file=EXTERNAL_DIR / "bacdive_growth_full.parquet",
        fieldnames=_MEDIUM_GROWTH_FIELDNAMES,
        categorical_features=_STRAIN_CATEGORICALS,
        # grain: one row per (bacdive_id, media_name, temperature, oxygen_tolerance)
        isolate_col="bacdive_id",
        id_col="media_name",
        name_col="media_name",
        extra_key_cols=["temperature", "oxygen_tolerance"],
        positive_only=True,
    ),
    "metabolite_utilization": TargetSpec(
        name="metabolite_utilization",
        output_file=EXTERNAL_DIR / "bacdive_metabolite.parquet",
        fieldnames=_METABOLITE_FIELDNAMES,
        categorical_features=[*_STRAIN_CATEGORICALS, "kind_of_utilization"],
        id_col="chebi_id",
        name_col="substrate_name",
        extra_key_cols=["kind_of_utilization"],
        positive_only=False,
    ),
}

def stamp_temperature_expansion(path: Path, expanded: bool) -> Path:
    """Stamp a resolved ``temperature_range_expansion`` onto an output path.

    This flag changes row *count*, not just which feature columns are kept
    (a temperature range like "10-41" becomes one midpoint row, or three --
    min, max, mean), so two builds differing only in this flag must never be
    able to land on the same filename. The stamp is derived from the
    resolved config value actually passed to the pipeline, not from
    whatever path a caller happens to type -- a CLI ``--out`` override can
    no longer silently mask a mismatched setting, the way ``--out
    bacdive_growth_api_107k.parquet`` combined with the YAML's
    ``temperature_range_expansion: true`` overwrote that (unexpanded) file
    with expanded (132,821-row) data on 2026-08-24.

    Symmetric and idempotent regardless of what ``path`` already carries:
    ``expanded=True`` ensures the stem ends in ``_temp`` (adding it only
    if missing); ``expanded=False`` ensures it does NOT (stripping it if
    present) -- so a YAML ``output_file`` written with the suffix already
    baked in (e.g. because that YAML's own default is expanded) still
    resolves correctly if a caller overrides to unexpanded, rather than
    keeping a stale "_temp" that no longer describes the data.
    """
    has_suffix = path.stem.endswith("_temp")
    if expanded == has_suffix:
        return path
    if expanded:
        return path.with_name(f"{path.stem}_temp{path.suffix}")
    return path.with_name(f"{path.stem[: -len('_temp')]}{path.suffix}")


def stamp_dedup_w_negatives(path: Path, dedup_w_negatives: bool) -> Path:
    """Stamp a resolved ``dedup_w_negatives`` onto an output path.

    Same rationale and symmetric/idempotent behavior as
    ``stamp_temperature_expansion``: this flag also changes row count (a
    conflicting isolate/unit pair becomes one collapsed row, or two), so two
    builds differing only in this flag must never land on the same filename.
    ``dedup_w_negatives=True`` ensures the stem ends in ``_negs`` (adding it
    only if missing); ``False`` ensures it does not (stripping it if
    present).
    """
    has_suffix = path.stem.endswith("_negs")
    if dedup_w_negatives == has_suffix:
        return path
    if dedup_w_negatives:
        return path.with_name(f"{path.stem}_negs{path.suffix}")
    return path.with_name(f"{path.stem[: -len('_negs')]}{path.suffix}")


DEFAULT_TARGET = "medium_growth"
# Back-compat aliases
OUTPUT_FILE = TARGETS["medium_growth"].output_file
OUTPUT_FIELDNAMES = TARGETS["medium_growth"].fieldnames
CATEGORICAL_FEATURES = TARGETS["medium_growth"].categorical_features


# --------------------------------------------------------------------------
# taxa.yaml loaders
# --------------------------------------------------------------------------

def _load_yaml(path: Path | None) -> dict | list:
    path = path or TAXA_YAML
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_taxa(path: Path | None = None) -> list[str]:
    """Read the taxonomic scope from ``taxa.yaml``.

    The file is expected to contain a top-level ``taxa`` list of strings, each
    a genus, ``"Genus species"``, or ``"Genus species subspecies"`` query
    understood by ``BacdiveClient.search(taxonomy=...)``. A bare YAML list is
    also accepted.

    Returns a de-duplicated, order-preserving list of non-empty taxon strings.
    """
    data = _load_yaml(path)

    if isinstance(data, dict):
        raw = data.get("taxa", [])
    elif isinstance(data, list):
        raw = data
    else:
        raw = []

    seen: set[str] = set()
    taxa: list[str] = []
    for item in raw:
        taxon = str(item).strip()
        if taxon and taxon not in seen:
            seen.add(taxon)
            taxa.append(taxon)
    return taxa


def load_target(path: Path | None = None) -> str:
    """Read the ``target`` key from ``taxa.yaml`` (default ``medium_growth``).

    Raises ``ValueError`` if the configured target is not a known target.
    """
    data = _load_yaml(path)
    target = DEFAULT_TARGET
    if isinstance(data, dict) and data.get("target"):
        target = str(data["target"]).strip()
    if target not in TARGETS:
        raise ValueError(
            f"unknown target {target!r}; choose one of {sorted(TARGETS)}"
        )
    return target


# --------------------------------------------------------------------------
# build_from_cache.yaml loader
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BuildFromCacheConfig:
    """Resolved cache-only rebuild settings from ``build_from_cache.yaml``."""

    target: str
    fieldnames: list[str]
    categorical_features: list[str]
    cache_dir: Path
    output_file: Path
    output_writer: str = "to_parquet"
    temperature_range_expansion: bool = True
    dedup_w_negatives: bool = False
    strain_metadata: Path | None = None

    def to_target_spec(self) -> TargetSpec:
        """``TargetSpec`` with the selected output columns."""
        base = TARGETS[self.target]
        return TargetSpec(
            name=base.name,
            output_file=self.output_file,
            fieldnames=list(self.fieldnames),
            categorical_features=list(self.categorical_features),
            id_col=base.id_col,
            name_col=base.name_col,
            isolate_col=base.isolate_col,
            extra_key_cols=list(base.extra_key_cols),
            positive_only=base.positive_only,
        )


def _as_bool(raw: object, key: str, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key!r} must be a boolean, got {raw!r}")


def _as_str_list(raw: object, key: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{key!r} must be a list of strings")
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        name = str(item).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _validate_subset(selected: list[str], allowed: list[str], key: str) -> list[str]:
    allowed_set = set(allowed)
    unknown = [c for c in selected if c not in allowed_set]
    if unknown:
        raise ValueError(
            f"{key} has unknown column(s) {unknown}; "
            f"allowed: {allowed}"
        )
    return selected


def load_build_from_cache_config(path: Path | None = None) -> BuildFromCacheConfig:
    """Load ``build_from_cache.yaml`` and resolve the output column schema.

    Expected keys
    -------------
    target:
        ``medium_growth`` or ``metabolite_utilization``.
    strain_features:
        Subset of ``_STRAIN_FEATURES`` to keep. Omitting the key keeps all.
    medium_features:
        Subset of ``_MEDIUM_FEATURES`` (medium_growth only). Omitting keeps all.
    metabolite_features:
        Subset of ``_METABOLITE_FEATURES`` (metabolite only). Omitting keeps all.
    cache_dir / output_file:
        Optional path overrides (``out`` is accepted as an alias for
        ``output_file``).
    temperature_range_expansion:
        If true, a culture-temp range (``"30-37"``) emits three rows
        (min, max, mean) instead of one midpoint. Default false.
    dedup_w_negatives:
        If true, a (isolate, unit) pair with both growth labels keeps both
        rows instead of resolving the conflict to positive. Default false.
    strain_metadata:
        Optional CSV of per-strain metadata to left-merge onto the output on
        ``strain_name`` (left) == ``strain`` (right). Columns that already
        exist in the built table are kept from the left and dropped from the
        CSV. Omit or leave blank to skip the merge.
    """
    path = path or BUILD_FROM_CACHE_YAML
    data = _load_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a YAML mapping")

    target = str(data.get("target") or DEFAULT_TARGET).strip()
    if target not in TARGETS:
        raise ValueError(
            f"unknown target {target!r}; choose one of {sorted(TARGETS)}"
        )

    if "strain_features" in data:
        strain_feats = _validate_subset(
            _as_str_list(data.get("strain_features"), "strain_features"),
            _STRAIN_FEATURES,
            "strain_features",
        )
    else:
        strain_feats = list(_STRAIN_FEATURES)

    if target == "medium_growth":
        if "medium_features" in data:
            unit_feats = _validate_subset(
                _as_str_list(data.get("medium_features"), "medium_features"),
                _MEDIUM_FEATURES,
                "medium_features",
            )
        elif "medium_growth_features" in data:
            # alias used in early drafts of build_from_cache.yaml
            unit_feats = _validate_subset(
                _as_str_list(
                    data.get("medium_growth_features"), "medium_growth_features"
                ),
                _MEDIUM_FEATURES,
                "medium_growth_features",
            )
        else:
            unit_feats = list(_MEDIUM_FEATURES)
        fieldnames = [*_MEDIUM_ID_COLS, *strain_feats, *unit_feats]
    else:
        if "metabolite_features" in data:
            unit_feats = _validate_subset(
                _as_str_list(data.get("metabolite_features"), "metabolite_features"),
                _METABOLITE_FEATURES,
                "metabolite_features",
            )
        else:
            unit_feats = list(_METABOLITE_FEATURES)
        fieldnames = [*_METABOLITE_ID_COLS, *strain_feats, *unit_feats]

    categoricals = [c for c in fieldnames if c in _STRAIN_CATEGORICALS]
    if "kind_of_utilization" in fieldnames:
        categoricals.append("kind_of_utilization")

    cache_dir = Path(data["cache_dir"]) if data.get("cache_dir") else CACHE_DIR
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir

    out_raw = data.get("output_file") or data.get("out")
    output_file = Path(out_raw) if out_raw else TARGETS[target].output_file
    if not output_file.is_absolute():
        output_file = ROOT / output_file

    writer = str(data.get("output_writer") or "to_parquet").strip()
    expand = _as_bool(
        data.get("temperature_range_expansion"),
        "temperature_range_expansion",
        default=False,
    )
    dedup_w_negatives = _as_bool(
        data.get("dedup_w_negatives"),
        "dedup_w_negatives",
        default=False,
    )
    meta_raw = data.get("strain_metadata")
    strain_metadata = None
    if meta_raw not in (None, "", False):
        strain_metadata = Path(str(meta_raw).strip())
        if not strain_metadata.is_absolute():
            strain_metadata = ROOT / strain_metadata

    return BuildFromCacheConfig(
        target=target,
        fieldnames=fieldnames,
        categorical_features=categoricals,
        cache_dir=cache_dir,
        output_file=output_file,
        output_writer=writer,
        temperature_range_expansion=expand,
        dedup_w_negatives=dedup_w_negatives,
        strain_metadata=strain_metadata,
    )
