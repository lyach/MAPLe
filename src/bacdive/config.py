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

# --- outputs (data/external/) ---
EXTERNAL_DIR = ROOT / "data" / "external"
CACHE_DIR = EXTERNAL_DIR / "bacdive_cache"          # one <bacdive_id>.json per record
OUTPUT_CSV = EXTERNAL_DIR / "bacdive_growth.csv"    # tidy strain x medium rows
FAILURES_CSV = EXTERNAL_DIR / "bacdive_failures.csv"  # taxa that returned nothing / errored

# --- logs ---
LOGS_DIR = ROOT / "logs"

# --------------------------------------------------------------------------
# API BEHAVIOR TUNABLES
# --------------------------------------------------------------------------

# Polite pause between paged API calls (seconds). The public v2 API is
# unauthenticated; keep requests gentle to stay a good citizen.
REQUEST_DELAY = 0.2

# --------------------------------------------------------------------------
# OUTPUT SCHEMA
# --------------------------------------------------------------------------

# Column order for bacdive_growth.csv. One row = one (strain, culture medium).
OUTPUT_FIELDNAMES = [
    # identifiers
    "bacdive_id",
    "strain_name",
    "media_name",
    "mediadive_id",
    # target
    "growth_class",
    # strain features
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
    "n16s_accession",
    # condition features (strain-level)
    "temperature",
    "ph",
    # media features (parsed from composition text)
    "n_ingredients",
    "has_yeast_extract",
    "has_peptone",
    "has_agar",
    "has_blood_components",
    "has_nacl",
    "is_defined",
    # provenance
    "composition_raw",
    "refs",
]

# Columns for the failures.csv diagnostic report.
FAILURE_FIELDNAMES = [
    "taxon",
    "reason",
]

# Categorical feature columns whose missing values are filled with "unknown"
# before writing, so downstream OneHotEncoder never sees NaN.
CATEGORICAL_FEATURES = [
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


# --------------------------------------------------------------------------
# taxa.yaml loader
# --------------------------------------------------------------------------

def load_taxa(path: Path | None = None) -> list[str]:
    """Read the taxonomic scope from ``taxa.yaml``.

    The file is expected to contain a top-level ``taxa`` list of strings, each
    a genus, ``"Genus species"``, or ``"Genus species subspecies"`` query
    understood by ``BacdiveClient.search(taxonomy=...)``. A bare YAML list is
    also accepted.

    Returns a de-duplicated, order-preserving list of non-empty taxon strings.
    """
    path = path or TAXA_YAML
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

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
