"""
kegg_modules.py
================
Builds a KO -> KEGG-module-ID mapping (MULTI-LABEL: one KO commonly belongs
to several modules -- e.g. K00024 belongs to 9 different modules in DRAM's
own reference data, confirmed empirically below) for the gene x pathway
membership matrix.

Sourced from DRAM's own reference sheets, not an independently-fetched KEGG
mapping -- these are the exact files DRAM-bio downloads during
`DRAM-setup.py prepare_databases` and uses internally to compute
kegg_pathway_completeness.csv's percentages, so this mapping is guaranteed
consistent with that file's module universe:

    module_step_form.tsv     multi-step KEGG modules (glycolysis, TCA, ...)
    etc_module_database.tsv  electron-transport-chain complexes (Complex I-V)

IMPORTANT CAVEAT: DRAM's product.tsv (and kegg_pathway_completeness.csv) has
98 columns, but only 32 of them correspond to a real KEGG module ID covered
by the two sheets above. The other 66 (`CAZy: *`, `Nitrogen metabolism: *`,
`SCFA and alcohol conversions: *`, `Other Reductases: *`, `Photosynthesis: *`,
`Sulfur metabolism: *`, `Methanogenesis and methanotrophy: *`) are DRAM's own
hand-curated marker-gene categories sourced from a different sheet
(function_heatmap_form.tsv) that were never official KEGG modules with
M-numbers at all -- confirmed by checking product.tsv's actual column names
against this module's mapping: 32/98 matched, 66/98 did not, and none of the
66 appear anywhere in module_step_form.tsv or etc_module_database.tsv either.
The gene x pathway membership matrix this package builds only covers the 32
real-KEGG-module columns; it does not (and should not, without fabricating
IDs) cover the other 66.

Known minor edge case: module M00153 ("Cytochrome bd ubiquinol oxidase")
legitimately appears twice in etc_module_database.tsv, filed under both
"Complex III" and "Complex IV High affinity" (same KOs, same module, DRAM's
own curatorial choice, not a data error) -- meaning product.tsv likely has
two separate completeness columns backed by this one module ID. Since column
identity here is the bare module ID (per the task's "use KEGG module IDs"
requirement), this one module collapses to a single column in the gene x
pathway matrix rather than being split by complex context like
kegg_pathway_completeness.csv's columns are. Affects 1 of 32 modules.
"""

from __future__ import annotations

import csv
import logging
import re
import urllib.request
from pathlib import Path

from . import config

log = logging.getLogger("pathway_features.kegg_modules")

_KO_PATTERN = re.compile(r"K\d{5}")


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 -- fixed, hardcoded GitHub raw URLs, not user input


def _ensure_cached(refresh: bool) -> tuple[Path, Path]:
    """Return local paths to (module_step_form.tsv, etc_module_database.tsv),
    downloading into config.KEGG_MODULE_MAPPING_CACHE_DIR if not already
    cached (or if refresh=True)."""
    cache_dir = config.KEGG_MODULE_MAPPING_CACHE_DIR
    step_form = cache_dir / "module_step_form.tsv"
    etc_db = cache_dir / "etc_module_database.tsv"

    if refresh or not step_form.exists():
        _download(config.MODULE_STEP_FORM_URL, step_form)
    if refresh or not etc_db.exists():
        _download(config.ETC_MODULE_DATABASE_URL, etc_db)

    return step_form, etc_db


def _extract_kos(definition: str) -> set[str]:
    """etc_module_database.tsv's 'definition' column is a boolean-logic
    string over KOs, e.g. 'K00330+(K00331+K00332+K00333,...)'. We only need
    membership (which KOs are referenced at all), not the AND/OR structure --
    gene-level membership here is "this gene's KO is part of this complex's
    definition", not "this complex is complete", so a flat regex extraction
    of every K-number is sufficient and doesn't need a boolean-expression
    parser.
    """
    return set(_KO_PATTERN.findall(definition))


def load_ko_module_mapping(refresh: bool = False) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Returns (ko_to_modules, module_id_to_name).

    ko_to_modules     -- KO id (e.g. "K00844") -> sorted list of KEGG module
                          IDs it belongs to. MULTI-LABEL: do not assume one
                          KO maps to exactly one module.
    module_id_to_name -- KEGG module ID -> human-readable name, in the same
                          format DRAM's product.tsv displays it ("Complex I:
                          ..." for ETC complexes), for cross-referencing
                          against kegg_pathway_completeness.csv's column
                          names. This is a lookup table, not matrix column
                          headers -- see build_gene_pathway_matrix.py.
    """
    step_form_path, etc_db_path = _ensure_cached(refresh)

    ko_to_modules: dict[str, set[str]] = {}
    module_id_to_name: dict[str, str] = {}

    with step_form_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ko, module_id, module_name = row["ko"], row["module"], row["module_name"]
            if not ko or not module_id:
                continue
            ko_to_modules.setdefault(ko, set()).add(module_id)
            module_id_to_name[module_id] = module_name

    with etc_db_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            module_id = row["module_id"]
            module_id_to_name[module_id] = f"{row['complex']}: {row['module_name']}"
            for ko in _extract_kos(row["definition"]):
                ko_to_modules.setdefault(ko, set()).add(module_id)

    log.info(
        "Loaded KO-to-module mapping: %d KO(s) across %d real KEGG module(s)",
        len(ko_to_modules), len(module_id_to_name),
    )
    return {ko: sorted(mods) for ko, mods in ko_to_modules.items()}, module_id_to_name


def modules_in_completeness_matrix(
    product_tsv_columns: list[str],
    module_id_to_name: dict[str, str],
) -> dict[str, str]:
    """Restrict module_id_to_name down to just the module IDs that actually
    appear as a column in kegg_pathway_completeness.csv (matched by name,
    since that's what the completeness matrix's columns are keyed on).

    This is the deliberate scope boundary for the gene x pathway membership
    matrix: use ONLY the same module universe the completeness matrix uses
    (32 of its 98 columns, per this module's docstring), not the full ~399
    modules DRAM's reference sheets happen to define. Using the full set
    would make the two matrices cover different, non-corresponding module
    sets, defeating the point of keeping them joinable on the same axis.

    Returns a dict (module_id -> module_name) of just the matched subset,
    in the same shape as module_id_to_name so callers can use it the same
    way. Logs which completeness-matrix columns did NOT match a real module
    (expected: most of them -- see kegg_modules.py's module docstring).
    """
    name_to_id = {name: mid for mid, name in module_id_to_name.items()}
    matched_ids = {name_to_id[c] for c in product_tsv_columns if c in name_to_id}

    unmatched = [c for c in product_tsv_columns if c not in name_to_id]
    log.info(
        "%d/%d completeness-matrix columns correspond to a real KEGG module "
        "ID; %d are DRAM-curated categories with no official module ID "
        "(excluded from the gene x pathway matrix -- see module docstring)",
        len(matched_ids), len(product_tsv_columns), len(unmatched),
    )

    return {mid: module_id_to_name[mid] for mid in matched_ids}
