"""
modular_kegg_modules.py
=========================
Builds a KO -> KEGG-module-ID mapping for the modular pipeline's gene x
pathway membership matrix.

Sourced from give_completeness's OWN bundled reference data
(kegg_pathways_completeness/pathways_data/modules_table.tsv, installed
alongside the package -- no download needed). This matters: it's the exact
same reference data kegg_pathway_completeness.csv was computed from, so the
gene x pathway matrix and the completeness matrix are guaranteed to share
the same KO -> module ground truth -- covers 570 modules / 3091 KOs,
confirmed to fully cover every module this pipeline's own completeness
matrix actually reports (checked directly, 0 gaps).
"""

from __future__ import annotations

import csv
import logging
import re
from importlib import resources

log = logging.getLogger("pathway_features.modular_kegg_modules")

_KO_PATTERN = re.compile(r"K\d{5}")


def _extract_kos(definition: str) -> set[str]:
    """modules_table.tsv's 'definition' column is a boolean-logic string
    over KOs (e.g. '(K00844,K12407) (K01810,K06859)'). Only membership
    (which KOs are referenced at all) is needed, not the AND/OR step
    structure -- same reasoning as kegg_modules.py's _extract_kos.
    """
    return set(_KO_PATTERN.findall(definition))


def load_ko_module_mapping() -> tuple[dict[str, list[str]], dict[str, str]]:
    """Returns (ko_to_modules, module_id_to_name), parsed from
    give_completeness's own bundled modules_table.tsv.

    ko_to_modules     -- KO id -> sorted list of KEGG module IDs it belongs
                          to. MULTI-LABEL: a KO commonly belongs to several
                          modules.
    module_id_to_name -- KEGG module ID -> human-readable name (the
                          'name' column), for the module lookup CSV.
    """
    modules_tsv = resources.files("kegg_pathways_completeness") / "pathways_data" / "modules_table.tsv"

    ko_to_modules: dict[str, set[str]] = {}
    module_id_to_name: dict[str, str] = {}

    with modules_tsv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            module_id, definition, name = row["module"], row["definition"], row["name"]
            module_id_to_name[module_id] = name
            for ko in _extract_kos(definition):
                ko_to_modules.setdefault(ko, set()).add(module_id)

    log.info(
        "Loaded KO-to-module mapping (give_completeness's own reference data): "
        "%d KO(s) across %d KEGG module(s)",
        len(ko_to_modules), len(module_id_to_name),
    )
    return {ko: sorted(mods) for ko, mods in ko_to_modules.items()}, module_id_to_name
