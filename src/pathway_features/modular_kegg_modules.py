"""Builds the KO -> KEGG-module-ID mapping from give_completeness's own reference data."""

from __future__ import annotations

import csv
import logging
import re
from importlib import resources

log = logging.getLogger("pathway_features.modular_kegg_modules")

_KO_PATTERN = re.compile(r"K\d{5}")


def _extract_kos(definition: str) -> set[str]:
    """Pull KO IDs out of a modules_table.tsv boolean-logic definition string."""
    return set(_KO_PATTERN.findall(definition))


def load_ko_module_mapping() -> tuple[dict[str, list[str]], dict[str, str]]:
    """Returns (ko_to_modules, module_id_to_name) from modules_table.tsv."""
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
