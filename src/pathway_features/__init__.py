"""Modular metabolic-pathway feature pipeline: Pyrodigal + PyKofamSearch +
give_completeness + CheckM2. See docs/modular_pathway_features.md.
"""

from .manifest import build_manifest, write_manifest
from .strain_names import join_strain_names
from .checkm2_quality import load_quality_report
from .modular_ko import load_ko_hits, kos_for_accession, load_gene_universe
from .modular_completeness import write_give_completeness_input, run_give_completeness, build_completeness_matrix
from .modular_kegg_modules import load_ko_module_mapping
from .modular_gene_pathway_membership import build_gene_module_matrix

__all__ = [
    "build_manifest",
    "write_manifest",
    "join_strain_names",
    "load_quality_report",
    "load_ko_hits",
    "kos_for_accession",
    "load_gene_universe",
    "write_give_completeness_input",
    "run_give_completeness",
    "build_completeness_matrix",
    "load_ko_module_mapping",
    "build_gene_module_matrix",
]
