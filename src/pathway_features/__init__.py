"""
pathway_features package
=========================
Internal package backing scripts/prepare_dram_manifest.py,
scripts/build_pathway_features.py, and scripts/build_gene_pathway_matrix.py.
Split into small, single-purpose modules:

    config.py       paths and tunable constants
    manifest.py       strain <-> accession <-> FASTA-path table (pre-DRAM)
    distillate.py       parses DRAM's product.tsv into a tidy completeness matrix
                          (THIS is the canonical modeling input, written directly)
    matrix.py             thresholds completeness into a presence/absence matrix
                            (--emit-diagnostics only, not a modeling input)
    pca.py                   a single global PCA fit for EDA/visualization
                                (--emit-diagnostics only, NEVER a modeling input --
                                 fitting PCA outside cross-validation leaks
                                 test-strain information)
    validation.py              sanity checks used by both scripts
    kegg_modules.py              KO -> KEGG-module-ID mapping, sourced from
                                   DRAM's own reference sheets (multi-label)
    gene_pathway_membership.py     per-strain gene x KEGG-module-ID binary
                                     membership matrix, built from DRAM's raw
                                     annotations.tsv -- additive output,
                                     independent of the completeness matrix

The actual DRAM run happens on a DRAC HPC cluster, driven by scripts/hpc/ --
nothing in this package invokes DRAM itself. See docs/pathway_features.md
for the full two-environment workflow.
"""

from .manifest import build_manifest, write_manifest
from .distillate import load_product_tsv, join_strain_names
from .matrix import build_presence_matrix
from .pca import run_eda_pca
from .kegg_modules import load_ko_module_mapping, modules_in_completeness_matrix
from .gene_pathway_membership import load_annotations, genes_for_accession, build_gene_module_matrix

__all__ = [
    "build_manifest",
    "write_manifest",
    "load_product_tsv",
    "join_strain_names",
    "build_presence_matrix",
    "run_eda_pca",
    "load_ko_module_mapping",
    "modules_in_completeness_matrix",
    "load_annotations",
    "genes_for_accession",
    "build_gene_module_matrix",
]
