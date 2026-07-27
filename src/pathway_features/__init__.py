"""
pathway_features package
=========================
Internal package backing scripts/prepare_modular_manifest.py,
scripts/build_modular_pathway_features.py, and
scripts/build_modular_gene_pathway_matrix.py -- the modular metabolic-
pathway feature pipeline: Pyrodigal (gene calling) + PyKofamSearch (KO
annotation) + EBI's give_completeness (module completeness) + CheckM2
(genome quality confounder). Split into small, single-purpose modules:

    config.py                    paths and tunable constants
    manifest.py                    strain <-> accession <-> FASTA-path table
    drac_credentials.py              resolves your Slurm billing account
                                       (no lab hardcoded)
    strain_names.py                    joins any accession-indexed table to
                                         canonical strain names
    checkm2_quality.py                   parses CheckM2's quality_report.tsv
                                           (required confounder feature)
    modular_ko.py                          loads PyKofamSearch's ko_hits.tsv +
                                             Pyrodigal's per-genome gene universe
    modular_completeness.py                  runs give_completeness locally and
                                               reshapes its sparse output into
                                               the canonical completeness matrix
                                               (THIS is the canonical modeling
                                               input, written directly)
    modular_kegg_modules.py                    KO -> KEGG-module-ID mapping,
                                                 sourced from give_completeness's
                                                 OWN bundled reference data
    modular_gene_pathway_membership.py           per-strain gene x KEGG-module-ID
                                                   binary membership matrix --
                                                   additive output, independent
                                                   of the completeness matrix

Gene calling and KO annotation run on a DRAC HPC cluster, driven by
scripts/hpc/ -- nothing in this package invokes them itself.
give_completeness and CheckM2's quality_report.tsv parsing run locally. See
docs/pathway_features.md for the full two-environment workflow.
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
