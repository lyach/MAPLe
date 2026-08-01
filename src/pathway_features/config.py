"""
config.py
=========
Paths and tunable constants for the modular pathway-features pipeline.
Kept in one place, like strain_metadata/config.py, so every other module
imports from here rather than redefining its own copy of a path or a magic
number.

This pipeline has two halves that run in different environments:
  * LOCAL  -- prepare_modular_manifest.py (before) and
              build_modular_pathway_features.py / build_modular_gene_pathway_
              matrix.py (after) run in this repo.
  * REMOTE -- Pyrodigal, PyKofamSearch, and CheckM2 run on a DRAC HPC
              cluster, driven by the artifacts under scripts/hpc/.
              give_completeness runs locally (fast, database-free). See
              docs/modular_pathway_features.md.
"""

from __future__ import annotations

from pathlib import Path

# config.py -> pathway_features/ -> src/ -> <project root>
ROOT = Path(__file__).resolve().parents[2]

# --- inputs ---
# strains_metadata.csv's "strain" column is already the canonical name (it's
# the pipeline's own output, from get_strain_metadata.py), so it's the only
# join key needed to map a genome accession back to a strain name -- no
# separate name-overrides file required here.
STRAINS_METADATA_CSV = ROOT / "data" / "external" / "strains_metadata.csv"
FASTA_DIR = ROOT / "data" / "external" / "fasta"

# Human-readable module_id -> module_name lookup, written once alongside the
# per-strain gene x pathway matrices -- see build_modular_gene_pathway_matrix
# .py's docstring for why labels live here and not as matrix column headers.
MODULE_LOOKUP_CSV_NAME = "kegg_module_lookup.csv"

# --- DRAC credentials (gitignored -- see drac_credentials.py) ---
# No lab's Slurm billing account is hardcoded anywhere in this pipeline.
# Each user is prompted for their own the first time they run
# prepare_modular_manifest.py, same pattern as strain_metadata's NCBI email.
DRAC_CREDENTIALS_FILE = ROOT / ".drac_credentials.json"

# --- modular pipeline paths -------------------------------------------------
# Pyrodigal + pykofamsearch (PyHMMER-based KOfam search) + CheckM2 +
# EBI kegg-pathways-completeness-tool. See docs/modular_pathway_features.md.
MODULAR_INTERIM_DIR = ROOT / "data" / "interim" / "modular_pathway"
MODULAR_FASTA_MANIFEST_CSV = MODULAR_INTERIM_DIR / "fasta_manifest.csv"

# Raw per-genome KO hits from pykofamsearch, rsynced back from
# $MODULAR_ROOT/output/ko_annotation/ on the cluster (see
# scripts/hpc/run_modular_annotate.sbatch).
MODULAR_KO_ANNOTATION_DIR = ROOT / "data" / "external" / "modular_pathway" / "ko_annotation"

# CheckM2's quality_report.tsv, rsynced back from
# $MODULAR_ROOT/output/checkm2/ on the cluster -- the "required confounder
# feature" (per-genome completeness/contamination), independent of the
# KEGG-module completeness matrix below.
MODULAR_CHECKM2_DIR = ROOT / "data" / "external" / "modular_pathway" / "checkm2"

MODULAR_GENE_PATHWAY_OUTPUT_DIR = ROOT / "data" / "external" / "modular_pathway" / "gene_pathway_membership"

# Per-genome protein FASTAs from Pyrodigal (rsynced back from
# $MODULAR_ROOT/output/genes/ -- NOT the combined all_proteins.faa), needed
# for the full per-strain gene universe (including zero-hit genes) when
# building the gene x pathway membership matrix, since ko_hits.tsv only
# contains genes that DID hit a KO. See modular_ko.py.
MODULAR_GENES_DIR = ROOT / "data" / "external" / "modular_pathway" / "genes"

# CheckM2's required confounder feature (completeness/contamination, [0, 1]).
MODULAR_CHECKM2_QUALITY_CSV_NAME = "checkm2_quality.csv"

# CANONICAL modeling input: strain x KEGG-module, continuous completeness
# fractions (0.0-1.0), from the EBI kegg-pathways-completeness-tool (give_
# completeness). give_completeness reports real KEGG module IDs directly for
# every module it computes -- no free-text-name recovery step needed
# (contrast modular_kegg_modules.py, which parses give_completeness's own
# bundled reference data for the same reason). Column scope is whatever
# module set actually appears with nonzero completeness across this
# pipeline's own strain set, which the gene x pathway membership matrix
# reuses as its own column axis for internal consistency.
MODULAR_COMPLETENESS_CSV_NAME = "kegg_pathway_completeness.csv"
