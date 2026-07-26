"""
config.py
=========
Paths and tunable constants for the pathway-features pipeline. Kept in one
place, like strain_metadata/config.py, so every other module imports from
here rather than redefining its own copy of a path or a magic number.

This pipeline has two halves that run in different environments:
  * LOCAL  -- prepare_dram_manifest.py (before) and build_pathway_features.py
              (after) run in this repo.
  * REMOTE -- DRAM itself runs on a DRAC HPC cluster, driven by the artifacts
              under scripts/hpc/. See docs/pathway_features.md.
"""

from __future__ import annotations

from pathlib import Path

# config.py -> pathway_features/ -> src/ -> <project root>
ROOT = Path(__file__).resolve().parents[2]

# --- inputs ---
# strains_metadata.csv's "strain" column is already the canonical name (it's
# the pipeline's own output, from get_strain_metadata.py), so it's the only
# join key needed to map a DRAM genome ID back to a strain name -- no
# separate name-overrides file required here.
STRAINS_METADATA_CSV = ROOT / "data" / "external" / "strains_metadata.csv"
FASTA_DIR = ROOT / "data" / "external" / "fasta"

# --- manifest (local, pre-DRAM) ---
INTERIM_DIR = ROOT / "data" / "interim" / "dram"
FASTA_MANIFEST_CSV = INTERIM_DIR / "fasta_manifest.csv"

# --- gene x pathway membership (local, post-DRAM) ---
# The raw per-gene DRAM annotation output (annotations.tsv, genes.gff, ...)
# -- NOT the distillate/ genome-level summary. Must be rsynced back from
# $DRAM_ROOT/output/annotation/ on the cluster separately; the distillate/
# rsync alone doesn't include it.
DRAM_ANNOTATION_DIR = ROOT / "data" / "external" / "dram" / "annotation"

GENE_PATHWAY_OUTPUT_DIR = ROOT / "data" / "external" / "dram" / "gene_pathway_membership"

# Human-readable module_id -> module_name lookup, written once alongside the
# per-strain matrices -- see build_gene_pathway_matrix.py's docstring for why
# labels live here and not as matrix column headers.
MODULE_LOOKUP_CSV_NAME = "kegg_module_lookup.csv"

# DRAM's own KO<->KEGG-module reference sheets (the same files DRAM-bio
# downloads into its database during `DRAM-setup.py prepare_databases`,
# and the same data it uses internally to compute kegg_pathway_completeness
# .csv's percentages) -- fetched from here rather than re-deriving a
# possibly-inconsistent mapping from a separate KEGG source. See
# src/pathway_features/kegg_modules.py.
MODULE_STEP_FORM_URL = "https://raw.githubusercontent.com/WrightonLabCSU/DRAM/master/data/module_step_form.tsv"
ETC_MODULE_DATABASE_URL = "https://raw.githubusercontent.com/WrightonLabCSU/DRAM/master/data/etc_module_database.tsv"

# Local cache for the two files above, so repeated runs don't re-fetch over
# the network. Delete this directory (or pass --refresh-mapping) to pick up
# any upstream change to DRAM's reference sheets.
KEGG_MODULE_MAPPING_CACHE_DIR = ROOT / "data" / "interim" / "dram" / "kegg_module_mapping"

# --- pipeline output filenames (local, post-DRAM) ---
# Bare filenames, not full paths: the containing directory is runtime-
# configurable via config/dataset/pathway_features.yaml's output_dir, set by
# build_pathway_features.py -- these constants exist so the names live in
# exactly one place instead of being duplicated as string literals there.

# CANONICAL modeling input: strain x pathway, continuous completeness
# fractions (0.0-1.0). This -- not the binary matrix below -- is what
# downstream feature pipelines should read. Always written.
COMPLETENESS_CSV_NAME = "kegg_pathway_completeness.csv"

# Optional diagnostic export only (--emit-diagnostics). Binary presence/
# absence thresholded from the completeness matrix -- useful for manual
# inspection, NOT for modeling (thresholding throws away signal a model
# could otherwise use).
PRESENCE_CSV_NAME = "kegg_pathway_presence.csv"

# Optional EDA export only (--emit-diagnostics). A single PCA fit across
# ALL strains, for visualization/exploration. MUST NOT be joined into
# training data as-is: fitting PCA globally (rather than per-CV-fold, on
# training strains only) leaks test-strain information into the
# transform. The modeling pipeline must fit its own PCA inside the CV
# loop if pathway features are to be dimensionality-reduced there.
EDA_PCA_CSV_NAME = "eda_pca_exploratory.csv"
EDA_PCA_LOADINGS_CSV_NAME = "eda_pca_exploratory_loadings.csv"

# --- DRAC credentials (gitignored -- see drac_credentials.py) ---
# No lab's Slurm billing account is hardcoded anywhere in this pipeline.
# Each user is prompted for their own the first time they run
# prepare_dram_manifest.py, same pattern as strain_metadata's NCBI email.
DRAC_CREDENTIALS_FILE = ROOT / ".drac_credentials.json"

# --------------------------------------------------------------------------
# TUNABLES (overridable via config/dataset/pathway_features.yaml)
# --------------------------------------------------------------------------

# Only used for the optional --emit-diagnostics presence/absence export, not
# for the canonical completeness matrix. A module/pathway is "present" once
# its completeness (0-1) reaches this threshold. 0.8 favors precision over
# recall: with only ~26 strains and ~500 candidate pathways, false positives
# from fragmented/partial pathway annotations are a bigger risk than missing
# a borderline-complete one.
DEFAULT_COMPLETENESS_THRESHOLD = 0.8

# Only used for the optional --emit-diagnostics EDA PCA export (eda_pca_
# exploratory.csv), never for the canonical modeling input. PCA there runs
# on raw completeness percentages, not the thresholded binary matrix, so the
# gradient between e.g. 40%- and 90%-complete pathways isn't collapsed
# before PCA gets to see it. The modeling pipeline must fit its own PCA
# inside cross-validation if it wants a reduced pathway representation.
DEFAULT_PCA_N_COMPONENTS = 15
DEFAULT_PCA_RANDOM_STATE = 42
