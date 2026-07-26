#!/bin/bash
# scripts/hpc/run_dram_annotate.sbatch
# ─────────────────────────────────────────────────────────────────────────────
# Runs DRAM annotate + distill on the FASTA files shipped to this cluster by
# prepare_dram_manifest.py. Requires scripts/hpc/setup_dram_env.sh to have
# been run first (modules + venv + databases -- see that script for why
# conda is deliberately avoided here).
#
# Submit with (--account is required -- there's no lab-specific default
# baked into this script; prepare_dram_manifest.py prints the exact command
# with your own account filled in, resolved via drac_credentials.py):
#   sbatch --account=<your-cpu-account> scripts/hpc/run_dram_annotate.sbatch
#
# See docs/pathway_features.md for the full local + DRAC workflow.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=dram-annotate
#SBATCH --time=12:00:00                   # confirmed run: 25 genomes / 16 cpus took 5h38m; scale up for more genomes
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G                         # KOfam-only annotation needs <50GB -- confirmed
#SBATCH --output=%x-%j.out

set -euo pipefail

DRAM_ROOT="${DRAM_ROOT:-$HOME/scratch/dram}"

# Matches the environment set up by setup_dram_env.sh: Alliance modules for
# DRAM's binary dependencies + a Python virtualenv for DRAM-bio itself
# (conda is avoided -- see setup_dram_env.sh for why).
module load StdEnv/2023 python prodigal/2.6.3 hmmer/3.4 mmseqs2/17-b804f trnascan-se/2.0.12 bedtools/2.31.0
# Nibi's prodigal/2.6.3 module produces broken/truncated GFF output (missing
# ID=/partial=/start_type=/score fields DRAM's annotation logic needs) --
# see RERUN_WHEN_ALLIANCE_PRODIGAL_FIXED in setup_dram_env.sh. This PATH
# prepend must come after the module load so it shadows the broken one.
export PATH="${DRAM_ROOT}/tools/prodigal:${DRAM_ROOT}/tools/barrnap/bin:${PATH}"
# shellcheck disable=SC1091
source "${DRAM_ROOT}/venv/bin/activate"

FASTA_DIR="${FASTA_DIR:-$DRAM_ROOT/fasta}"
OUTPUT_DIR="${OUTPUT_DIR:-$DRAM_ROOT/output}"

# DRAM.py annotate refuses to write into a pre-existing output directory
# (by design -- it won't silently overwrite a prior run). Since setup_dram_
# env.sh's database prep hit this exact class of leftover-state issue three
# separate times, clear this fresh each attempt rather than requiring a
# manual cleanup step if a retry is ever needed. Nothing here is the final
# output -- that only exists once copied back to this repo's
# data/external/dram/distillate/.
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

DRAM.py annotate \
    -i "${FASTA_DIR}/*.fna" \
    -o "${OUTPUT_DIR}/annotation" \
    --threads "${SLURM_CPUS_PER_TASK}"

DRAM.py distill \
    -i "${OUTPUT_DIR}/annotation/annotations.tsv" \
    -o "${OUTPUT_DIR}/distillate" \
    --trna_path "${OUTPUT_DIR}/annotation/trnas.tsv" \
    --rrna_path "${OUTPUT_DIR}/annotation/rrnas.tsv"

echo "Done. Copy ${OUTPUT_DIR}/distillate/ back to this repo's"
echo "data/external/dram/distillate/ (e.g. via rsync) then run"
echo "scripts/build_pathway_features.py locally."
