#!/bin/bash
# scripts/hpc/run_modular_annotate.sbatch
# ─────────────────────────────────────────────────────────────────────────────
# Runs the modular annotation pipeline on the FASTA files shipped to this
# cluster by prepare_modular_manifest.py. Requires
# scripts/hpc/setup_modular_env.sh to have been run first (venv + binaries +
# databases).
#
# Three steps only, all on this cluster:
#   1. Pyrodigal   -- gene calling, once per genome (single mode: these are
#                      assembled isolate genomes, not metagenomic contigs, so
#                      per-genome training is more accurate than meta mode)
#   2. PyKofamSearch -- KO annotation, one combined call across all genomes'
#                      called proteins (avoids reloading the KOfam HMM
#                      database once per genome)
#   3. CheckM2     -- genome completeness/contamination, one batched call
#                      across all raw genome FASTAs (it does its own internal
#                      gene calling via the hand-installed Prodigal/DIAMOND
#                      binaries -- does NOT consume Pyrodigal's output)
#
# give_completeness (module completeness scoring) is DELIBERATELY NOT run
# here -- it runs locally against ko_annotation/'s output after rsync-back.
# See setup_modular_env.sh and docs/pathway_features.md.
#
# Submit with (--account is required -- prepare_modular_manifest.py prints
# the exact command with your own account filled in):
#   sbatch --account=<your-cpu-account> scripts/hpc/run_modular_annotate.sbatch
#
# See docs/pathway_features.md for the full local + DRAC workflow.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=modular-annotate
#SBATCH --time=01:00:00                   # confirmed full 25-genome run: 36m26s @ 16 cpus
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G                         # confirmed full 25-genome run: peak 24.9GB RSS (26118348K) -- PyKofamSearch's fixed ~28k-HMM database load dominates this, not genome count
#SBATCH --output=%x-%j.out

set -euo pipefail

MODULAR_ROOT="${MODULAR_ROOT:-$HOME/scratch/modular_pathway}"

# Matches the environment set up by setup_modular_env.sh.
module load StdEnv/2023 python
export PATH="${MODULAR_ROOT}/tools/diamond:${MODULAR_ROOT}/tools/prodigal:${PATH}"
# shellcheck disable=SC1091
source "${MODULAR_ROOT}/venv/bin/activate"

FASTA_DIR="${FASTA_DIR:-$MODULAR_ROOT/fasta}"
OUTPUT_DIR="${OUTPUT_DIR:-$MODULAR_ROOT/output}"
KOFAM_DB="${KOFAM_DB:-$MODULAR_ROOT/databases/kofam/database.pkl.gz}"
CHECKM2_DB_DIR="${CHECKM2_DB_DIR:-$MODULAR_ROOT/databases/checkm2}"

GENES_DIR="${OUTPUT_DIR}/genes"
KO_DIR="${OUTPUT_DIR}/ko_annotation"
CHECKM2_OUT_DIR="${OUTPUT_DIR}/checkm2"

# Fresh output each attempt -- nothing here is the final result, that only
# exists once rsynced back to this repo's data/external/modular_pathway/.
rm -rf "${OUTPUT_DIR}"
mkdir -p "${GENES_DIR}" "${KO_DIR}" "${CHECKM2_OUT_DIR}"

# ── Step 1: Pyrodigal gene calling, once per genome ─────────────────────────
# Protein headers are prefixed with the genome accession (Pyrodigal's default
# header is just "{contig_id}_{gene_number}", with no genome identifier) so
# the combined protein FASTA below stays traceable back to its source genome
# (see src/pathway_features/modular_ko.py).
echo "== Step 1/3: Pyrodigal gene calling =="
ALL_PROTEINS="${GENES_DIR}/all_proteins.faa"
: > "${ALL_PROTEINS}"
for genome in "${FASTA_DIR}"/*.fna; do
    accession="$(basename "${genome}" .fna)"
    pyrodigal -i "${genome}" \
        -a "${GENES_DIR}/${accession}.faa" \
        -o "${GENES_DIR}/${accession}.gff" -f gff \
        -p single -j "${SLURM_CPUS_PER_TASK}"
    sed "s/^>/>${accession}_/" "${GENES_DIR}/${accession}.faa" >> "${ALL_PROTEINS}"
done

# ── Step 2: PyKofamSearch KO annotation, one combined call ─────────────────
echo "== Step 2/3: PyKofamSearch KO annotation =="
pykofamsearch -i "${ALL_PROTEINS}" \
    -o "${KO_DIR}/ko_hits.tsv" \
    -b "${KOFAM_DB}" \
    -p "${SLURM_CPUS_PER_TASK}"

# ── Step 3: CheckM2 genome quality prediction ───────────────────────────────
# Runs on the RAW genome FASTAs directly (its own internal Prodigal/DIAMOND
# call gene calling itself) -- independent of Step 1's Pyrodigal output.
echo "== Step 3/3: CheckM2 genome quality prediction =="
checkm2 predict \
    --threads "${SLURM_CPUS_PER_TASK}" \
    --input "${FASTA_DIR}" \
    --output-directory "${CHECKM2_OUT_DIR}" \
    --database_path "$(find "${CHECKM2_DB_DIR}" -name '*.dmnd' -print -quit)"

echo "Done."
echo "Copy ${KO_DIR}/ back to this repo's data/external/modular_pathway/ko_annotation/"
echo "Copy ${CHECKM2_OUT_DIR}/quality_report.tsv back to this repo's data/external/modular_pathway/checkm2/"
echo "(e.g. via rsync) then run the local build scripts -- see docs/pathway_features.md."

