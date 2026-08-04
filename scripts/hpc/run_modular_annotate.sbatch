#!/bin/bash
# Runs Pyrodigal -> PyKofamSearch -> CheckM2 on FASTA files shipped by
# prepare_modular_manifest.py. Requires setup_modular_env.sh run first.
# give_completeness runs locally afterward, not here.
#
#   sbatch --account=<your-cpu-account> scripts/hpc/run_modular_annotate.sbatch
#
# See docs/modular_pathway_features.md.

#SBATCH --job-name=modular-annotate
#SBATCH --time=01:00:00                   
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
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

rm -rf "${OUTPUT_DIR}"
mkdir -p "${GENES_DIR}" "${KO_DIR}" "${CHECKM2_OUT_DIR}"

# Step 1: Pyrodigal, once per genome. Headers prefixed with accession so
# the combined FASTA below stays traceable to its source genome.
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

# Step 2: PyKofamSearch, one combined call across all genomes
echo "== Step 2/3: PyKofamSearch KO annotation =="
pykofamsearch -i "${ALL_PROTEINS}" \
    -o "${KO_DIR}/ko_hits.tsv" \
    -b "${KOFAM_DB}" \
    -p "${SLURM_CPUS_PER_TASK}"

# Step 3: CheckM2, runs its own internal gene calling on raw FASTAs
echo "== Step 3/3: CheckM2 genome quality prediction =="
checkm2 predict \
    --threads "${SLURM_CPUS_PER_TASK}" \
    --input "${FASTA_DIR}" \
    --output-directory "${CHECKM2_OUT_DIR}" \
    --database_path "$(find "${CHECKM2_DB_DIR}" -name '*.dmnd' -print -quit)"

echo "Done."
echo "Copy ${KO_DIR}/ back to this repo's data/external/modular_pathway/ko_annotation/"
echo "Copy ${CHECKM2_OUT_DIR}/quality_report.tsv back to this repo's data/external/modular_pathway/checkm2/"
echo "(e.g. via rsync) then run the local build scripts -- see docs/modular_pathway_features.md."

