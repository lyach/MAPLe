#!/usr/bin/env bash
# One-time environment + database setup for the modular pathway pipeline
# on a DRAC HPC cluster: Pyrodigal, PyKofamSearch, CheckM2 + its DIAMOND/
# Prodigal dependencies. Submit via Slurm, not on a login node:
#
#   sbatch --account=<your-cpu-account> --mem=32G --cpus-per-task=4 --time=3:00:00 \
#     --job-name=modular-setup --output=%x-%j.out \
#     --wrap="bash \$HOME/scratch/modular_pathway/hpc/setup_modular_env.sh"
#
# See docs/modular_pathway_features.md for the full workflow.
set -euo pipefail

MODULAR_ROOT="${MODULAR_ROOT:-$HOME/scratch/modular_pathway}"
VENV_DIR="${VENV_DIR:-$MODULAR_ROOT/venv}"
TOOLS_DIR="${TOOLS_DIR:-$MODULAR_ROOT/tools}"
KOFAM_DB_DIR="${KOFAM_DB_DIR:-$MODULAR_ROOT/databases/kofam}"
CHECKM2_DB_DIR="${CHECKM2_DB_DIR:-$MODULAR_ROOT/databases/checkm2}"

echo "== Loading Alliance's python module =="
module load StdEnv/2023 python

echo "== Installing official DIAMOND v2.1.11 static binary (pinned to match CheckM2 1.1.0's own tested version, not the latest release) =="
mkdir -p "${TOOLS_DIR}/diamond"
if [ ! -f "${TOOLS_DIR}/diamond/diamond" ]; then
    curl -L https://github.com/bbuchfink/diamond/releases/download/v2.1.11/diamond-linux64.tar.gz \
        | tar -xz -C "${TOOLS_DIR}/diamond"
    chmod +x "${TOOLS_DIR}/diamond/diamond"
fi

echo "== Installing official Prodigal v2.6.3 static binary (CheckM2's internal gene caller) =="
mkdir -p "${TOOLS_DIR}/prodigal"
if [ ! -f "${TOOLS_DIR}/prodigal/prodigal" ]; then
    curl -L -o "${TOOLS_DIR}/prodigal/prodigal" https://github.com/hyattpd/Prodigal/releases/download/v2.6.3/prodigal.linux
    chmod +x "${TOOLS_DIR}/prodigal/prodigal"
fi
export PATH="${TOOLS_DIR}/diamond:${TOOLS_DIR}/prodigal:${PATH}"

echo "== Creating Python virtualenv at ${VENV_DIR} =="
virtualenv --no-download "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --no-index --upgrade pip

echo "== Installing Pyrodigal 3.7.1 (gene calling) and PyKofamSearch 2025.9.5 (KO annotation) =="
pip install pyrodigal==3.7.1 pykofamsearch==2025.9.5

echo "== Installing CheckM2 1.1.0's ML/data dependencies (pinned to match its own checkm2.yml conda recipe, since CheckM2 declares no pip install_requires itself) =="
pip install \
    "scikit-learn==1.6.1" \
    "tensorflow==2.17.*" \
    lightgbm pandas numpy scipy requests packaging tqdm setuptools

echo "== Installing CheckM2 1.1.0 from its GitHub tag (PyPI is stuck on the stale 2023-01 1.0.1 release -- see header note above) =="
pip install "git+https://github.com/chklovski/CheckM2.git@1.1.0"

echo "== Downloading + serializing the KOfam database for PyKofamSearch =="
mkdir -p "${KOFAM_DB_DIR}"
if [ ! -f "${KOFAM_DB_DIR}/database.pkl.gz" ]; then
    serialize_kofam_models -o "${KOFAM_DB_DIR}"
else
    echo "   ${KOFAM_DB_DIR}/database.pkl.gz already exists -- skipping re-download."
fi

echo "== Downloading the CheckM2 reference database (Zenodo record 14897628, matches CheckM2 1.1.0) =="
mkdir -p "${CHECKM2_DB_DIR}"
if [ -z "$(find "${CHECKM2_DB_DIR}" -name '*.dmnd' -print -quit 2>/dev/null)" ]; then
    checkm2 database --download --path "${CHECKM2_DB_DIR}"
else
    echo "   A .dmnd database already exists under ${CHECKM2_DB_DIR} -- skipping re-download."
fi

echo "== Done. run_modular_annotate.sbatch expects this environment: =="
echo "   module load StdEnv/2023 python"
echo "   export PATH=\"${TOOLS_DIR}/diamond:${TOOLS_DIR}/prodigal:\$PATH\""
echo "   source ${VENV_DIR}/bin/activate"
