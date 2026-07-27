#!/usr/bin/env bash
# scripts/hpc/setup_modular_env.sh
# ─────────────────────────────────────────────────────────────────────────────
# One-time environment + database setup for the modular pathway pipeline,
# on a DRAC HPC cluster.
#
# Tools (versions confirmed/pinned after checking upstream maintenance
# health and, for CheckM2, that PyPI is stale relative to its own tagged
# releases -- see CHECKPOINT 1 report and docs/pathway_features.md):
#   * Pyrodigal 3.7.1        -- gene calling, pure pip wheel, no external binary
#   * PyKofamSearch 2025.9.5 -- PyHMMER-based KOfam search, pure pip wheel
#   * CheckM2 1.1.0          -- genome completeness/contamination confounder
#                                feature. PyPI is stuck on the 2023-01 1.0.1
#                                release; 1.1.0 is tagged on GitHub (2025-02)
#                                and matches the current reference database
#                                (Zenodo record 14897628), so installed from
#                                the GitHub tag, not PyPI.
#   * DIAMOND v2.1.11, Prodigal v2.6.3 -- CheckM2's own internal dependencies
#                                (gene calling + fast alignment it runs on
#                                itself before ML scoring). Versions pinned
#                                to exactly what CheckM2's own conda recipe
#                                (checkm2.yml) tests against, NOT the latest
#                                releases -- CheckM2 doesn't declare these as
#                                pip installable deps, so pip can't pin them
#                                for us; hand-installed as static binaries
#                                instead.
#
# give_completeness (EBI kegg-pathways-completeness-tool) is DELIBERATELY
# NOT installed here -- it runs locally against the KO tables shipped back
# from this script's output, not on the cluster. See pyproject.toml /
# docs/pathway_features.md.
#
# Alliance's docs discourage conda envs on the shared filesystem (too many
# small files). Everything here is pip + hand-installed static binaries, no
# conda/Ruby dependency.
#
# Do NOT run this directly on a login node -- submit it through Slurm
# instead (database downloads + CheckM2's ML deps are more than a login
# node should be doing):
#
#   sbatch --account=<your-cpu-account> --mem=32G --cpus-per-task=4 --time=3:00:00 \
#     --job-name=modular-setup --output=%x-%j.out \
#     --wrap="bash \$HOME/scratch/modular_pathway/hpc/setup_modular_env.sh"
#
# ($HOME-relative, not a relative path -- so it works regardless of your
# current directory when you submit. prepare_modular_manifest.py prints
# this exact command for you, with your account and remote dir filled in.)
# --mem/--time above ran clean with no OOM on Nibi (see the confirmed
# footprint note near the end of this script).
#
# Needs internet access for pip installs, the DIAMOND/Prodigal binary
# downloads, the pykofamsearch KOfam database download (KEGG's own FTP
# server), and the CheckM2 database download (Zenodo) -- confirmed
# available from Nibi's compute nodes; re-verify FTP specifically (KOfam's
# DB is served over FTP, not HTTPS) if this is run on a different cluster.
#
# See docs/pathway_features.md for the full local + DRAC workflow.
# ─────────────────────────────────────────────────────────────────────────────
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

# This script's own --mem=32G/--time=3:00:00 setup-job estimate above ran
# clean with no OOM (real peak not separately measured -- database downloads
# dominate wall time here, not RAM). run_modular_annotate.sbatch's own
# #SBATCH directives ARE now confirmed empirically: full 25-genome run
# measured at 24.9GB peak RSS / 36m26s.

echo "== Done. run_modular_annotate.sbatch expects this environment: =="
echo "   module load StdEnv/2023 python"
echo "   export PATH=\"${TOOLS_DIR}/diamond:${TOOLS_DIR}/prodigal:\$PATH\""
echo "   source ${VENV_DIR}/bin/activate"
