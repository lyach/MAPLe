#!/bin/bash
# Rebuilds data/external/bacdive_growth_full_texpand.parquet from the local
# BacDive cache only (no API) -- the full default schema (--full-schema,
# the script's default -- passed explicitly here to keep the invocation
# self-documenting), with temperature_range_expansion ON, so a reported
# temperature range like "10-41" emits three rows (min, max, mean) instead
# of one midpoint. This is the expanded sibling of
# bacdive_growth_full.parquet (see run_bacdive_build_full_schema.sbatch) --
# same schema, more rows, never the same filename
# (config.stamp_temperature_expansion appends "_texpand" whenever this flag
# resolves true).
#
# Reads ~100k small JSON files off the shared filesystem, so it belongs on
# the scheduler rather than a login node even though it makes no network
# calls. Historical timing: ~130 records/sec single-threaded, so ~13 min
# for the full cache.
#
# Submit:
#   sbatch scripts/hpc/run_bacdive_build_full_schema_texpand.sbatch

#SBATCH --job-name=bacdive-build-full-schema-texpand
#SBATCH --account=def-mahadeva_cpu
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/projects/rrg-mahadeva/MicrobeNet}"

source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"

python scripts/bacdive/build_bacdive_from_cache.py \
    --full-schema \
    --temperature-range-expansion \
    --no-progress
