#!/bin/bash
# Rebuilds data/external/bacdive_growth_full.parquet from the local BacDive
# cache only (no API), picking up the new composition_raw backfill
# (src/bacdive/pipeline._backfill_composition) on top of the same 100,191
# cached records and extraction logic that produced the file currently at
# data/external/bacdive_growth_api_107k.parquet -- so the row count/identity
# should come out unchanged; only composition_raw (+ its derived
# has_*/complexity/n_ingredients columns) and the new composition_source
# column improve.
#
# --full-schema is the script's default now (every project build always
# wants the complete column set); it's passed explicitly here anyway so the
# invocation is self-documenting rather than relying on remembering the
# current default.
#
# --no-temperature-range-expansion is explicit and REQUIRED, and is
# load-bearing for the OUTPUT FILENAME, not just the data: it must match how
# the file being replaced was built (expansion off, 106,995 rows); the
# YAML's own default is on (132,821 rows for the same cache -- a different
# dataset, not a superset). The resolved value of this flag is what
# determines the actual output filename (below) --
# config.stamp_temperature_expansion / TARGETS["medium_growth"].output_file --
# so the two configs can never again collide on one path the way relying
# on the YAML default here overwrote bacdive_growth_api_107k.parquet with
# the other dataset's row count on 2026-08-24.
#
# Output path is left to resolve on its own (no --out): full schema +
# --no-temperature-range-expansion resolves to
# data/external/bacdive_growth_full.parquet.
#
# Reads ~100k small JSON files off the shared filesystem, so it belongs on
# the scheduler rather than a login node even though it makes no network
# calls. Measured (job 21350053, 2026-09-08): 6m59s elapsed, 1.9G MaxRSS --
# time/mem below sized off that with headroom, not off the original guess.
#
# Submit:
#   sbatch scripts/hpc/run_bacdive_build_full_schema.sbatch

#SBATCH --job-name=bacdive-build-full-schema
#SBATCH --account=def-mahadeva_cpu
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --output=%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/projects/rrg-mahadeva/MicrobeNet}"

source "${REPO_ROOT}/.venv/bin/activate"
cd "${REPO_ROOT}"

python scripts/bacdive/build_bacdive_from_cache.py \
    --full-schema \
    --no-temperature-range-expansion \
    --no-progress
