#!/usr/bin/env bash
# scripts/hpc/setup_dram_env.sh
# ─────────────────────────────────────────────────────────────────────────────
# One-time DRAM environment + database setup on a DRAC HPC cluster.
#
# Uses Alliance's module system + a Python virtualenv rather than conda:
# Alliance's docs discourage building conda environments in $HOME, since the
# tens of thousands of small files a conda env creates strain their shared
# parallel filesystem. Most of DRAM's binary dependencies are already
# available as Alliance modules; barrnap (pure Perl) is hand-installed below
# since it isn't a module at all. Confirmed available on Nibi via
# `module spider`: prodigal/2.6.3, hmmer/3.4, mmseqs2/17-b804f,
# trnascan-se/2.0.12, bedtools/2.31.0 (all under StdEnv/2023; bedtools is a
# barrnap dependency, only needed for the annotate step, not this script).
# Re-run `module spider <name>` on your own cluster/login node if any of
# these aren't found there.
#
# TODO(RERUN_WHEN_ALLIANCE_PRODIGAL_FIXED): Nibi's prodigal/2.6.3 module is
# ALSO hand-overridden below, for a different reason than barrnap -- it's a
# real module that exists, but its build is broken: it silently produces
# truncated GFF output missing the `ID=`/`partial=`/`start_type=`/score
# fields DRAM's annotation logic requires (confirmed by running the exact
# same input/flags through the official static v2.6.3 binary from
# https://github.com/hyattpd/Prodigal/releases/tag/v2.6.3, which produces
# correct output -- same Prodigal version, so this is specific to Alliance's
# build, not a real DRAM/Prodigal compatibility gap). Using the official
# static binary instead avoids ever needing to patch DRAM's gene-ID logic.
# If Alliance fixes their module, this override can be dropped -- but there's
# no real downside to leaving it, so no urgency either way.
#
# Do NOT run this directly on a login node -- the database-processing step
# can need up to ~128GB RAM, which will get throttled/killed on a shared
# login node. Submit it through Slurm instead:
#
#   sbatch --account=<your-cpu-account> --mem=128G --cpus-per-task=8 --time=6:00:00 \
#     --job-name=dram-setup --output=%x-%j.out \
#     --wrap="bash \$HOME/scratch/dram/hpc/setup_dram_env.sh"
#
# ($HOME-relative, not a relative path -- so it works regardless of your
# current directory when you submit. prepare_dram_manifest.py prints this
# exact command for you, with your account and remote dir already filled in.)
#
# This also needs internet access for `pip install` and the ~30GB
# KOfam/Pfam/dbCAN database download -- confirmed available from Nibi's
# compute nodes. Some other DRAC clusters restrict compute-node internet
# entirely, in which case the network-heavy parts of this script would need
# to run from a login/data-transfer node instead, with only the heavy
# processing submitted via Slurm.
#
# See docs/pathway_features.md for the full local + DRAC workflow.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DRAM_ROOT="${DRAM_ROOT:-$HOME/scratch/dram}"
VENV_DIR="${VENV_DIR:-$DRAM_ROOT/venv}"
TOOLS_DIR="${TOOLS_DIR:-$DRAM_ROOT/tools}"
DRAM_DB_DIR="${DRAM_DB_DIR:-$DRAM_ROOT/databases}"

# The bare `python` module resolves to CPython 3.11.5 on Nibi as of this
# writing (confirmed via successful venv creation) -- re-check `module
# spider python` on other clusters if this needs pinning to an exact version.
echo "== Loading Alliance modules for DRAM's binary dependencies =="
module load StdEnv/2023 python prodigal/2.6.3 hmmer/3.4 mmseqs2/17-b804f trnascan-se/2.0.12

echo "== Installing barrnap (not an Alliance module -- pure Perl, hand-installed) =="
# Pinned to tag 0.9, NOT master HEAD: master has grown into a much larger
# rewrite (barrnap 1.x) that checks for 12 external tools at startup
# (cmsearch, cmscan, bedtools, aragorn, any2fasta, seqkit, make, transterm,
# ziggypep, pyrodigal, diamond, taxonkit) -- discovered the hard way, one
# missing dependency at a time. Tag 0.9 is the original, minimal barrnap
# DRAM-bio 1.5.0 was actually built against: it only requires `nhmmer`
# (covered by the hmmer/3.4 module already loaded) and `bedtools`.
mkdir -p "${TOOLS_DIR}"
if [ ! -d "${TOOLS_DIR}/barrnap" ]; then
    git clone --branch 0.9 --depth 1 https://github.com/tseemann/barrnap.git "${TOOLS_DIR}/barrnap"
fi
export PATH="${TOOLS_DIR}/barrnap/bin:${PATH}"

echo "== Installing official Prodigal 2.6.3 static binary (overrides Nibi's broken module -- see RERUN_WHEN_ALLIANCE_PRODIGAL_FIXED above) =="
mkdir -p "${TOOLS_DIR}/prodigal"
if [ ! -f "${TOOLS_DIR}/prodigal/prodigal" ]; then
    curl -L -o "${TOOLS_DIR}/prodigal/prodigal" https://github.com/hyattpd/Prodigal/releases/download/v2.6.3/prodigal.linux
    chmod +x "${TOOLS_DIR}/prodigal/prodigal"
fi
# Must come after the module load above so this shadows the module's broken
# prodigal on PATH, not the other way around.
export PATH="${TOOLS_DIR}/prodigal:${PATH}"

echo "== Creating Python virtualenv at ${VENV_DIR} =="
virtualenv --no-download "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --no-index --upgrade pip
# DRAM-bio isn't in Alliance's local wheelhouse, so this one package installs
# from PyPI (fine -- we're on a login node with internet). Its own
# dependencies will prefer Alliance's --no-index wheels where available.
pip install DRAM-bio

echo "== Checking/patching known DRAM-bio 1.5.0 syntax bug in DRAM-setup.py (WrightonLabCSU/DRAM#330, #360) =="
# Verify-before-write: only patches if the exact known-broken block is found
# (exactly once); aborts loudly otherwise rather than guessing, e.g. if a
# newer DRAM-bio release ships a different bug or has already fixed this one.
DRAM_SETUP_PY="${VENV_DIR}/bin/DRAM-setup.py"
if python3 -m py_compile "${DRAM_SETUP_PY}" 2>/dev/null; then
    echo "DRAM-setup.py already compiles cleanly -- no patch needed."
else
    echo "DRAM-setup.py has the known syntax bug -- patching..."
    DRAM_SETUP_PY="${DRAM_SETUP_PY}" python3 - <<'PYEOF'
import os
import pathlib

path = pathlib.Path(os.environ["DRAM_SETUP_PY"])
text = path.read_text()

old_block = """    set_db_locs_parser.add_argument('--viral_loc', default=None,
    set_db_locs_parser.add_argument('--camper_tar_gz_loc', default=None,
                                    help='The source for the CAMPER database files, this is a tar.gz file downloaded'
                                    ' from the release page https://github.com/WrightonLabCSU/CAMPER/releases')
                                    help='mmseqs2 database file from ref seq viral gene collection')"""

new_block = """    set_db_locs_parser.add_argument('--viral_loc', default=None,
                                    help='mmseqs2 database file from ref seq viral gene collection')
    set_db_locs_parser.add_argument('--camper_tar_gz_loc', default=None,
                                    help='The source for the CAMPER database files, this is a tar.gz file downloaded'
                                    ' from the release page https://github.com/WrightonLabCSU/CAMPER/releases')"""

count = text.count(old_block)
if count != 1:
    raise SystemExit(
        f"Expected exactly 1 match for the known DRAM-bio 1.5.0 syntax-bug block in {path}, "
        f"found {count}. DRAM-bio may have shipped a different bug in this release -- check "
        "docs/pathway_features.md's 'Known upstream bug: DRAM-setup.py syntax error' section "
        "and patch manually."
    )
path.write_text(text.replace(old_block, new_block))
print(f"Patched {path}")
PYEOF
    python3 -m py_compile "${DRAM_SETUP_PY}" && echo "Confirmed: DRAM-setup.py now compiles cleanly."
fi

# TODO(RERUN_WHEN_KEGG_LICENSED): KOfam-only (--skip_uniref) is the correct
# config FOR NOW only because we don't have a KEGG license yet -- it is NOT
# the final config. KOfam gives KO/pathway calls without needing licensed
# KEGG data, which is enough to unblock modeling-pipeline development this
# week, but it omits UniRef90 (broader functional coverage). Once the KEGG
# license is in hand, DRAM must be re-run without --skip_uniref and all
# downstream pathway-feature CSVs regenerated -- KOfam-only output should
# not be treated as final for any published result.
#
# TODO(RERUN_WHEN_DBCAN_URL_FIXED): dbCAN (dbcan, dbcan_fam_activities,
# dbcan_subfam_ec) is excluded via --select_db below for an UNRELATED reason:
# DRAM-bio 1.5.0 hardcodes dbCAN download URLs at bcb.unl.edu, and that
# entire domain now redirects to https://pro.unl.edu/dbCAN2 (a webpage, not
# the raw files) -- an upstream DRAM-bio bug (stale URLs), not a licensing
# issue. dbCAN annotates carbohydrate-active enzymes, a different category
# from the KEGG/KOfam pathway-completeness scoring this pipeline needs, so
# it's safe to exclude for now. Once DRAM-bio ships working dbCAN URLs (or
# the correct current URLs are found and DRAM-bio's source is patched
# locally), re-run with dbcan/dbcan_fam_activities/dbcan_subfam_ec added
# back to --select_db.
#
# TODO(RERUN_WHEN_VOGDB_URL_FIXED): vogdb and vog_annotations are excluded
# via --select_db below for the SAME class of issue as dbCAN, different host:
# DRAM-bio 1.5.0 hardcodes http://fileshare.csb.univie.ac.at/vog/..., and
# that host now 301-redirects to fileshare.lisc.univie.ac.at -- the download
# silently produces a 0-byte file instead of erroring at download time (the
# failure only surfaces later, when hmmpress rejects the empty file). VOGDB
# (Virus Orthologous Groups) is, like dbCAN, unrelated to KEGG/KOfam pathway
# completeness, so safe to exclude for now. Re-run with vogdb/vog_annotations
# added back to --select_db once DRAM-bio's source is patched (or update the
# hardcoded host to fileshare.lisc.univie.ac.at directly).
echo "== Downloading and processing DRAM databases =="
echo "   NOTE: KOfam-only (--skip_uniref) -- license-pending fallback, see"
echo "   RERUN_WHEN_KEGG_LICENSED above. dbCAN and VOGDB also excluded"
echo "   (--select_db) -- unrelated stale-URL bugs, see"
echo "   RERUN_WHEN_DBCAN_URL_FIXED / RERUN_WHEN_VOGDB_URL_FIXED above."
echo "   Target: ${DRAM_DB_DIR}"
echo "   This can take up to ~5 hours and needs ~128GB RAM while processing --"
echo "   submit it as its own sbatch job if your login node can't provide that much."
# DRAM-setup.py's processing functions each mkdir() their own working
# directory (e.g. database_files/, kofam_profiles/, ...) without checking
# whether it already exists, and prepare_databases doesn't clean any of them
# up after a failed run -- so a retry against a dir with ANY leftover state
# from a previous attempt trips a bare FileExistsError, at whichever step's
# working directory happens to still be there. This isn't one specific
# directory to special-case: since prepare_databases always redownloads and
# reprocesses everything from scratch anyway (no incremental/resume support),
# a fully clean output directory before every run is both simpler and safer
# than chasing each stale subdirectory individually as it's discovered.
rm -rf "${DRAM_DB_DIR}"
mkdir -p "${DRAM_DB_DIR}"

DRAM-setup.py prepare_databases \
    --output_dir "${DRAM_DB_DIR}" \
    --skip_uniref \
    --select_db kofam_hmm --select_db kofam_ko_list \
    --select_db pfam --select_db pfam_hmm \
    --select_db viral --select_db peptidase \
    --select_db genome_summary_form --select_db module_step_form \
    --select_db function_heatmap_form --select_db amg_database \
    --select_db etc_module_database \
    --threads "${SLURM_CPUS_PER_TASK:-4}"

echo "== Done. run_dram_annotate.sbatch expects this environment: =="
echo "   module load StdEnv/2023 python prodigal/2.6.3 hmmer/3.4 mmseqs2/17-b804f trnascan-se/2.0.12 bedtools/2.31.0"
echo "   export PATH=\"${TOOLS_DIR}/prodigal:${TOOLS_DIR}/barrnap/bin:\$PATH\""
echo "   source ${VENV_DIR}/bin/activate"
