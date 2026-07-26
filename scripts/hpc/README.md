# scripts/hpc/

Artifacts for running DRAM on a DRAC (Digital Research Alliance of Canada)
HPC cluster. These are **not** run by anything in this repo automatically —
`prepare_dram_manifest.py` prints the rsync commands to ship them over, and
you run them by hand on the cluster.

- `setup_dram_env.sh` — one-time module/venv + database setup (run from a
  login/data-transfer node).
- `run_dram_annotate.sbatch` — the actual annotation job (`sbatch` this).

Full workflow, including what to do with the output afterward: see
[docs/pathway_features.md](../../docs/pathway_features.md).
