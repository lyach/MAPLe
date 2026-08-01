# scripts/hpc/

Artifacts for running the modular pathway-feature pipeline (Pyrodigal +
PyKofamSearch + CheckM2) on a DRAC (Digital Research Alliance of Canada) HPC
cluster. These are **not** run by anything in this repo automatically —
`prepare_modular_manifest.py` prints the rsync commands to ship them over,
and you run them by hand on the cluster.

- `setup_modular_env.sh` — one-time venv + hand-installed binaries +
  database setup (run from a login/data-transfer node).
- `run_modular_annotate.sbatch` — the actual annotation job (`sbatch` this).

Full workflow, including what to do with the output afterward: see
[docs/modular_pathway_features.md](../../docs/modular_pathway_features.md).
