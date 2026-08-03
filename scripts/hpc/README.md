# scripts/hpc/

Artifacts for running Pyrodigal + PyKofamSearch + CheckM2 on a DRAC HPC cluster.
Not run automatically -- `prepare_modular_manifest.py` prints the rsync commands;
run these by hand on the cluster.

- `setup_modular_env.sh`: one-time venv + binaries + database setup.
- `run_modular_annotate.sbatch`: the annotation job (`sbatch` this).

See [docs/modular_pathway_features.md](../../docs/modular_pathway_features.md).
