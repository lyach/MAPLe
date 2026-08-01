# Building metabolic-pathway features (modular pipeline)

## Overview

This pipeline derives a strain × KEGG-pathway feature matrix from the genome FASTA files in `data/external/fasta/` (see [`docs/plans/pathway_features.md`](plans/pathway_features.md) for the original design). It's a modular, license-independent pipeline built from four small, purpose-specific tools:

- **Pyrodigal** — gene calling
- **PyKofamSearch** — KO (KEGG Orthology) annotation
- **EBI's `give_completeness`** (kegg-pathways-completeness-tool) — KEGG-module completeness scoring
- **CheckM2** — genome completeness/contamination (a required confounder feature, independent of pathway coverage)

No dimensionality reduction is precomputed — if the modeling pipeline wants a reduced pathway representation, it must fit PCA inside its own cross-validation loop, on training strains only, not globally here.

**Status: complete, confirmed working end-to-end on real data, all 25 strains.** Environment setup, a smoke test, the full 25-genome annotation run, and all local post-processing (module completeness, CheckM2 confounder, gene × pathway matrix) are done and verified against the real 25-strain dataset — not fixtures. **Full-run footprint: 24.9GB peak RSS / 36m26s** for annotation on Nibi.

---

## Running this pipeline end-to-end

Every command below, in order, no commentary — see "Step-by-step workflow" further down for the reasoning behind each one.

```bash
# 1. Local: build the manifest
python scripts/prepare_modular_manifest.py

# 2. Ship FASTA files + HPC scripts to your cluster (replace Nibi with your own SSH host/alias)
rsync -avz data/external/fasta/ Nibi:scratch/modular_pathway/fasta/
rsync -avz scripts/hpc/ Nibi:scratch/modular_pathway/hpc/

# 3. On the cluster: one-time environment + database setup
sbatch --account=<your-cpu-account> --mem=32G --cpus-per-task=4 --time=3:00:00 \
  --job-name=modular-setup --output=%x-%j.out \
  --wrap="bash \$HOME/scratch/modular_pathway/hpc/setup_modular_env.sh"

# 4. On the cluster, once step 3 finishes: run the annotation (Pyrodigal -> PyKofamSearch -> CheckM2)
sbatch --account=<your-cpu-account> $HOME/scratch/modular_pathway/hpc/run_modular_annotate.sbatch

# 5. Copy the results back
rsync -avz Nibi:scratch/modular_pathway/output/ko_annotation/ data/external/modular_pathway/ko_annotation/
rsync -avz Nibi:scratch/modular_pathway/output/checkm2/quality_report.tsv data/external/modular_pathway/checkm2/
rsync -avz --exclude='all_proteins.faa' --include='*.faa' --exclude='*.gff' \
  Nibi:scratch/modular_pathway/output/genes/ data/external/modular_pathway/genes/

# 6. Local: build the canonical completeness matrix + CheckM2 confounder feature
python scripts/build_modular_pathway_features.py config/dataset/modular_pathway_features.yaml
# -> data/external/modular_pathway/kegg_pathway_completeness.csv  (25 strains x 385 modules)
# -> data/external/modular_pathway/checkm2_quality.csv

# 7. Local: build the gene x pathway membership matrix
python scripts/build_modular_gene_pathway_matrix.py config/dataset/modular_gene_pathway_membership.yaml
# -> data/external/modular_pathway/gene_pathway_membership/gene_pathway_membership_{strain}.csv
# -> data/external/modular_pathway/gene_pathway_membership/kegg_module_lookup.csv
```

Steps 1 and 6–7 are fast, local, and safe to re-run (pass `--force` to overwrite existing output). Steps 3–5 involve the HPC cluster and take from minutes (step 3) to ~40 minutes (step 4, confirmed on 25 genomes) — not something to re-run live in a short demo.

### Why these tools

No KEGG license, no Ruby/conda dependency (matching Alliance's guidance against conda on the shared filesystem), and no heavy database build — everything installs via pip plus two small hand-installed static binaries (DIAMOND, Prodigal — both needed internally by CheckM2, not by the main gene-calling step).

| Job | Tool | Version | Why this one |
|---|---|---|---|
| Gene calling | [Pyrodigal](https://github.com/althonos/pyrodigal) | `3.7.1` | Cython reimplementation of Prodigal, pure pip wheel (no external binary), actively maintained. CLI mirrors Prodigal's own flags. |
| KO annotation | [PyKofamSearch](https://github.com/jolespin/pykofamsearch) | `2025.9.5` | PyHMMER-based KofamScan reimplementation, pure pip wheel. Checked maintenance health directly: `main` branch activity stops 2025-09, but `dev` branch has a merged PR as recent as 2026-07-22 — actively maintained via a dev-branch workflow, not stale. Pinned to the last PyPI release (`2025.9.5`) rather than installing from `dev`, since that branch's newer feature (`--anvio_bitscore_heuristic`) isn't needed here. |
| Module completeness | [kegg-pathways-completeness](https://github.com/EBI-Metagenomics/kegg-pathways-completeness-tool) (`give_completeness`) | `1.4.3` | EBI tool, bundles its own reference data (no database download), runs **locally** — see below. Verified live against a real KO table; outputs real KEGG module IDs (`module_accession`) directly, no separate ID-mapping step needed. |
| Genome quality (confounder feature) | [CheckM2](https://github.com/chklovski/CheckM2) | `1.1.0` | ML-based completeness/contamination predictor. **Installed from its GitHub tag, not PyPI** — PyPI is stuck on the 2023-01 `1.0.1` release, while `1.1.0` (tagged 2025-02) is the version matching the current reference database (Zenodo record 14897628). Needs DIAMOND `2.1.11` + Prodigal `2.6.3` internally (both hand-installed static binaries, versions pinned to exactly what CheckM2's own `checkm2.yml` conda recipe tests against — CheckM2 declares no pip `install_requires` itself, so nothing pins these for you). |

### This is the first HPC-dependent step in the repo

Pyrodigal, PyKofamSearch, and CheckM2 run on a DRAC (Digital Research Alliance of Canada) HPC cluster. `give_completeness` is fast and needs no database, so it runs **locally** instead — meaning the module-completeness step is quick to iterate on/re-run without touching DRAC at all. The pipeline runs across **two environments**:

```
LOCAL                          REMOTE (DRAC/Nibi)                    LOCAL
─────                          ───────────────────                  ─────
prepare_modular_manifest.py →  1. Pyrodigal (gene calling)
                                2. PyKofamSearch (KO annotation)  →  build_modular_pathway_features.py
                                3. CheckM2 (genome quality)       →    (give_completeness + CheckM2 confounder)
                                                                   →  build_modular_gene_pathway_matrix.py
```

Nothing in this repo can execute the remote step for you — a DRAC account/allocation and a manual `rsync`/`sbatch`/`rsync` round trip are required.

---

## What it produces

| Output | Path | Contents |
|---|---|---|
| FASTA manifest | `data/interim/modular_pathway/fasta_manifest.csv` | strain ↔ accession ↔ FASTA filename, cross-validated against `strains_metadata.csv` |
| KO annotation (copied back from DRAC) | `data/external/modular_pathway/ko_annotation/ko_hits.tsv` | PyKofamSearch's raw per-protein KO hits |
| CheckM2 output (copied back from DRAC) | `data/external/modular_pathway/checkm2/quality_report.tsv` | per-genome completeness/contamination, raw |
| Per-genome protein FASTAs (copied back from DRAC) | `data/external/modular_pathway/genes/{accession}.faa` | Pyrodigal's predicted proteins — needed for the full gene universe, not just hits |
| **Completeness matrix — canonical modeling input** | `data/external/modular_pathway/kegg_pathway_completeness.csv` | strain × KEGG-module, continuous 0.0-1.0 fractions. **Confirmed on real data: 25 strains × 385 modules.** |
| **CheckM2 confounder — required feature** | `data/external/modular_pathway/checkm2_quality.csv` | strain × `[checkm2_completeness, checkm2_contamination]`, 0.0-1.0 fractions |
| Gene × pathway membership (additive) | `data/external/modular_pathway/gene_pathway_membership/gene_pathway_membership_{strain}.csv` | gene × KEGG-module-ID, binary, multi-label. One file per strain. Independent of the completeness matrix. |
| KEGG module lookup | `data/external/modular_pathway/gene_pathway_membership/kegg_module_lookup.csv` | module ID → human-readable name, written once |
| Run log | `logs/run_<timestamp>.log` | full record of the local scripts |

---

## Step-by-step workflow

### 1. Local: prepare the manifest

```
python scripts/prepare_modular_manifest.py
```

Cross-checks `data/external/strains_metadata.csv` against `data/external/fasta/`, writes `data/interim/modular_pathway/fasta_manifest.csv`, and prints the rsync + sbatch commands to ship the FASTA files and `scripts/hpc/` to your DRAC cluster. Any FASTA file without a metadata row (or vice versa) is excluded and logged rather than silently included. Confirmed working against the real 25/26-strain FASTA set.

The printed `sbatch` commands include your Slurm billing account, filled in automatically. **No lab's account is hardcoded anywhere in this repo** — the first time you run this script, you'll be prompted for your own (e.g. `def-yourpi` or `rrg-yourpi`; find yours on the cluster with `sacctmgr show associations user=$USER format=account%30`), and it's cached locally in `.drac_credentials.json` (gitignored, mirrors how `strain_metadata`'s NCBI email works — see `src/pathway_features/drac_credentials.py`). Set the `DRAC_ACCOUNT` environment variable instead to skip the prompt (e.g. in CI).

### 2. Remote: run the pipeline on DRAC

```
sbatch --account=<your-cpu-account> --mem=32G --cpus-per-task=4 --time=3:00:00 \
  --job-name=modular-setup --output=%x-%j.out \
  --wrap="bash $HOME/scratch/modular_pathway/hpc/setup_modular_env.sh"
```

then, once that completes:

```
sbatch --account=<your-cpu-account> $HOME/scratch/modular_pathway/hpc/run_modular_annotate.sbatch
```

`setup_modular_env.sh` creates a venv, pip-installs Pyrodigal/PyKofamSearch/CheckM2's dependencies (pinned versions per the table above), hand-installs the DIAMOND/Prodigal static binaries CheckM2 needs internally, downloads the KOfam database (`serialize_kofam_models`, over KEGG's FTP server — re-verify compute-node FTP access if run on a cluster other than Nibi), and downloads CheckM2's reference database (`checkm2 database --download`, from Zenodo). Confirmed running clean end-to-end on Nibi with no errors.

`run_modular_annotate.sbatch` runs the three remote steps: Pyrodigal once per genome (single mode — these are assembled isolate genomes, not metagenomic contigs, so per-genome training is more accurate than meta mode), a single combined PyKofamSearch call across all genomes' proteins (avoids reloading the KOfam HMM database once per genome), and one batched CheckM2 call across all raw genome FASTAs (CheckM2 does its own internal gene calling, independent of Pyrodigal's output).

**`#SBATCH --mem`/`--time` are confirmed empirically, not guessed:** a 3-genome smoke test first (`--mem=8G` OOM-killed almost immediately — PyKofamSearch loads its entire ~28k-HMM database into memory up front regardless of genome count, so even a tiny smoke test needs close to full-scale memory; `--mem=32G` then succeeded at 17.5GB peak / 7m22s), followed by the real 25-genome run, measured at **24.9GB peak RSS / 36m26s**. The script's `#SBATCH` directives are set from that real full-run number with headroom, not from a first-pass guess.

Copy the result back:

```
rsync -avz Nibi:scratch/modular_pathway/output/ko_annotation/ data/external/modular_pathway/ko_annotation/
rsync -avz Nibi:scratch/modular_pathway/output/checkm2/quality_report.tsv data/external/modular_pathway/checkm2/
rsync -avz --exclude='all_proteins.faa' --include='*.faa' --exclude='*.gff' \
  Nibi:scratch/modular_pathway/output/genes/ data/external/modular_pathway/genes/
```

### 3. Local: module completeness (`give_completeness`) + CheckM2 confounder

```
python scripts/build_modular_pathway_features.py config/dataset/modular_pathway_features.yaml
```

Runs `give_completeness` **locally via subprocess** (it's fast and database-free) against PyKofamSearch's KO hits, and writes both required outputs:

- `data/external/modular_pathway/kegg_pathway_completeness.csv` — canonical modeling input, strain × KEGG-module, continuous `[0, 1]` fractions. **Confirmed on real data: 25 strains × 385 modules** — `give_completeness` reports real module IDs directly for everything it scores.
- `data/external/modular_pathway/checkm2_quality.csv` — required confounder feature, strain × `[checkm2_completeness, checkm2_contamination]`, `[0, 1]` (CheckM2 reports 0–100, rescaled to match this repo's convention).

Two things worth knowing before trusting this script's output on new data:

- **`give_completeness` without `--add-per-contig` pools every genome in the input into one combined result** — confirmed by testing directly. This script always passes `--add-per-contig` to get real per-genome rows.
- **A genome with zero KO hits produces zero rows in the output, silently — not an error, not a zero-filled row.** `build_completeness_matrix` explicitly reindexes against the full known strain list and zero-fills, rather than trusting a plain pivot to include every strain.

`quality_report.tsv`'s row presence also doubles as this script's per-strain validation (a strain missing from it means the pipeline genuinely failed on that genome) — a genome with real predicted genes but zero KO hits is a separate, valid, non-error case (all-zero completeness row), since `ko_hits.tsv` alone can't tell the two apart.

Same `--force` overwrite guard pattern used throughout this pipeline: refuses to overwrite existing output unless `--force` is passed.

### 4. Local: gene × pathway membership

```
python scripts/build_modular_gene_pathway_matrix.py config/dataset/modular_gene_pathway_membership.yaml
```

ADDITIVE: does not read, write, or modify `kegg_pathway_completeness.csv` or `checkm2_quality.csv`, or their generation logic. Sourced from PyKofamSearch's `ko_hits.tsv`. Produces a binary matrix of every predicted gene (rows, including zero-hit genes as explicit all-zero rows) against KEGG module IDs (columns, the same module set `kegg_pathway_completeness.csv` actually has). **Confirmed on real data: 25 strains, 94,149 total genes, 11,589 genes (12.3%) with ≥1 module hit.**

**Multi-label, not one-hot.** A gene's KO can legitimately belong to more than one KEGG module. Confirmed directly on real output: 258 genes in *E. coli* K12 alone hit more than one module, one hitting 8 simultaneously (a central-metabolism enzyme spanning TCA/related modules). Every applicable module gets a 1 for that gene; this is not forced into a single "best" pathway assignment, because that would misrepresent real biology.

Two design points that came from testing against real output, not assumption:

- **Requires the per-genome protein FASTAs** (`data/external/modular_pathway/genes/{accession}.faa`, rsynced from `$MODULAR_ROOT/output/genes/` — excluding the combined `all_proteins.faa`), not just `ko_hits.tsv`. Genes with zero KO hits never appear in `ko_hits.tsv` at all, so the all-zero-row spec needs Pyrodigal's own header list as the true gene universe. **The per-genome `.faa` files on disk still have Pyrodigal's raw, unprefixed headers** (e.g. `NZ_AP023410.1_1`) — only the *combined* `all_proteins.faa` that PyKofamSearch actually ran on got accession-prefixed (`run_modular_annotate.sbatch`'s `sed` step). `load_gene_universe` re-applies that same prefix so gene IDs match `ko_hits.tsv`'s `id_protein` values exactly — verified directly against real IDs.
- **The KO → module mapping is sourced from `give_completeness`'s own bundled reference data** (`kegg_pathways_completeness/pathways_data/modules_table.tsv`, installed alongside the package — no download needed), not re-derived from a separate source. This guarantees the gene × pathway matrix and the completeness matrix share the same KO → module ground truth. Covers 570 modules / 3091 KOs — confirmed to fully cover every module the completeness matrix actually reports (checked directly, 0 gaps).

Same `--force` overwrite guard as step 3. Fails loudly and writes nothing (not a partial result set) if any strain's protein FASTA is missing or empty — this is different from a strain's genes legitimately having no KO hits, which is normal and produces an all-zero-but-present row, not an error.

---

## Module layout

```
scripts/
├── prepare_modular_manifest.py          entry point — local, pre-annotation
├── build_modular_pathway_features.py    entry point — local, post-annotation
│                                          (give_completeness + CheckM2 confounder)
├── build_modular_gene_pathway_matrix.py entry point — local, post-annotation, additive
└── hpc/
    ├── setup_modular_env.sh        remote — venv + binaries + database setup
    ├── run_modular_annotate.sbatch remote — Pyrodigal + PyKofamSearch + CheckM2
    └── README.md
src/
├── common/
│   └── logging_setup.py           shared console + timestamped-file logging
└── pathway_features/
    ├── config.py                        paths, tunable defaults
    ├── manifest.py                        strain ↔ accession ↔ FASTA-path table
    ├── drac_credentials.py                  resolves your Slurm billing account (no lab hardcoded)
    ├── strain_names.py                        joins any accession-indexed table to canonical strain names
    ├── checkm2_quality.py                       parses quality_report.tsv (confounder feature)
    ├── modular_ko.py                              loads ko_hits.tsv + Pyrodigal's per-genome gene universe
    ├── modular_completeness.py                      runs give_completeness locally, reshapes its sparse
    │                                                  output into the dense completeness matrix (canonical
    │                                                  modeling input, written directly)
    ├── modular_kegg_modules.py                        KO → KEGG-module-ID mapping, from give_completeness's
    │                                                    own bundled reference data
    └── modular_gene_pathway_membership.py               per-strain gene × module-ID binary matrix
                                                            (multi-label, additive output)
```

---

## Known limitations

- **Requires a DRAC account** — both `setup_modular_env.sh` and `run_modular_annotate.sbatch` are confirmed working end-to-end on Nibi, but re-check module/FTP availability on other clusters, as they can differ.
- **PyKofamSearch's memory footprint is dominated by its database load, not genome count** — confirmed directly: a 3-genome smoke test needed nearly the same memory as the full 25-genome run (loading the ~28k-HMM serialized database up front is the fixed cost). Don't assume a smaller test run implies proportionally smaller memory needs at full scale.
- **Not yet wired into the modeling dataset.** `src/modeling/features.py` lives only on the unmerged `origin/modeling` branch. Wiring this in means pointing a `path:` feature spec at `kegg_pathway_completeness.csv`, **and** making sure `features.py` fits any PCA/dimensionality reduction on pathway columns inside its own cross-validation, not by re-reading a precomputed PCA file.
