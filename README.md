# MAPLE: A **M**ultimodal **A**pproach to **P**henotype **LE**arning

MAPLE is a benchmark and modeling framework for the task

```text
(genome, medium)  →  {growth, no growth}
```

It provides the datasets, splits, and controls needed to separate real genome × medium signal.

## Navigation

1. [Motivation](#motivation)
2. [Installation](#installation)
3. [Framework components](#framework-components)
4. [Quick starts](#quick-start)
5. [Project structure](#project-structure)
6. [Documentation](#documentation)

## Motivation

Knowing which culture medium will grow a given strain is a prerequisite for designing media rationally and advance downstream biotechnology applications.

Growth is not a property of the genome alone. It is a **genome × environment interaction**, and learning it from the available data (i.e. BacDive, the largest database on bacterial phenotypes) runs into two problems:

1. **Labels are heavily imbalanced and biased.** Failed growth conditions are rarely reported, so negatives are scarce and concentrated on a few strains and media. A model can score well by memorising which strains or media tend to fail, without learning any biology.
2. **Learning an interaction needs factorial data.** If most strains appear with only one medium, or nearly every record is positive, no model can show that it has learned a genome × medium interaction.

This project contributes with:
- **2 datasets**: **BD-med**, and ML-ready `(genome, medium) → growth` dataset mined from BacDive; and **MAPLE-Lab**, a new, controlled factorial dataset over strains × media × temperature × atmosphere.
- **A benchmark**: through splits that that separates memorisation from transfer

## Installation

1. Clone the repository and enter it:
   ```bash
   git clone https://github.com/lyach/MAPLE.git
   cd MAPLE
   ```

2. Create the environment (Python ≥ 3.11).

   **Using `uv` (recommended)**
   ```bash
   # Install uv (first time only)
   curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux/macOS
   # or
   pip install uv                                    # Windows

   # Create .venv and install the locked dependencies
   uv sync
   source .venv/bin/activate
   ```

## Framework components

| Component | Role | Code | Docs |
|---|---|---|---|
| **BacDive** | Download BacDive, build `(strain, medium) → growth` tables (BD-Med) | `src/bacdive/`, `scripts/bacdive/` | [bacdive.md](docs/bacdive.md) |
| **Experimental dataset** | Parse plate-reader CSVs, apply QC, extract growth-curve parameters and growth labels (MAPLE-Lab) | `src/dataset/`, `scripts/experimental_dataset/` | [dataset_pipeline.md](docs/dataset_pipeline.md) |
| **Strain metadata** | Fetch NCBI | `src/strain_metadata/`, `scripts/external_tools/` | [strain_metadata.md](docs/strain_metadata.md) |
| **Pathway features** | Recover metabolic pathways of strains, to be used as model features | `src/pathway_features/`, `scripts/external_tools/`, `scripts/hpc/` | [modular_pathway_features.md](docs/modular_pathway_features.md) |
| **Modeling** | Build modeling tables, precompute splits, featurise, train, cross-validate, evaluate, and interpret (SHAP) | `src/modeling/`, `scripts/modeling/` | [modeling_pipeline.md](docs/modeling_pipeline.md) |

Every step is driven by YAML configs under `config/`:

```text
BacDive API ─► bacdive cache ─► BD tables ─┐
                                           ├─► (+ strain metadata, pathway features) ─► splits ─► run_experiment ─► reports/<name>/
plate reader ─► QC + curve params ─────────┘
```

## Quick start

### 1. BacDive dataset (BD-med)

```bash
# 1. Download BacDive records
python scripts/bacdive/get_bacdive_dataset.py --taxa bacdive/all_taxa_medium.yaml

# 2. Build the modeling table
python scripts/bacdive/build_bacdive_from_cache.py --config bacdive/build_from_cache.yaml

# 3. Get modeling table. Filter to culture media labels only (bd_temp → bd_med), canonicalise media names, filter to strains with a genome
python scripts/bacdive/filter_bacdive_to_strain_media.py
python scripts/bacdive/add_media_canon.py
python scripts/bacdive/filter_bacdive_with_genome.py --parquet data/processed/bacdive/bd_med.parquet
```


### 2. Factorial dataset (MAPLE-Lab)

```bash
# Raw CSVs → Tidy curves
python scripts/experimental_dataset/run_dataset_pipeline.py aerobic

# Curves + strain metadata → modeling table
python scripts/modeling/create_modeling_dataset.py config/dataset/curves/curves_dataset_aero_anaero.yaml
```

### 3. Splits and experiments

```bash
# Precompute a reusable group → fold map (here: held-out medium, grouped on media_canon)
python scripts/modeling/create_splits.py config/splits/bd_media.yaml

# Train and evaluate
python scripts/modeling/run_experiment.py config/experiments/bacdive/bd_med_w_genome_xgb_smedia.yaml
python scripts/modeling/run_experiment.py config/experiments/align/class_aero_anaero.yaml
```

Each run writes to `reports/<experiment_name>/`: logs, metrics, and SHAP analysis.


### 4. Utils

```bash
python scripts/utils/paper/create_distribution_table.py --format all   # dataset profiles
python scripts/utils/paper/create_interaction_table.py                 # single-factor flip rates
python scripts/utils/paper/compile_results_csv.py                      # reports/*/ → reports/all_results.csv
```

## Project structure

```
bacdive/                     ← BacDive configs

config/
  dataset/
    curves/                  ← Curves configs for create_modeling_dataset.py (MAPLE-Lab)
    modeling_align/          ← MAPLE-Lab modeling dataset configs
    pathways/                ← Pathway feature configs
  experiments/
    bacdive/                 ← BD-med / BD-temp experiments (bd_<corpus>_xgb_<split>[_ablation].yaml)
    align/                   ← MAPLE-Lab experiments
  models/                    ← Model hyperparameters
  splits/                    ← Precomputed split configs (genus, family, medium)

data/
  raw/                       ← Plate-reader CSVs
  metadata/                  ← Experiment metadata, plate layouts, NCBI strain lists
  processed/                 ← QC'd curves and built BacDive tables (bd_med*, bd_temp*)
  interim/                   ← Merged modeling tables
  external/                  ← BacDive cache, genomes, recipe manifest, alias map, pathway outputs
  modeling/                  ← Saved split maps (splits_<dataset>/)

src/
  bacdive/                   ← BacDive client, extraction, medium helpers, build pipeline
  dataset/                   ← Plate-reader parsers, QC, growth-curve parameters
  strain_metadata/           ← NCBI genome / taxonomy resolution
  pathway_features/          ← Manifest, KEGG-module completeness, CheckM2 quality
  modeling/                  ← Features, models, CV, splits, evaluation, interpretability
  common/                    ← Shared logging setup

scripts/
  bacdive/                   ← BacDive download, build, filtering, medium review
  experimental_dataset/      ← MAPLE-Lab dataset pipeline
  external_tools/            ← Strain metadata, genome download, pathway features
  modeling/                  ← create_modeling_dataset, create_splits, run_experiment
  utils/                     ← Diagnostic tables
  hpc/, slurm/               ← Cluster job scripts

databases/                   ← KOfam and CheckM2 databases
tests/                       ← Regression tests (media_canon)
notebooks/                   ← EDA and modeling notebooks
docs/                        ← Component documentation
reports/                     ← Experiment outputs
```

## Documentation

- [BacDive pipeline](docs/bacdive.md)
- [Experimental dataset pipeline](docs/dataset_pipeline.md)
- [Strain metadata](docs/strain_metadata.md)
- [Pathway features](docs/modular_pathway_features.md)
- [Modeling pipeline](docs/modeling_pipeline.md)

