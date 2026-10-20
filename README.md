# MAPLe: A **M**ultimodal **A**pproach to **P**henotype **LE**arning

MAPLe is a benchmark and modeling framework for the task:

```text
(organism, environment)  -->  {growth, no growth}
```

It provides the datasets, splits, and controls needed to separate real organism × environment signal.

## Navigation

1. [Motivation](#motivation)
2. [Installation](#installation)
3. [Framework components](#framework-components)
4. [Quick starts](#quick-start)
5. [Project structure](#project-structure)
6. [Documentation](#documentation)

## Motivation

Knowing which environment (e.g., the culture media, temperature, atmosphere) will allow an organism to grow, can accelerate the design of experiments in microbiology.

However, growth is not a property of an organism's phenotype alone. It is a **phenotype × environment interaction**, and learning it from the available data (i.e. BacDive, the largest database on bacterial phenotypes) runs into two problems:

1. **Labels are imbalanced and biased.** Failed growth conditions are rarely reported, so negatives are scarce and concentrated on a few organisms and media. A model can score well by memorising which organism or media tend to fail, without learning any biology.
2. **Learning an interaction needs factorial data.** If most organisms appear with only one medium, or nearly every record is positive, no model can show that it has learned a genome × medium interaction.

This project contributes with:
- **2 datasets**: **BD-med**, an ML-ready `(genome, medium) --> growth` dataset mined from BacDive; and **MAPLe-Lab**, a new experimental dataset covering growth labels for 25 strains, tested across 11 media × 2 temperatures × 2 atmospheres.
- **A benchmark**: splits that hold out organisms, or media, so a model has to generalise the organism × environment interaction instead of memorising which strains/recipes tend to fail.

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
| **BacDive dataset** | Download BacDive, build `(strain, medium) → growth` tables (BD-Med) | `src/bacdive/`, `scripts/bacdive/` | [bacdive.md](docs/bacdive.md) |
| **Experimental dataset** | Parse plate-reader CSVs, apply QC, extract growth-curve parameters and growth labels (MAPLe-Lab) | `src/dataset/`, `scripts/experimental_dataset/` | [dataset_pipeline.md](docs/dataset_pipeline.md) |
| **Strain metadata** | Recover metadata from NCBI | `src/strain_metadata/`, `scripts/external_tools/` | [strain_metadata.md](docs/strain_metadata.md) |
| **Pathway features** | Recover metabolic pathways of strains, to be used as model features | `src/pathway_features/`, `scripts/external_tools/`, `scripts/hpc/` | [modular_pathway_features.md](docs/modular_pathway_features.md) |
| **Modeling** | Build modeling tables, precompute splits, featurise, train, cross-validate, evaluate, and interpret (SHAP) | `src/modeling/`, `scripts/modeling/` | [modeling_pipeline.md](docs/modeling_pipeline.md) |

The overview:

```text
BacDive API ─► bacdive cache ─► BD tables ─┐
                                           ├─► (+ strain metadata, pathway features) ─► splits ─► run_experiment ─► reports/<name>/
plate reader ─► QC + curve params ─────────┘
```

## Quick start

### 1. Build the BacDive dataset (BD-med)

```bash
# 1. Download BacDive records
python scripts/bacdive/get_bacdive_dataset.py --taxa bacdive/all_taxa_medium.yaml

# 2. Build the modeling table
python scripts/bacdive/build_bacdive_from_cache.py --config bacdive/build_from_cache.yaml

# 3. Get modeling table: filter to culture media labels, canonicalise media names, filter to strains with a genome
python scripts/bacdive/filter_bacdive_to_strain_media.py
python scripts/bacdive/add_media_canon.py
python scripts/bacdive/filter_bacdive_with_genome.py --parquet data/processed/bacdive/bd_med.parquet
```


### 2. Factorial dataset (MAPLe-Lab)
*Note: Not currently available. To be released.*
```bash
# Raw CSVs → Tidy curves
python scripts/experimental_dataset/run_dataset_pipeline.py aerobic

# Curves + strain metadata → modeling table
python scripts/modeling/create_modeling_dataset.py config/dataset/curves/curves_dataset_aero_anaero.yaml
```

### 3. Splits and experiments

```bash
# Get reusable split group (here: held-out medium for BD-Temp)
python scripts/modeling/create_splits.py config/splits/bd_media.yaml

# Train and evaluate
python scripts/modeling/run_experiment.py config/experiments/bacdive/bd_med_w_genome_xgb_smedia.yaml
```

Each run writes to `reports/<experiment_name>/`: logs, metrics, and SHAP analysis.


### 4. Utils

```bash
python scripts/utils/paper/create_distribution_table.py --format all   # dataset profiles
python scripts/utils/paper/compile_results_csv.py                      # aggregate all metrics under reports/*/
```

## Project structure

```
bacdive/                     ← BacDive configs

config/
  dataset/
    curves/                  ← Curves configs for create_modeling_dataset.py (MAPLe-Lab)
    modeling_align/          ← MAPLe-Lab modeling dataset configs
    pathways/                ← Pathway feature configs
  experiments/
    bacdive/                 ← BD-med / BD-temp experiments (bd_<corpus>_xgb_<split>[_ablation].yaml)
    align/                   ← MAPLe-Lab experiments
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
  experimental_dataset/      ← MAPLe-Lab dataset pipeline
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

