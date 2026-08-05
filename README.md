# MicrobeNet

**Predict microbial growth from genotype and media.**

MicrobeNet is a machine learning framework for modeling the genotype–phenotype relationship in microbiology. Given a microbe and a cultivation medium, the model predicts growth outcomes.

## Motivation

Microbial diversity remains largely underutilized because phenotype data linking genotype to growth is scarce and inconsistent. Robust predictive models of organism–environment interactions would unlock applications in biomanufacturing, climate science, and agriculture. This project focuses on the modeling layer that makes those predictions possible. The core idea:

```text
(microbe features, media / condition features)  →  growth phenotype (growth/no growth)
```

## Getting Started

0. Install [Git](https://github.com/git-guides/install-git)

1. Clone the repository:
   ```bash
   git clone https://github.com/lyach/MicrobeNet.git
   ```

2. Navigate to the project directory:
   ```bash
   cd MicrobeNet
   ```

3. Create an isolated virtual environment:

   **Using `uv` (recommended)**
   ```bash
   # Install uv (if first time)
   curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux/macOS
   # or
   pip install uv   # Windows

   # Create environment
   uv venv --python 3.11 .venv

   # Activate environment
   source .venv/bin/activate   # Linux/macOS
   # or
   .venv\Scripts\activate      # Windows

   # Install dependencies
   uv sync
   ```

## Components

| Component | Role |
|---|---|
| **Dataset** | Load plate-reader growth curves, apply QC, and derive growth parameters (`src/dataset/`) |
| **Strain metadata** | Resolve taxonomy and genome-linked strain identity (`src/strain_metadata/`) |
| **Pathway features** | Build modular KEGG / KO pathway completeness features from genomes (`src/pathway_features/`) |
| **Modeling** | Featurize microbe + media, train models, cross-validate, and interpret predictions (`src/modeling/`) |

The modeling pipeline merges growth-curve metrics with strain and pathway features, then runs configurable experiments (classification or regression) via YAML configs under `config/`. See [docs/modeling_pipeline.md](docs/modeling_pipeline.md) and [docs/dataset_pipeline.md](docs/dataset_pipeline.md).

### Quick start: modeling

```bash
# 1. Build interim modeling dataset
python scripts/create_modeling_dataset.py config/dataset/modeling_dataset_aero_anaero.yaml

# 2. Train / evaluate an experiment
python scripts/run_experiment.py config/experiments/class_aero_anaero.yaml
```

Results are written under `reports/<experiment_id>/` (metrics, predictions, SHAP summaries).

## Project Structure

```
config/
  dataset/                 ← Modeling dataset configs
  experiments/             ← Experiment configs (task, CV, metrics)
  models/                  ← Model hyperparameters (XGBoost, RF, MLP, linear)

data/
  metadata/                ← Experiment metadata and plate layouts
  raw/                     ← Plate-reader CSVs
  processed/               ← QC'd curves ready for analysis
  interim/                 ← Merged modeling datasets
  external/                ← Strain metadata and external features

src/
  dataset/                 ← Parsers, QC, growth-curve parameters
  strain_metadata/         ← Taxonomy / genome metadata pipeline
  pathway_features/        ← Modular pathway feature construction
  modeling/                ← Features, models, CV, evaluation, interpretability
  common/                  ← Shared utilities

scripts/
  run_dataset_pipeline.py
  create_modeling_dataset.py
  run_experiment.py
  ...

notebooks/                 ← EDA and modeling notebooks
docs/                      ← Pipeline documentation
reports/                   ← Experiment outputs
```
