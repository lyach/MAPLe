# Modeling Pipeline

## Overview

This pipeline processes growth-curve parameters and strain metadata into machine learning datasets to execute three prediction tasks:

- **Growth Classification** (binary/multiclass growth vs. no growth)
- **Maximum $\text{OD}_{600}$ Regression** (yield prediction)
- **Growth-Rate Regression** ($\mu_{\max}$ prediction)

Execution is split into two primary CLI steps, driven by three configuration files (`dataset`, `model`, and `experiment` YAMLs). For interactive workflows, all core steps can also be imported directly from `src/modeling/`.

### Pipeline

```text

curves.csv ──────┐
                 ├──► [ create_modeling_dataset.py ] ──► data/interim/modeled_dataset.csv
metadata.csv ────┘                                                      │
                                                                        │
config/                                                                 ▼
├── dataset/     ──────────────────────────────────────────► [ run_experiment.py ]
├── models/
└── experiments/                                                        │
                                                                        ▼
                                                          reports/<experiment_id>/
                                                          ├── metrics.json
                                                          ├── predictions.csv
                                                          └── shap_feature_importance.png

```

---

### Pipeline Steps

1. `create_modeling_dataset.py`: 
Merges growth-curve metrics with strain metadata, handles replicate averaging/filtering based on configuration, and exports a unified interim dataset.

2. `run_experiment.py`: 
Ingests the interim dataset and YAML configs to handle feature engineering, model training, cross-validation, and SHAP interpretability analysis.

## 1. Prerequisites

Before running the modeling pipeline, you need:


| Input               | Source                                                                          | Required |
| ------------------- | ------------------------------------------------------------------------------- | -------- |
| Curves CSV          | `data/processed/*_tidy_curves.csv` from [dataset pipeline](dataset_pipeline.md) | Yes      |
| Strain metadata CSV | `data/external/strains_metadata.csv` (via `get_strain_metadata.py`)             | No       |
| Dataset config      | `config/dataset/modeling_dataset_*.yaml`                                        | Yes      |
| Model config        | `config/models/*.yaml`                                                          | Yes      |
| Experiment config   | `config/experiments/*.yaml`                                                     | Yes      |


The interim modeling dataset must be created **before** running an experiment. `run_experiment.py` reads from `data/interim/<output_name>` as specified in the dataset config.

---

## 2. Create the modeling dataset

### 2a. Dataset config — `config/dataset/modeling_dataset_*.yaml`


| Field                  | Type                         | Description                                                                                                                |
| ---------------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `curves_file`          | str or list[str]             | One or more curves CSV paths; multiple files are concatenated                                                              |
| `strain_metadata_file` | str                          | Strain metadata CSV                                                                                                        |
| `output_name`          | str                          | Output filename under `data/interim/`                                                                                      |
| `grain`                | `"per_well"` or `"averaged"` | Datapoint (row) granularity (see [2c](#2c-merge-and-grain-srcmodelingdatasetpy))                                           |
| `merge.left`           | str                          | Join key in the curves table                                                                                               |
| `merge.right`          | str                          | Join key in the metadata table                                                                                             |
| `features`             | dict                         | Features to include and their encoding spec (used by `run_experiment.py`; see [6](#6-featurization-srcmodelingfeaturespy)) |
| `targets`              | list[str]                    | Available prediction targets                                                                                               |


### 2b. Run the script

```bash
python scripts/create_modeling_dataset.py config/dataset/modeling_dataset_aero_anaero.yaml
```

### 2c. Merge and grain

`build_modeling_dataset` performs three operations:

1. **Load curves** — reads one or more curves CSVs and concatenates them.
2. **Merge strain metadata** — left-joins curves CSV with strain metadata columns.
3. **Replicate averaging** (optional) — controlled by `grain`:


| `grain`                  | One row =                                    | Replicate handling                                        |
| ------------------------ | -------------------------------------------- | --------------------------------------------------------- |
| `"per_well"` *(default)* | one (experiment, well)                       | keeps `exp_id`, `date`, `replicate`                       |
| `"averaged"`             | one (strain, media, atmosphere, temperature) | continuous targets averaged; `growth_class` majority vote |


When `grain: "averaged"`, rows are grouped by `(strain_name, media_name, atmosphere, temperature)`:

- Continuous parameters (`max_od`, `auc`, `growth_rate`, `lag_time_h`, `time_to_max_od_h`) are **averaged**.
- `growth_class` uses **majority vote**: `1` if mean ≥ 0.5.

### 2d. Interim output

The output is a CSV at `data/interim/<output_name>`.

**One row = one (experiment, well) datapoint** when `grain: "per_well"`.  
**One row = one (strain, media, condition) average** when `grain: "averaged"`.

The table retains all columns from the curves CSV plus any merged metadata columns.

---

## 3. Run an experiment

### 3a. Experiment config — `config/experiments/*.yaml`


| Field                        | Type             | Example                                       | Description                                       |
| ---------------------------- | ---------------- | --------------------------------------------- | ------------------------------------------------- |
| `name`                       | str              | `"aerobic_xgboost_growth_class"`              | Experiment name; used as output subdirectory      |
| `dataset`                    | str              | `"config/dataset/modeling_dataset_aero.yaml"` | Path to dataset config                            |
| `model`                      | str              | `"config/models/xgboost.yaml"`                | Path to model config                              |
| `task`                       | str              | `"growth_class"`                              | Prediction target                                 |
| `cv.strategy`                | str or list[str] | `[loso, lomo]`                                | CV strategy                                       |
| `cv.n_repeats`               | int              | `1`                                           | Number of CV repeats                              |
| `metrics`                    | list[str]        | `[auroc, f1, mcc]`                            | Metrics to compute                                |
| `interpret.shap`             | bool             | `true`                                        | Run SHAP analysis after CV                        |
| `interpret.group_importance` | bool             | `true`                                        | Compute strain/media/condition grouped importance |
| `seed`                       | int              | `42`                                          | Random seed                                       |
| `output_dir`                 | str              | `"reports"`                                   | Base output directory                             |
| `wandb.enabled`              | bool             | `false`                                       | Weights & Biases logging (placeholder)            |


Example:

```yaml
name: aerobic_xgboost_growth_class
dataset: config/dataset/modeling_dataset_aero.yaml
model: config/models/xgboost.yaml
task: growth_class

cv:
  strategy: [loso, lomo]
  n_repeats: 1

metrics: [auroc, f1, mcc, balanced_accuracy]

interpret:
  shap: true
  group_importance: true

seed: 42
output_dir: reports
```

### 3b. Model config — `config/models/*.yaml`


| Field            | Type | Description                                                 |
| ---------------- | ---- | ----------------------------------------------------------- |
| `type`           | str  | One of `linear`, `random_forest`, `xgboost`, `mlp`, `dummy` |
| `params`         | dict | Hyperparameters passed to the estimator                     |
| `task_overrides` | dict | Per-task parameter overrides (keyed by task name)           |


Available model configs:


| File                 | Classifier               | Regressor               |
| -------------------- | ------------------------ | ----------------------- |
| `xgboost.yaml`       | `XGBClassifier`          | `XGBRegressor`          |
| `linear.yaml`        | `LogisticRegression`     | `Ridge`                 |
| `random_forest.yaml` | `RandomForestClassifier` | `RandomForestRegressor` |
| `mlp.yaml`           | `MLPClassifier`          | `MLPRegressor`          |


The `dummy` type (most-frequent / mean) is available in code but has no config file yet.

### 3c. Run the script

```bash
python scripts/run_experiment.py config/experiments/aerobic_xgboost.yaml
```

The script:

1. Loads and resolves the experiment, dataset, and model configs.
2. Reads the interim CSV from `data/interim/<output_name>`.
3. Filters rows for the task (e.g. `growth_rate` keeps only `growth_class == 1`).
4. Runs CV: for each fold, fits the feature pipeline on train, transforms train/test, fits the model, predicts, and evaluates.
5. Aggregates metrics across folds (mean ± std per strategy).
6. Refits on all data and runs SHAP interpretability (if enabled).
7. Saves everything to `reports/<name>/`.

---

## 4. Prediction tasks


| Task           | Type                  | Target column  | Notes                               |
| -------------- | --------------------- | -------------- | ----------------------------------- |
| `growth_class` | Binary classification | `growth_class` | `1` = growth, `0` = no growth       |
| `max_od`       | Regression            | `max_od`       | All wells                           |
| `growth_rate`  | Regression            | `growth_rate`  | Only wells with `growth_class == 1` |


To run a different task, change `task` in the experiment config and re-run. The same interim dataset and model config can be reused.

---

## 5. Cross-validation strategies (`src/modeling/cv.py`)


| Strategy             | Splitter               | Group column  | Folds                    | Use case                        |
| -------------------- | ---------------------- | ------------- | ------------------------ | ------------------------------- |
| `loso`               | `LeaveOneGroupOut`     | `strain_name` | one per strain           | Generalization to new organisms |
| `lomo`               | `LeaveOneGroupOut`     | `media_name`  | one per medium           | Generalization to new media     |
| `grouped`            | `GroupKFold`           | `strain_name` | configurable (default 5) | Replicates stay together        |
| `stratified_grouped` | `StratifiedGroupKFold` | `strain_name` | configurable (default 5) | Stratified + grouped            |


Multiple strategies can be specified as a list in the experiment config; each is evaluated independently.

#### LOSO / LOMO warning

- Under Leave-One-Strain-Out, one-hot encoded `strain_name` produces an all-zero vector for the held-out strain, which carries no signal. Meaningful LOSO generalization requires strain features that transfer across organisms (genome size, GC%, taxonomy, genome embeddings). 
- The same applies to `media_name` under LOMO. 
- The pipeline logs a warning when identity one-hot is the only strain/media feature under these strategies.

---

## 6. Featurization (`src/modeling/features.py`)

Feature encoding is defined in the dataset config's `features` dict and applied inside CV — the transformer is **fit on the train fold only** to prevent leakage.


| Spec            | Encoder          | Description                                                       |
| --------------- | ---------------- | ----------------------------------------------------------------- |
| `"OH"`          | `OneHotEncoder`  | One-hot encoding; `handle_unknown="ignore"` for unseen categories |
| `"ordinal"`     | `OrdinalEncoder` | Integer encoding; unknown values → `-1`                           |
| `"numeric"`     | `StandardScaler` | Z-score normalization                                             |
| `"path:<file>"` | passthrough      | Precomputed embedding CSV (first column = key, rest = floats)     |


Each raw column is mapped to a semantic group for interpretability:


| Column                      | Group                      |
| --------------------------- | -------------------------- |
| `strain_name`               | `strain`                   |
| `media_name`                | `media`                    |
| `atmosphere`, `temperature` | `condition`                |
| genome metadata columns     | `genome`                   |
| embedding columns           | `strain_emb` / `media_emb` |


**Notebook / EDA use only** — `featurize()` fits on all data and transforms in one call. Do not use its output for CV reporting:

```python
from src.modeling.features import featurize

X, y, feature_names, groups, fgm = featurize(df, cfg, target="growth_class")
```

---

## 7. Evaluation metrics (`src/modeling/evaluate.py`)

**Classification** (`growth_class`):


| Metric               | Key                 |
| -------------------- | ------------------- |
| AUROC                | `auroc`             |
| PR-AUC               | `pr_auc`            |
| F1 score             | `f1`                |
| Matthews correlation | `mcc`               |
| Balanced accuracy    | `balanced_accuracy` |


**Regression** (`max_od`, `growth_rate`):


| Metric    | Key         |
| --------- | ----------- |
| RMSE      | `rmse`      |
| R²        | `r2`        |
| Pearson r | `pearson_r` |
| MAE       | `mae`       |


Metrics are computed per fold, then aggregated (mean ± std) per CV strategy. Per-strain and per-media breakdowns are saved separately.

---

## 8. Interpretability (`src/modeling/interpret.py`)

After CV, the model is refit on all data and SHAP values are computed:


| Model type                 | SHAP explainer                            |
| -------------------------- | ----------------------------------------- |
| XGBoost, Random Forest     | `TreeExplainer`                           |
| Logistic Regression, Ridge | `LinearExplainer`                         |
| MLP, other                 | `KernelExplainer` (subsampled background) |


Outputs saved to `reports/<name>/figures/`:


| File                   | Description                                                            |
| ---------------------- | ---------------------------------------------------------------------- |
| `shap_beeswarm.png`    | Beeswarm plot of feature-level SHAP values                             |
| `shap_bar.png`         | Mean                                                                   |
| `group_importance.png` | Aggregated importance by feature group (strain, media, condition, ...) |


---

## 9. Experiment output

Each run writes to `reports/<experiment_name>/`:

```
reports/aerobic_xgboost_growth_class/
  config.yaml              ← snapshot of all three configs + timestamp
  experiment.log           ← full run log
  fold_metrics.csv         ← per-fold metrics (strategy, fold, fold_label, metrics)
  metrics.json             ← aggregate mean ± std per CV strategy
  predictions.csv          ← y_true, y_pred, y_prob, strategy, fold, strain, media
  metrics_by_strain_name.csv
  metrics_by_media_name.csv
  figures/
    shap_beeswarm.png
    shap_bar.png
    group_importance.png
```

---

## 10. Source code reference

All modules live under `src/modeling/` and are importable from notebooks:


| Module         | Key functions                                                                              | Role                                                  |
| -------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------------- |
| `config.py`    | `load_dataset_config`, `load_model_config`, `load_experiment_config`, `resolve_experiment` | Pydantic config models + YAML loading                 |
| `dataset.py`   | `build_modeling_dataset`                                                                   | Merge curves + metadata, optional replicate averaging |
| `features.py`  | `build_feature_pipeline`, `featurize`, `warn_loso_onehot`                                  | Leakage-safe sklearn `ColumnTransformer`              |
| `models.py`    | `build_model`                                                                              | Model registry (linear, RF, XGBoost, MLP, dummy)      |
| `cv.py`        | `make_splits`, `get_fold_group_label`                                                      | LOSO, LOMO, grouped, stratified-grouped splitters     |
| `evaluate.py`  | `evaluate_fold`, `aggregate_metrics`, `per_group_metrics`                                  | Classification and regression metrics                 |
| `interpret.py` | `compute_shap_values`, `run_interpretation`                                                | SHAP + grouped feature importance plots               |


**Scripts:**


| Script                               | Usage                          |
| ------------------------------------ | ------------------------------ |
| `scripts/create_modeling_dataset.py` | Step 1: build interim CSV      |
| `scripts/run_experiment.py`          | Step 2: CV + evaluation + SHAP |


**Notebooks:**


| Notebook                           | Contents                                          |
| ---------------------------------- | ------------------------------------------------- |
| `notebooks/02_featurization.ipynb` | Build dataset → inspect feature matrix and groups |
| `notebooks/03_modeling.ipynb`      | Step-by-step CV fold, evaluation, SHAP            |


---

## 11. Adding new features

1. Add the column to the curves CSV or strain metadata CSV (no code change for metadata columns merged in step 1).
2. Add an entry to the `features` dict in the dataset config:

```yaml
features:
  strain_name: "OH"
  genome_size_bp: "numeric"       # from strain metadata
  genus: "ordinal"                # from strain metadata
  strain_name: "path:data/external/evo2_embeddings.csv"  # precomputed
```

1. Re-run `create_modeling_dataset.py` (if new columns come from metadata) and `run_experiment.py`.

For precomputed embeddings, the CSV must have the entity name as the first column and numeric embedding dimensions as subsequent columns.

---

## 12. Adding a new model

1. Create `config/models/<name>.yaml`:

```yaml
type: random_forest
params:
  n_estimators: 300
  max_depth: 10
task_overrides:
  growth_class:
    class_weight: balanced
```

1. Reference it in an experiment config:

```yaml
model: config/models/<name>.yaml
```

To register a new model **type** (not just hyperparameters), add an entry to `_REGISTRY` in `src/modeling/models.py`.

---

## 13. Pluggable feature providers (not yet implemented)

These scripts produce inputs that slot into the modeling pipeline via config paths:


| Script                   | Output                               | Config hook                              |
| ------------------------ | ------------------------------------ | ---------------------------------------- |
| `get_strain_metadata.py` | `data/external/strains_metadata.csv` | `strain_metadata_file` in dataset config |
| `get_evo_embeddings.py`  | genome embedding CSV                 | `path:<file>` in `features` dict         |
| `get_ecfp_embeddings.py` | media fingerprint CSV                | `path:<file>` in `features` dict         |


No changes to `src/modeling/` are needed to use their outputs — only config updates.