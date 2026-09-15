# Modeling Pipeline

## Overview

This pipeline processes growth-curve parameters and strain metadata into machine learning datasets to execute three prediction tasks:

- **Growth Classification**
- **Maximum $\text{OD}_{600}$ Regression**
- **Growth-Rate Regression**

Execution is split into two primary CLI steps, driven by three configuration files (`curves`, `model`, and `experiment` YAMLs). The experiment config carries its own dataset block, including: the table path, features, and targets live directly in it. For interactive workflows, all core steps can also be imported directly from `src/modeling/`.

### Pipeline

```text

curves.csv ──────┐
                 ├──► [ create_modeling_dataset.py ] ──► data/interim/modeled_dataset.csv
metadata.csv ────┘     (config/dataset/curves_dataset_*.yaml)
                                                                        │
processed / interim table ──► [ create_splits.py ] ──► data/modeling/splits_<stem>/
       ▲ (optional, reusable CV splits)              <split_on>_<stem>.csv
config/                                                    │            │
├── dataset/curves/  curves_dataset_*.yaml (step 1 only)   │ (optional) │
├── models/      ──────────────────────────────────────────────► [ run_experiment.py ]
├── experiments/  (table path + features + targets inline)              │
└── splits/                                                             ▼
                                                          reports/<experiment_id>/
                                                          ├── metrics.json
                                                          ├── predictions.csv
                                                          └── shap_feature_importance.png

```

---

### Pipeline Steps

1. `create_modeling_dataset.py`:
Merges growth-curve metrics with strain metadata, handles replicate averaging/filtering based on a **curves** config, and exports a unified table (typically under `data/interim/`).

2. `run_experiment.py`:
Ingests the table at `modeling_dataset_path` and YAML configs to handle feature engineering, model training, cross-validation, and SHAP interpretability analysis.

## 1. Prerequisites

Before running the modeling pipeline, you need:


| Input               | Source                                                                          | Required |
| ------------------- | ------------------------------------------------------------------------------- | -------- |
| Curves table         | `data/processed/*_tidy_curves.csv` or `.parquet` from [dataset pipeline](dataset_pipeline.md) / external sources | Yes (for step 1) |
| Strain metadata table | `data/external/strains_metadata.csv` (via `get_strain_metadata.py`)                                             | No       |
| Curves config       | `config/dataset/curves_dataset_*.yaml`                                          | Yes (for step 1) |
| Modeling dataset config | `config/dataset/modeling_dataset_*.yaml` | No       |
| Model config        | `config/models/*.yaml`                                                          | Yes      |
| Experiment config   | `config/experiments/*.yaml`                                                     | Yes      |


`run_experiment.py` reads the table at `modeling_dataset_path` in the experiment config. That path may be an interim file produced by `create_modeling_dataset.py`, or an already-processed external table (e.g. BacDive parquet). When using precomputed splits, this path must be the **same table** (same row count) that `create_splits.py` was run on.

---

## 2. Create the modeling dataset

### 2a. Curves config — `config/dataset/curves_dataset_*.yaml`


| Field                  | Type                         | Description                                                                                                                |
| ---------------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `curves_file`          | str or list[str]             | One or more curves CSV/Parquet paths; multiple files are concatenated                                                      |
| `strain_metadata_file` | str                          | Strain metadata CSV/Parquet                                                                                                |
| `output_name`          | str                          | Output filename under `data/interim/`                                                                                      |
| `grain`                | `"per_well"` or `"averaged"` | Datapoint (row) granularity (see [2c](#2c-merge-and-grain-srcmodelingdatasetpy))                                           |
| `merge.left`           | str                          | Join key in the curves table                                                                                               |
| `merge.right`          | str                          | Join key in the metadata table                                                                                             |


### 2b. Run the script

```bash
python scripts/modeling/create_modeling_dataset.py config/dataset/curves_dataset_aero_anaero.yaml
```

### 2c. Merge and grain

`build_modeling_dataset` performs three operations:

1. **Load curves** — reads one or more curves CSV/Parquet files and concatenates them.
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

The output is a CSV or Parquet file at `data/interim/<output_name>` (format follows the file extension).

**One row = one (experiment, well) datapoint** when `grain: "per_well"`.  
**One row = one (strain, media, condition) average** when `grain: "averaged"`.

The table retains all columns from the curves CSV plus any merged metadata columns.

### 2e. Which table an experiment reads

The table an experiment runs on, and the features it builds from that table, are declared **in the experiment config itself** — see [3b](#3b-dataset-block). There is no separate modeling dataset file to keep in sync.

> **Legacy.** An experiment config may instead point at a standalone dataset YAML with `dataset: config/dataset/modeling_dataset_*.yaml`, holding the same `modeling_dataset_path` / `features` / `targets` / `filters` keys. This still loads, but the two forms are mutually exclusive: declaring both is an error. To merge an old pair into one file, run
> `python scripts/utils/migrate_dataset_into_experiment.py <config-or-dir> --apply`.

---

## 3. Run an experiment

### 3a. Experiment config — `config/experiments/*.yaml`


| Field                        | Type             | Example                                       | Description                                       |
| ---------------------------- | ---------------- | --------------------------------------------- | ------------------------------------------------- |
| `name`                       | str              | `"aerobic_xgboost_growth_class"`              | Experiment name; used as output subdirectory      |
| `modeling_dataset_path`      | str              | `"data/processed/bacdive/bd_temp.parquet"`    | CSV/Parquet table to load (see [3b](#3b-dataset-block)) |
| `features`                   | dict             | see [3b](#3b-dataset-block)                   | Features to include and their encoding spec       |
| `targets`                    | list[str]        | `[growth_class]`                              | Available prediction targets                      |
| `filters`                    | dict (optional)  | see [3b](#3b-dataset-block)                   | Row-level `include` / `exclude` rules, keyed by column |
| `dataset`                    | str (legacy)     | `"config/dataset/modeling_dataset_aero.yaml"` | Path to a standalone dataset config, instead of the four keys above |
| `model`                      | str              | `"config/models/xgboost.yaml"`                | Path to model config                              |
| `task`                       | str              | `"growth_class"`                              | Prediction target                                 |
| `cv.strategy`                | str or list[str] | `[loso, lomo]`                                | CV strategy                                       |
| `cv.n_repeats`               | int              | `1`                                           | Number of CV repeats                              |
| `cv.splits_file`             | str (optional)   | `"data/modeling/splits_bacdive_growth_150k/genus_bacdive_growth_150k.csv"` | Consume precomputed splits (see [5a](#5a-precomputed-reusable-splits-create_splitspy)) |
| `cv.split_on`                | str (optional)   | `"genus"`                                     | Group column the saved split map is keyed on (required with `cv.splits_file`) |
| `class_weight`               | str or dict      | `"balanced"`                                  | Per-sample class reweighting (see [3e](#3e-class-imbalance-handling-reweighting)) |
| `metrics`                    | list[str]        | `[auroc, f1, mcc]`                            | Metrics to compute                                |
| `interpret.shap`             | bool             | `true`                                        | Run SHAP analysis after CV                        |
| `interpret.group_importance` | bool             | `true`                                        | Compute strain/media/condition grouped importance |
| `interpret.shap_class`       | str              | `"positive"`                                  | For binary classifiers: `"positive"` or `"negative"` class SHAP |
| `seed`                       | int              | `42`                                          | Random seed                                       |
| `output_dir`                 | str              | `"reports"`                                   | Base output directory                             |
| `wandb.enabled`              | bool             | `false`                                       | Weights & Biases logging (placeholder)            |


Example:

```yaml
# ================ EXPERIMENT ================
name: aerobic_xgboost_growth_class
model: config/models/xgboost.yaml
task: growth_class

cv:
  strategy: [loso, lomo]
  n_repeats: 1

class_weight: balanced

metrics: [auroc, f1, mcc, balanced_accuracy]

interpret:
  shap: true
  group_importance: true
  shap_class: positive   # or negative; ignored for regression

seed: 42
output_dir: reports

# ================ DATASET ================
modeling_dataset_path: data/external/bacdive_growth_150k.parquet

# ================ FEATURIZATION ================
features:
  genus: 'OneHotEncoder(sparse_output=False, handle_unknown="ignore")'
  complexity: 'OneHotEncoder(sparse_output=False, handle_unknown="ignore")'

targets:
  - growth_class
```

### 3b. Dataset block

These keys name the table to model and how to featurize it. They live in the experiment config alongside the fields above; the banner comments are convention, not syntax.


| Field                    | Type      | Description                                                                                       |
| ------------------------ | --------- | ------------------------------------------------------------------------------------------------- |
| `modeling_dataset_path`  | str       | CSV/Parquet path to load (interim output of step 1, or an already-processed external table)       |
| `features`               | dict      | Features to include and their encoding spec (see [6](#6-featurization-srcmodelingfeaturespy))     |
| `targets`                | list[str] | Available prediction targets                                                                      |
| `filters`                | dict      | Optional row-level `include` / `exclude` rules, keyed by column (see below)                        |


When using precomputed splits, `modeling_dataset_path` must be the **same table** (same row count) that `create_splits.py` was run on.

#### Row filters — `filters`

For ablations that remove a *rows* rather than a feature. Each entry
is `column: [value, ...]`; `exclude` drops matching rows, `include` keeps only
matching rows, and multiple columns are ANDed. Values are compared as strings,
so YAML quoting cannot change the result.

```yaml
filters:
  exclude:
    media_canon: [macconkey]
```

Filters are applied **after** the split map row count, so a filtered run reuses the unfiltered split map and its folds stay directly comparable to the baseline (folds match by group value, not row position). The run logs how many rows and how many minority-class rows were dropped, and warns if an `exclude` list matched nothing. Filtering away every row is an error.

### 3c. Model config — `config/models/*.yaml`


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

### 3d. Run the script

```bash
python scripts/modeling/run_experiment.py config/experiments/class_aero.yaml
```

The script:

1. Loads and resolves the experiment, modeling-dataset, and model configs.
2. Reads the CSV/Parquet at `modeling_dataset_path`.
3. If `cv.splits_file` is set, checks that this table has the same row count as the split-source table recorded in the companion `*_stats.json`.
4. Filters rows for the task (e.g. `growth_rate` keeps only `growth_class == 1`).
5. Runs CV: for each fold, fits the feature pipeline on train, transforms train/test, fits the model, predicts, and evaluates.
6. Aggregates metrics across folds (mean ± std per strategy).
7. Refits on all data and runs SHAP interpretability (if enabled).
8. Saves everything to `reports/<name>/`.

### 3e. Class imbalance handling (reweighting)

For skewed targets (e.g. `growth_class` with rare no-growth wells), set `class_weight` in the experiment config:

```yaml
class_weight: balanced          # or a {class: weight} dict, e.g. {0: 5.0, 1: 1.0}
```

- Weights are computed with `sklearn.utils.class_weight.compute_sample_weight` on the **train fold only** (leakage-safe) and passed as `sample_weight` to `model.fit`. Estimators whose `fit` doesn't accept `sample_weight` fall back to an unweighted fit with a warning.
- The decision threshold is not fixed at 0.5: `tune_threshold` (`src/modeling/evaluate.py`) sweeps thresholds over the **train fold's** predicted probabilities and picks the one maximizing MCC (or F1 of the no-growth class), stored as `threshold` per fold in `fold_metrics.csv`.
- Because rare classes make per-fold estimates noisy, headline metrics are also computed on **pooled out-of-fold predictions** across all folds and saved under the `pooled` key in `metrics.json`, alongside the threshold-free ranking metrics.

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
| `random`             | `KFold` (shuffled)     | *(none)*      | configurable (default 5) | Interpolation / leakage baseline |
| `stratified`         | `StratifiedKFold`      | *(none)*      | configurable (default 5) | Class-balanced interpolation baseline |


Multiple strategies can be specified as a list in the experiment config; each is evaluated independently.

### 5a. Precomputed, reusable splits (`create_splits.py`)

By default `run_experiment.py` generates splits inline. For reproducibility and reuse across experiments/models, splits can instead be **precomputed once** and saved to disk with `scripts/modeling/create_splits.py`, then referenced from any experiment config.

Splits are stored as a compact **group -> fold map**: one row per `split_on_column` value (e.g. one row per unique genus) with `strategy`, `repeat`, and `fold` columns. Because `GroupKFold` / `StratifiedGroupKFold` never split a group across folds, this map fully determines the CV assignment while staying tiny and robust to row filtering/reordering at experiment time (rows are matched to their fold by group value, not by position). This makes the saved splits safe to reuse even after task filtering (e.g. `growth_rate`) or external-feature merges reorder or drop rows.

Only group-consistent strategies can be stored this way: `grouped` and `stratified_grouped` (keyed on `split_on_column`), plus `loso` / `lomo` when `split_on_column` is `strain_name` / `media_name`. Per-row strategies (`random`, `stratified`) are rejected.

#### Split config — `config/splits/*.yaml`

| Field             | Type             | Description                                                        |
| ----------------- | ---------------- | ------------------------------------------------------------------ |
| `dataset_path`    | str              | Processed dataset (CSV/Parquet) the splits are built from          |
| `split_on_column` | str              | Column whose values define the CV groups (map key, e.g. `genus`)   |
| `target`          | str              | Target column (used for `stratified_grouped` and distribution stats) |
| `cv.strategy`     | str or list[str] | Group-consistent CV strategy/strategies                            |
| `cv.n_repeats`    | int              | Number of CV repeats (repeats > 1 shuffle groups per repeat)       |
| `n_splits`        | int              | Number of folds for `grouped` / `stratified_grouped` (default 5)   |
| `seed`            | int              | Random seed (default 42)                                           |
| `output_format`   | `"csv"` or `"parquet"` | Map file format (default `csv`; the map is small)            |

Example (`config/splits/bacdive_genus.yaml`):

```yaml
dataset_path: data/external/bacdive_growth_150k.parquet
split_on_column: genus
target: growth_class

cv:
  strategy: [stratified_grouped]
  n_repeats: 1

n_splits: 5
seed: 42
output_format: csv
```

#### Run the script

```bash
python scripts/modeling/create_splits.py config/splits/bacdive_genus.yaml
```

Outputs are written to `data/modeling/splits_<dataset_stem>/`:

```
data/modeling/splits_bacdive_growth_150k/
  genus_bacdive_growth_150k.csv         ← group->fold map [strategy, repeat, fold, genus]
  genus_bacdive_growth_150k.log         ← full process log + per-fold distribution stats
  genus_bacdive_growth_150k_stats.json  ← machine-readable distribution stats
```

The log and stats JSON report, per `(strategy, repeat, fold)`: train/test row counts, train/test group counts, and the target distribution on train vs test (class counts + fractions for classification; mean/std/min/max for regression). A leakage assertion guarantees train/test group sets are disjoint.

#### Consume saved splits in an experiment

Reference the map and its group column from the experiment config's `cv` block:

```yaml
cv:
  strategy: [stratified_grouped]
  splits_file: data/modeling/splits_bacdive_growth_150k/genus_bacdive_growth_150k.csv
  split_on: genus
```

When `cv.splits_file` is set, `run_experiment.py` loads the map and derives folds by matching `split_on` values instead of calling `make_splits`. Before CV it compares `len(modeling_dataset)` to `dataset_shape[0]` in the companion `*_stats.json` (the row count of the table `create_splits.py` was run on). A mismatch raises — this catches pointing `modeling_dataset_path` at `data/interim/<output_name>` after `create_modeling_dataset.py` filtered rows, while splits were built from the original processed table. The requested strategies must exist in the map, and every `split_on` value in the modeling dataset must be present in the map (otherwise the run errors, preventing silent mismatch/leakage). Per-fold metrics and predictions gain a `repeat` column.

#### Held-out medium (S6): group on `media_canon`, never `media_name`

For a held-out-medium split, set `split_on_column: media_canon` -- the
alias-canonicalised medium identity added by
`src/bacdive/media.py::canonical_media_name`. Grouping on the raw `media_name`
**leaks**: BacDive spells one medium many ways, so a fold that "holds out"
`MacConkey agar` still trains on `MacConkey`, `MacConkey Agar`, `MacConcey
agar` and `Mac-Conkey agar`. On `bd_med_w_genome`, 93% of the no-growth rows
sit on a medium that is spelled more than one way, so a raw-name split would
leak almost the entire minority class and measure alias memorisation rather
than chemistry transfer.

```yaml
# config/splits/bd_med_w_genome_s6_medium.yaml
dataset_path: data/processed/bacdive/bd_med_w_genome.parquet
split_on_column: media_canon
target: growth_class
cv:
  strategy: [stratified_grouped]
n_splits: 5
```

Use `stratified_grouped`, not `lomo`: `lomo` is literal leave-one-medium-out
(one fold per medium, ~1.9k folds) and its fold labelling assumes a single
medium per fold. Rows whose medium is not identifiable carry
`media_canon_status` of `ambiguous` (the name lists several media) or
`unspecified` (no name at all) and can be dropped with a row filter:

```yaml
filters:
  exclude:
    media_canon_status: [ambiguous, unspecified]
```

#### LOSO / LOMO warning

- Under Leave-One-Strain-Out, one-hot encoded `strain_name` produces an all-zero vector for the held-out strain, which carries no signal. Meaningful LOSO generalization requires strain features that transfer across organisms (genome size, GC%, taxonomy, genome embeddings). 
- The same applies to `media_name` under LOMO. 
- The pipeline logs a warning when identity one-hot is the only strain/media feature under these strategies.

---

## 6. Featurization (`src/modeling/features.py`)

Feature encoding is defined in the experiment config's `features` dict and applied inside CV — the transformer is **fit on the train fold only** to prevent leakage.


| Spec            | Encoder          | Description                                                       |
| --------------- | ---------------- | ----------------------------------------------------------------- |
| `"OH"`          | `OneHotEncoder`  | One-hot encoding; `handle_unknown="ignore"` for unseen categories |
| `"ordinal"`     | `OrdinalEncoder` | Integer encoding; unknown values → `-1`                           |
| `"numeric"`     | `StandardScaler` | Z-score normalization                                             |
| `"path:<file>"` | passthrough      | Precomputed embedding CSV (first column = key, rest = floats)     |
| `["path:<file>", {pca_components: N}]` | `StandardScaler → PCA(N)` | External matrix with in-fold PCA reduction (see below) |


Each raw column is mapped to a semantic group for interpretability:


| Column                      | Group                      |
| --------------------------- | -------------------------- |
| `strain_name`               | `strain`                   |
| `media_name`                | `media`                    |
| `atmosphere`, `temperature` | `condition`                |
| genome metadata columns     | `genome`                   |
| embedding columns           | `strain_emb` / `media_emb` |
| PCA-reduced matrix columns  | feature key name (e.g. `metabolic_pathways`) |


#### External matrix with PCA

For high-dimensional strain-level features (e.g. metabolic pathway completeness), a matrix CSV can be merged and reduced with PCA inside each CV fold to avoid leakage:

```yaml
features:
  metabolic_pathways:
    - "path:data/external/modular_pathway/kegg_pathway_completeness.csv"
    - pca_components: 10          # optional; omit for raw passthrough
    # - merge_on: strain_name     # optional; defaults to strain_name
```

The CSV must have an entity name as its first column (matched against `merge_on`) and numeric values in all subsequent columns. During CV, a `StandardScaler → PCA(n_components)` pipeline is fit on each training fold only, producing `pca0 … pcaN-1` features. Unmatched rows are filled with 0.0.


**Notebook / EDA use only** — `featurize()` fits on all data and transforms in one call. Do not use its output for CV reporting:

```python
from src.modeling.features import featurize

X, y, feature_names, groups, fgm = featurize(df, cfg, target="growth_class")
```

---

## 7. Evaluation metrics (`src/modeling/evaluate.py`)

**Classification** (`growth_class`):


| Metric                             | Key                   |
| ----------------------------------- | --------------------- |
| AUROC                               | `auroc`               |
| PR-AUC (growth, label 1)            | `pr_auc`               |
| PR-AUC (no-growth, label 0)         | `pr_auc_nogrowth`     |
| F1 score (growth)                   | `f1`                  |
| F1 score (no-growth)                | `f1_nogrowth`         |
| Recall (no-growth)                  | `recall_nogrowth`     |
| Precision (no-growth)               | `precision_nogrowth`  |
| Matthews correlation                | `mcc`                 |
| Balanced accuracy                   | `balanced_accuracy`   |

`auroc`, `pr_auc`, and `pr_auc_nogrowth` are threshold-free (ranking); the rest are computed at the fold's tuned `threshold` (see [3e](#3e-class-imbalance-handling-reweighting)). The `*_nogrowth` variants target the minority class and are recommended over `f1`/`pr_auc` for imbalanced growth-class datasets.

**Regression** (`max_od`, `growth_rate`):


| Metric    | Key         |
| --------- | ----------- |
| RMSE      | `rmse`      |
| R²        | `r2`        |
| Pearson r | `pearson_r` |
| MAE       | `mae`       |


Metrics are computed per fold, then aggregated (mean ± std) per CV strategy, plus a pooled out-of-fold summary (see [3e](#3e-class-imbalance-handling-reweighting)). Per-strain and per-media breakdowns are saved separately, with support counts `n`/`n_pos`/`n_neg` so `NaN` ranking metrics on single-class groups are self-explanatory; a `*_withneg.csv` variant filters to groups containing at least one no-growth case.

---

## 8. Interpretability (`src/modeling/interpret.py`)

After CV, the model is refit on all data and SHAP values are computed:


| Model type                 | SHAP explainer                            |
| -------------------------- | ----------------------------------------- |
| XGBoost, Random Forest     | `TreeExplainer`                           |
| Logistic Regression, Ridge | `LinearExplainer`                         |
| MLP, other                 | `KernelExplainer` (subsampled background) |


For classifiers, TreeExplainer returns multi-output SHAP of shape `(n_samples, n_features, n_classes)`. Beeswarm/bar plots and group importance collapse this to a single output via `interpret.shap_class` (`"positive"` → class index 1, `"negative"` → index 0; ignored for regression).


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
  fold_metrics.csv         ← per-fold metrics (strategy, repeat, fold, fold_label, threshold, metrics)
  metrics.json             ← aggregate mean ± std per CV strategy + pooled OOF summary
  predictions.csv          ← y_true, y_pred, y_prob, strategy, repeat, fold, strain, media
  metrics_by_strain_name.csv          ← includes n, n_pos, n_neg
  metrics_by_strain_name_withneg.csv  ← filtered to groups with n_neg > 0
  metrics_by_media_name.csv
  metrics_by_media_name_withneg.csv
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
| `config.py`    | `load_curves_dataset_config`, `load_dataset_config`, `load_model_config`, `load_experiment_config`, `resolve_experiment` | Pydantic config models + YAML loading |
| `dataset.py`   | `build_modeling_dataset`                                                                   | Merge curves + metadata, optional replicate averaging |
| `io.py`        | `read_table`, `write_table`                                                                | CSV/Parquet load and save by file extension           |
| `features.py`  | `build_feature_pipeline`, `featurize`, `warn_loso_onehot`                                  | Leakage-safe sklearn `ColumnTransformer`              |
| `models.py`    | `build_model`                                                                              | Model registry (linear, RF, XGBoost, MLP, dummy)      |
| `cv.py`        | `make_splits`, `get_fold_group_label`                                                      | LOSO, LOMO, grouped, stratified-grouped, random, stratified splitters |
| `splits.py`    | `build_group_fold_map`, `fold_distribution_stats`, `save_split_map`, `load_split_map`, `iter_saved_splits` | Precompute / save / consume reusable group->fold splits |
| `evaluate.py`  | `evaluate_fold`, `aggregate_metrics`, `per_group_metrics`                                  | Classification and regression metrics                 |
| `interpret.py` | `compute_shap_values`, `run_interpretation`                                                | SHAP + grouped feature importance plots               |


**Scripts:**


| Script                               | Usage                          |
| ------------------------------------ | ------------------------------ |
| `scripts/modeling/create_modeling_dataset.py` | Step 1: build interim CSV/Parquet from a `curves_dataset_*.yaml` |
| `scripts/modeling/create_splits.py`  | Optional: precompute reusable CV splits (see [5a](#5a-precomputed-reusable-splits-create_splitspy)) |
| `scripts/modeling/run_experiment.py` | Step 2: CV + evaluation + SHAP |


**Notebooks:**


| Notebook                           | Contents                                          |
| ---------------------------------- | ------------------------------------------------- |
| `notebooks/02_featurization.ipynb` | Build dataset → inspect feature matrix and groups |
| `notebooks/03_modeling.ipynb`      | Step-by-step CV fold, evaluation, SHAP            |


---

## 11. Adding new features

1. Add the column to the curves CSV or strain metadata CSV (no code change for metadata columns merged in step 1).
2. Add an entry to the `features` dict in the experiment config (`config/experiments/**/*.yaml`):

```yaml
features:
  strain_name: "OH"
  genome_size_bp: "numeric"       # from strain metadata
  genus: "ordinal"                # from strain metadata
  strain_name: "path:data/external/evo2_embeddings.csv"  # precomputed
```

1. Re-run `create_modeling_dataset.py` with the curves config (if new columns come from metadata) and `run_experiment.py`.

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

## 13. Pluggable feature providers

These scripts produce inputs that slot into the modeling pipeline via config paths:


| Script                   | Output                               | Config hook                              |
| ------------------------ | ------------------------------------ | ---------------------------------------- |
| `get_strain_metadata.py` | `data/external/strains_metadata.csv` | `strain_metadata_file` in the **curves** dataset config |
| `get_evo_embeddings.py`  | genome embedding CSV                 | `path:<file>` in `features` dict         |
| `get_ecfp_embeddings.py` | media fingerprint CSV                | `path:<file>` in `features` dict         |
| KEGG pathway annotation  | `data/external/modular_pathway/kegg_pathway_completeness.csv` | `["path:<file>", {pca_components: N}]` in `features` dict |


No changes to `src/modeling/` are needed to use their outputs — only config updates.