# MicrobeNet Framework Overview

## Scripts

Order agnostic, needed for features:

* get_strain_metadata.py
* get_evo_embeddings.py
* get_ecfp_embeddings.py

In order:

1. run_dataset_pipeline.py
2. create_modeling_dataset.py
3. run_experiment.py

## Modeling Plan

### Tasks

- `growth_class` (binary classification)
- `max_od` (regression)
- `growth_rate` (regression, only on growing wells)

### Models

- Linear (Logistic and Ridge)
- XGBoost (classifier and regressor)
- Random Forest 
- MLP

### Cross-validation Strategies

- **Leave-One-Strain-Out (LOSO):** predict the held-out strain — simulates generalization to new organisms
- **Leave-One-Media-Out (LOMO):** predict the held-out medium — simulates generalization to new environments

Both give \(n\)-fold CV matching the number of unique strains or media.

### Evaluation Metrics

- Classification: AUROC, F1, MCC
- Regression: RMSE, R2, Pearson r (per-strain, per-media and overall)

### Interpretability

- SHAP values on the best XGBoost model to rank feature importance
- Identify which features (strain vs. media vs. condition) drive predictions most