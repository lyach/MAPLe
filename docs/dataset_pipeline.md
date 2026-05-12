# Dataset Pipeline — Data Loading & Quality Control

## Overview

This pipeline converts raw plate reader CSVs into a clean parquet file ready for analysis. It requires three input files per experiment batch and produces one output file.

```
metadata CSV  ─┐
layout CSV    ─┼─► loader ─► tidy DataFrame ─► Quality Control ─► data/processed/*.parquet
raw CSV(s)    ─┘
```

---

## 1. Prepare input files

All input files live under `data/`:

```
data/
  metadata/
    <exp_id>_metadata.csv
    <exp_id>_layout.csv
  raw/
    <date>_<exp_id>_raw.csv
```

### 1a. Metadata CSV — `<date>_<batch>_metadata.csv`

One row per experiment. Required columns:

| Column | Type | Example | Description |
|---|---|---|---|
| `exp_id` | str | `aerobic_25` | Unique experiment ID; must match layout and raw filename |
| `replicate` | int | `1` | Replicate number |
| `date` | str | `20260325` | Date in `YYYYMMDD` format |
| `atmosphere` | str | `aerobic` | Growth atmosphere |
| `temperature` | float | `25` | Incubation temperature (°C) |
| `shaking` | float | `200` | Shaking speed (rpm) |
| `equipment` | str | `sunrise 1` | Plate reader model (must match a registered parser) |

Extra columns are allowed and will be propagated to the output automatically.


### 1b. Layout CSV — `<date>_<batch>_layout.csv`

One row per experiment. Required columns:

| Column | Type | Example | Description |
|---|---|---|---|
| `exp_id` | str | `aerobic_25` | Must match metadata |
| `date` | str | `20260325` | |
| `replicate` | int | `1` | |
| `equipment` | str | `sunrise 1` | |
| `time_unit` | str | `min` | Unit of the time axis |
| `time_interval` | float | `30` | Time between consecutive reads (in `time_unit`) |
| `no_wells` | int | `96` | Total wells on the plate |
| `used_cols_index` | int | `8` | Number of used columns |
| `used_rows_index` | int | `12` | Number of used rows |
| `name_cols` | str | `"LB, MB, NB, ..."` | Comma-separated media names, in plate-column order (col 1 → col N) |
| `name_rows` | str | `"E. coli, B. subtilis, ..."` | Comma-separated sample names, in plate-row order (row A → row H) |
| `blank_cols_index` | int or list | `12` | Column number(s) of blank wells; range notation supported |
| `blank_rows_index` | str | `A` | Row letter(s) of blank wells; range notation supported |

**Notes:**
- `name_cols` and `name_rows` are positional: the i-th name maps to the i-th plate column/row. 
- Well ID format in the output: `{row}_{col}_{well_index}` e.g. `a_1_0`, `b_3_14`.


### 1c. Raw CSV — `<date>_<exp_id>_raw.csv`

One file per experiment. The format depends on the equipment.

**Sunrise 1 / Sunrise 2:** consecutive plate-read blocks, one per timepoint. Each block is `(n_rows + 1)` lines:
- 1 header line: `<>,1,2,...,12`
- `n_rows` data lines: `A,0.133,0.143,...`

**BioSpa:** single CSV with time as rows and all wells as columns:
- 1 header line: `Time,Temperature,A1,A2,...,H12`
- 1 data line per timepoint: `H:MM:SS,temp,od1,od2,...`

The `Time` and `Temperature` columns are ignored by the parser; timepoints are assigned as 0-based indices from row order.

The file must be named exactly `<date>_<exp_id>_raw*.csv` (e.g. `20260325_aerobic_25_raw.csv`).

---

## 2. Run the pipeline

From the project root:

```bash
python scripts/run_dataset_pipeline.py \
  --metadata data/metadata/anaerobic_metadata.csv \
  --layout   data/metadata/anaerobic_layout.csv \
  --raw-dir  data/raw
```

Defaults point to the `aerobic` example files, so a bare call also works:

```bash
python scripts/run_dataset_pipeline.py
```

---

## 3. Output

The output is a long-format parquet file at the path given by `--output`.

**One row = one (experiment, well, timepoint) observation.**

| Column | Description |
|---|---|
| `exp_id` | Experiment ID |
| `date` | Date of experiment |
| `replicate` | Replicate number |
| `atmosphere` | Atmosphere condition |
| `temperature` | Temperature (°C) |
| `shaking` | Shaking speed (rpm) |
| `equipment` | Plate reader model |
| `well_id` | Unique well identifier (`row_col_index`) |
| `row` | Plate row (lowercase letter) |
| `col` | Plate column (integer) |
| `well_index` | Linear well index (row-major) |
| `sample_name` | Name of the sample/strain in this row |
| `media_name` | Name of the media in this column |
| `is_blank` | `True` for blank/control wells |
| `timepoint` | Timepoint index (0-based) |
| `time_value` | Actual time value |
| `time_unit` | Unit of `time_value` |
| `od600` | Blank-subtracted and clipped OD600 |
| `od600_raw` | Original OD600 before subtraction |
| `no_growth` | `True` if `max(od600) < 2 × mean(blank)` |
| `blank_cv` | CV of blank wells at this timepoint |
| `plate_cv_flag` | `True` if `blank_cv > 0.15` |

Any extra columns from the metadata CSV appear after `plate_cv_flag`.

Read the parquet in Python:

```python
import pandas as pd
df = pd.read_parquet("data/processed/20260325_aerobic_tidy.parquet")
```

---

## 4. Adding a new equipment parser

1. Create `src/dataset/parsers/<equipment_name>.py`.
2. Subclass `BaseParser` and implement `parse(path, layout) -> pd.DataFrame`.
3. Decorate the class with `@register("equipment name")`.
4. Import the module in `src/dataset/parsers/__init__.py`.

```python
# src/dataset/parsers/my_reader.py
from src.dataset.parsers import register
from src.dataset.parsers.base import BaseParser

@register("my reader model")
class MyReaderParser(BaseParser):
    def parse(self, path, layout):
        # return DataFrame with columns: timepoint, row, col, od600
        ...
```

The `equipment` value in the metadata CSV must exactly match the registered name (case-insensitive).

---

## 5. Adding new metadata fields

Add the column to the metadata CSV. No code changes required — `ExperimentMetadata` uses `extra="allow"` and the loader propagates all fields to the output.


---

## 6. Growth curve parameter extraction (`src/dataset/curves.py`)

`extract_growth_curves` takes the QC'd long-format DataFrame and produces a **wide-format** table where **one row = one (experiment, well) datapoint** with extracted growth parameters. It runs in three stages: smoothing, parameter extraction, and optional replicate averaging.

```
qc_df (long-format) ─► smooth OD per well ─► extract parameters ─► [average replicates] ─► curves_df (wide-format CSV)
```

### 6a. Smoothing

Each well's OD600 time-series is smoothed before parameter extraction. The smoother is selected by name via the `smoother` argument to `extract_growth_curves`. Two smoothers are available:

| Key | Function | Default parameters | Description |
|---|---|---|---|
| `"savgol"` *(default)* | `smooth_savgol` | `window_length=7`, `polyorder=3` | Savitzky-Golay filter via `scipy.signal.savgol_filter`. Preserves curve shape better than a simple average; window is automatically shrunk for short series. |
| `"moving_average"` | `smooth_moving_average` | `window=5` | Centered moving average using convolution with edge replication. Faster but more aggressive smoothing. |

To switch smoother or pass custom parameters:

```python
from src.dataset.curves import extract_growth_curves

# Savitzky-Golay with wider window
curves_df = extract_growth_curves(qc_df, smoother="savgol", smoother_kwargs={"window_length": 11, "polyorder": 3})

# Simple moving average
curves_df = extract_growth_curves(qc_df, smoother="moving_average", smoother_kwargs={"window": 7})
```

**Adding a new smoother:** define a function with signature `f(od: np.ndarray, **kwargs) -> np.ndarray` and insert it into `SMOOTHERS`:

```python
# in src/dataset/curves.py
def smooth_my_method(od, my_param=3):
    ...

SMOOTHERS["my_method"] = smooth_my_method
```

### 6b. Parameter extraction (`extract_params`)

Called once per well on the smoothed OD600 array. Input units follow the `time_unit` of the experiment (typically minutes for the aerobic batch).

| Parameter | Formula / method |
|---|---|
| `max_od` | `max(OD600)` |
| `auc` | Trapezoidal integral: `∫ OD600 dt` |
| `growth_rate` | Sustained maximum slope: per-window slopes from a sliding 4-point window on `log(OD600 + ε)` are computed, then a rolling mean over 3 consecutive windows selects the region of most consistent growth. The steepest individual slope within that region is reported (`ε = 1e-6` prevents `log(0)`). Units: OD/time_unit. |
| `lag_time_h` | X-intercept of the log-phase tangent line extrapolated to the initial baseline OD. Clamped to ≥ 0; set to the last timepoint if no positive growth rate is detected. |
| `time_to_max_od_h` | Time at which `max_od` occurs. |

**Growth rate / lag time geometry:**

```
log(OD)
  │         /← log-phase slope (growth_rate)
  │        /
  │       /
  │──────/  ←── lag_time_h = x-intercept of this line at baseline log(OD[0])
  └───────────── time
```

### 6c. Growth classification (`compute_growth_threshold`)

The threshold for `growth_class` is computed per experiment from the blank wells:

```
threshold = 3 × std(blank od600_raw)
```

`od600_raw` (pre-blank-subtraction) is used so the spread reflects raw instrument noise. After blank subtraction, a well is classified as **growing** (`growth_class = 1`) if `max_od > threshold`, and **not growing** (`growth_class = 0`) otherwise.

### 6d. Replicate averaging (`with_replicates`)

By default (`with_replicates=False`) the output has one row per well per experiment, with `exp_id` and `replicate` retained as identifier columns.

When `with_replicates=True`, rows sharing `(strain_name, media_name, atmosphere, temperature)` are collapsed into a single row:
- Continuous parameters (`max_od`, `auc`, `growth_rate`, `lag_time_h`, `time_to_max_od_h`) are **averaged** across replicates.
- `growth_class` uses a **majority vote**: `1` if the mean across replicates ≥ 0.5.
- `exp_id` and `replicate` are dropped from the output.

```python
# Per-well output (default)
curves_df = extract_growth_curves(qc_df)

# Averaged across replicates
curves_df = extract_growth_curves(qc_df, with_replicates=True)
```

### 6e. Output schema

**One row = one (experiment, well) datapoint** when `with_replicates=False`.  
**One row = one (strain, media, condition) average** when `with_replicates=True`.

| Column | Present when | Description |
|---|---|---|
| `exp_id` | `with_replicates=False` | Experiment ID |
| `date` | `with_replicates=False` | Date of experiment |
| `replicate` | `with_replicates=False` | Replicate number |
| `strain_name` | always | Strain name |
| `media_name` | always | Media name |
| `atmosphere` | always | Atmosphere condition |
| `temperature` | always | Temperature (°C) |
| `max_od` | always | Maximum OD600 (blank-subtracted, smoothed) |
| `auc` | always | Area under the growth curve |
| `growth_rate` | always | Maximum specific growth rate (log-phase slope) |
| `lag_time_h` | always | Lag time before exponential growth |
| `time_to_max_od_h` | always | Time at which maximum OD is reached |
| `growth_class` | always | `1` = growth, `0` = no growth |

The curves CSV is saved alongside the parquet file:

```python
import pandas as pd
curves_df = pd.read_csv("data/processed/aerobic_tidy_curves.csv")
```