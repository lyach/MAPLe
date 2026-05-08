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
    <date>_<batch>_metadata.csv
    <date>_<batch>_layout.csv
  raw/
    <date>_<exp_id>_raw.csv      (one file per experiment in the batch)
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
| `blank_cols_index` | int or list | `12` | Column number(s) of blank wells |
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
  --metadata data/metadata/20260325_aerobic_metadata.csv \
  --layout   data/metadata/20260325_aerobic_layout.csv \
  --raw-dir  data/raw \
  --output   data/processed/20260325_aerobic_tidy.parquet
```

Defaults point to the `20260325_aerobic` example files, so a bare call also works:

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
