# MicrobeNet: A Framework for Modeling Microbial Genotype-Phenotype

A microbial phenotyping project to generate reproducible, scalable, and standardized growth data across structured conditions (media, nutrients, stressors)

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
   curl -LsSf https://astral.sh/uv/install.sh | sh # Linux/macOS
   # or
   pip install uv # Windows

   # Create environment
   uv venv --python 3.11 .venv

   # Activate environment
   source .venv/bin/activate # Linux/macOS
   # or 
   .venv\Scripts\activate # Windows

   # Install dependencies
   uv sync
   ```

## Dataset Pipeline

After the steps above are complete, you should see on your terminal something like:

```bash
(.venv) [your_user MicrobeNet]$ 
```

Then you can proceed to the following steps:

1. Prepare and upload your data files.
   - Place the `data/` folder on this directory.
   - Edit the csv with the experiment specs `aerobic_metadata.csv` (or `anaerobic_metadata.csv`). This is on the `data/metadata` folder *[(here)](data/metadata/)*. 
   - Edit the csv with the plate layout `aerobic_metadata.csv` (or `anaerobic_metadata.csv`). This is on the `data/metadata` folder *[(here)](data/metadata/)*. 
   - Place on the `data/metadata` folder raw platereader files. **Note:** The anaerobic (Biospa) still needs manual adjusting. *See example for [aerobic](data/raw/20260325_aerobic_25_raw.csv) and [anaerobic](data/raw/20260325_anaerobic_a_37_raw.csv)*.  
2. Run the script for the Dataset Pipeline
   ```bash
   python scripts/run_dataset_pipeline.py aerobic
   # or
   python scripts/run_dataset_pipeline.py anaerobic
   ```
   The output will print:
   ```
   Saved long dataframe: ../MicrobeNet/data/processed/anaerobic_tidy.parquet  (447.4 KB)
   Saved curves: ../MicrobeNet/data/processed/anaerobic_tidy_curves.csv  (36.2 KB)
   ```
3. Explore the data using the [EDA notebook](notebooks/01_eda.ipynb).
Detailed documentation on the pipeline can be found [here](docs/dataset_pipeline.md).



## Project Structure

```
data/
  metadata/
    <exp_id>_metadata.csv      ←  Experiment metadata
    <exp_id>_layout.csv        ←  Plate layout specs
  raw/
    <date>_<exp_id>_raw.csv    ←  Raw files from platereaders

src/
  /dataset
      schema.py            ←  Data schemas of plates
      plate.py             ←  Layout parsing
      parsers/             ←  Equipment-parser mapping
        base.py            ←  Base parser
        sunrise.py         ←  Sunrise (aerobic) parser
        biospa.py          ←  Biospa (anaerobic) parser
      loader.py            ←  Data loading for modeling
      qc.py                ←  Blank subtraction, QC flags
      curves.py            ←  Growth parameters
   /modeling 
      features.py          ← Feature engineering for ML
      models.py            ← Model training, CV, evaluation
notebooks/
  01_eda.ipynb             ← Exploratory data analysis
  02_modeling.ipynb        ← Baseline modeling
data/
  raw/                     ← CSVs from platereaders
  processed/               ← Data ready for modeling
  metadata/                ← Metadata files, filled by user
```
