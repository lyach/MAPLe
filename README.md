# MicrobeNet: A Framework for Modeling Microbial Genotype-Phenotype
**The Laboratory for Metabolic Systems Engineering in collaboration with the Align Foundation**

## Getting Started

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

## Project Structure

```
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
