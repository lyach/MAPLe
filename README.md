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