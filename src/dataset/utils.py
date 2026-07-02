"""Utility functions for the dataset."""

from __future__ import annotations

import pandas as pd
from pathlib import Path

def concatenate_csv_files(file_paths: list[str], output_path: str) -> pd.DataFrame:
    """
    Concatenate multiple CSV files into a single DataFrame.
    
    Args:
        file_paths: List of paths to the input CSV files.
        output_path: Path where the concatenated CSV should be saved.
        
    Returns:
        The concatenated pandas DataFrame.
    """
    # Print len of file_paths
    print(f"Concatenating {len(file_paths)} files")
    for file in file_paths:
        df = pd.read_csv(file)
        print(f"File {file} has {len(df)} rows")
    
    # Concatenate
    df = pd.concat([pd.read_csv(file) for file in file_paths], ignore_index=True)
    
    print(f"Concatenated DataFrame has {len(df)} rows")
        
    # Save to the specified path without writing the dataframe index as a column
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return df