"""
Special loader for ECL dataset that handles S3 file loading, European decimal format,
chunking, and random column sampling.
"""

import pandas as pd
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Any


class ECLSpecialLoader:
    """
    Special loader for ECL dataset that handles:
    - S3 file collection and loading
    - European decimal format conversion (comma to dot)
    - Chunking dataframes into 1024-length pieces
    - Random column sampling (50 columns) with generic naming
    - NaN column dropping after window region selection
    """

    def __init__(self, data_path: str, random_ordering: bool = False):
        """
        Initialize the ECL special loader.
        
        Args:
            data_path: Path to the ECL data directory (or S3 path)
            random_ordering: Whether to use random ordering of files and chunks
        """
        self.data_path = Path(data_path)
        self.random_ordering = random_ordering
        self.parquet_files = self._collect_files()
        if self.random_ordering:
            random.shuffle(self.parquet_files)

    def _collect_files(self) -> List[Path]:
        """Collect parquet files from the data path."""
        # For now, assume local files - in production this would use S3
        parquet_files = list(self.data_path.rglob("*.parquet"))
        return sorted(parquet_files)

    def _load_and_preprocess_data(self, file_path: Path) -> pd.DataFrame:
        """Load and preprocess a single ECL data file."""
        df = pd.read_parquet(file_path)

        # Convert datetime column
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        df = df.set_index("datetime")
        df = df.reset_index()

        # Convert European decimal format (comma as decimal separator) to float
        for col in df.columns:
            if col != "datetime" and df[col].dtype == "object":
                # Check if the column contains any values with comma decimal separators
                has_comma_decimals = df[col].str.contains(r",", na=False).any()
                if has_comma_decimals:
                    # Replace comma with dot and convert to float, handling NaN values
                    df[col] = pd.to_numeric(
                        df[col].str.replace(",", "."), errors="coerce"
                    )
                else:
                    # Try to convert to numeric anyway in case it's already in correct format
                    df[col] = pd.to_numeric(df[col], errors="coerce")


        return df

    def _create_chunks(self, df: pd.DataFrame, chunk_size: int = 2048) -> List[pd.DataFrame]:
        """Create chunks and windows from the dataframe."""
        if len(df) == 0:
            return []

        target_cols = ["MT_001"]
        metadata_cols = [
            col for col in df.columns if col not in ["datetime", "MT_001"]
        ]

        # Subdivide the dataframe into 2048 length chunks
        df_chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), int(chunk_size / 4))]
        if self.random_ordering:
            random.shuffle(df_chunks)

        all_chunks = []
        for chunk in df_chunks:
            if len(chunk) <= 40:
                continue

            # Sample 50 random metadata columns
            if len(metadata_cols) >= 50:
                sampled_cols = random.sample(metadata_cols, 50)
            else:
                sampled_cols = metadata_cols  # Use all available if less than 50
            
            subchunk = chunk[["datetime"] + sampled_cols + target_cols]
            all_chunks.append(subchunk)

        return all_chunks

    def load_all_data(self) -> List[pd.DataFrame]:
        """
        Load all ECL data files as a list of DataFrames.
        
        Returns:
            List of DataFrames with processed data
        """
        all_dataframes = []
        print(f"Loading ECL data from {self.data_path}")
        print(f"Found {len(self.parquet_files)} parquet files")

        for file_path in self.parquet_files:
            df = self._load_and_preprocess_data(file_path)
            chunks = self._create_chunks(df)
            all_dataframes.extend(chunks)

        print(f"Successfully loaded {len(all_dataframes)} valid chunks")
        return all_dataframes

    def get_dataset_config(self) -> Dict[str, Any]:
        """
        Get the dataset configuration for ECL data.
        
        Returns:
            Dictionary with dataset configuration
        """
        return {
            "timestamp_col": "datetime",
            "target_cols": ["MT_001"],
            "metadata_cols": "ALL EXCEPT TS TARGET",  # Will be dynamically sampled
            "description": "ECL data with European decimal format handling, chunked processing, and random column sampling",
            "special_loader": "ecl_special"
        }
