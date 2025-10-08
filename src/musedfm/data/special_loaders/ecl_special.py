"""
Special loader for ECL dataset that handles S3 file loading, European decimal format,
chunking, and random column sampling.
"""

import pandas as pd
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Any
from musedfm.data.window import Window


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

    def _create_chunks_and_windows(self, df: pd.DataFrame, history_length: int = 30, 
                                 forecast_horizon: int = 1, stride: int = 1) -> List[Window]:
        """Create chunks and windows from the dataframe."""
        if len(df) == 0:
            return []

        target_cols = ["MT_001"]
        metadata_cols = [
            col for col in df.columns if col not in ["datetime", "MT_001"]
        ]

        # Subdivide the dataframe into 1024 length chunks
        df_chunks = [df.iloc[i : i + 1024] for i in range(0, len(df), 1024)]
        if self.random_ordering:
            random.shuffle(df_chunks)

        all_windows = []
        for chunk in df_chunks:
            # Sample 50 random metadata columns
            if len(metadata_cols) >= 50:
                sampled_cols = random.sample(metadata_cols, 50)
            else:
                sampled_cols = metadata_cols  # Use all available if less than 50

            # Create subchunk with sampled columns
            subchunk = chunk[["datetime"] + sampled_cols + target_cols]

            # Drop all NaN columns after window region selection
            subchunk = subchunk.dropna(axis=1, how='all')

            # Create windows from this chunk
            windows = self._create_windows_from_chunk(
                subchunk, history_length, forecast_horizon, stride
            )
            all_windows.extend(windows)

        return all_windows

    def _create_windows_from_chunk(self, chunk: pd.DataFrame, history_length: int,
                                 forecast_horizon: int, stride: int) -> List[Window]:
        """Create windows from a single chunk."""
        if len(chunk) < history_length + forecast_horizon:
            return []

        # Get target and metadata columns
        target_col = "MT_001"
        metadata_cols = [col for col in chunk.columns if col not in ["datetime", "MT_001"]]
        
        if target_col not in chunk.columns:
            return []

        # Extract time series data
        target_series = chunk[target_col].values
        if metadata_cols:
            covariate_data = chunk[metadata_cols].values
        else:
            covariate_data = np.zeros((len(chunk), 1))

        # Extract timestamp data
        timestamp_data = chunk["datetime"].values

        # Create sliding windows
        windows = []
        data_length = len(target_series)
        
        for start_idx in range(0, data_length - history_length - forecast_horizon + 1, stride):
            history_end = start_idx + history_length
            target_start = history_end
            target_end = target_start + forecast_horizon

            history = target_series[start_idx:history_end]
            target = target_series[target_start:target_end]
            covariates = covariate_data[start_idx:history_end]

            # Skip windows where target is completely NaN
            if np.all(np.isnan(target)) or np.all(np.isnan(history)):
                continue

            # Extract timestamps for this window
            timestamps = timestamp_data[start_idx:target_end]

            window = Window(history, target, covariates, timestamps)
            windows.append(window)

        return windows

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
            if len(df) > 0:
                all_dataframes.append(df)

        print(f"Successfully loaded {len(all_dataframes)} valid files")
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
