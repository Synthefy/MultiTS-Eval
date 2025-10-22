"""
Special loader for KITTI dataset that handles variable column structures.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


class KITTISpecialLoader:
    """
    Special loader for KITTI dataset that standardizes column structure.
    Handles variable track columns by padding with NaN values.
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.parquet_files = self._collect_files()

        # Standard column structure for KITTI
        self.target_cols = ["target_1", "target_2"]
        self.keep_cols = [
            "track_1_x", "track_1_y",
            "track_2_x", "track_2_y", 
            "track_3_x", "track_3_y",
            "track_4_x", "track_4_y"
        ]
        self.timestamp_col = "timestamp"

    def _collect_files(self) -> List[Path]:
        """Collect all parquet files in the data directory."""
        parquet_files = list(self.data_path.rglob("*.parquet"))
        return sorted(parquet_files)

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column structure by renaming and padding columns.
        Similar to the KITTI dataloader logic.
        """
        # Remove unnamed columns
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        # Get existing columns (excluding target and timestamp columns)
        existing_cols = [col for col in df.columns 
                        if col not in ["timestamp", "target_1", "target_2"]]

        # Rename existing columns to track format
        rename_dict = {}
        for i, col in enumerate(existing_cols):
            track_num = (i // 2) + 1
            coord = 'x' if i % 2 == 0 else 'y'
            new_name = f"track_{track_num}_{coord}"
            rename_dict[col] = new_name

        df = df.rename(columns=rename_dict)

        # Add missing track columns with NaN values
        existing_track_cols = [col for col in df.columns if col.startswith("track_")]
        missing_cols = [col for col in self.keep_cols if col not in existing_track_cols]

        if missing_cols:
            for col in missing_cols:
                df[col] = np.nan

        # Ensure all required columns exist
        for col in self.keep_cols:
            if col not in df.columns:
                df[col] = np.nan

        # Create timestamp if it doesn't exist
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.date_range(
                start=pd.to_datetime("2025-01-01") + pd.Timedelta(seconds=np.random.randint(0, 3600 * 24 * 365)),
                periods=len(df),
                freq="1s"
            )

        return df

    def _load_and_preprocess_data(self, file_path: Path) -> pd.DataFrame:
        """Load and preprocess a single KITTI data file."""
        try:
            # Load the parquet file
            df = pd.read_parquet(file_path)

            # Skip very short trajectories
            if len(df) < 50:
                return pd.DataFrame()

            # Standardize column structure
            df = self._standardize_columns(df)

            # Ensure we have the required columns
            required_cols = ["timestamp"] + self.target_cols + self.keep_cols
            missing_required = [col for col in required_cols if col not in df.columns]

            if missing_required:
                print(f"Warning: Missing required columns in {file_path}: {missing_required}")
                return pd.DataFrame()

            return df

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return pd.DataFrame()

    def load_all_data(self) -> List[pd.DataFrame]:
        """
        Load all KITTI data files as a list of DataFrames.
        
        Returns:
            List of DataFrames with standardized column structure
        """
        all_dataframes = []
        print(f"Loading KITTI data from {self.data_path}")
        print(f"Found {len(self.parquet_files)} parquet files")

        for file_path in self.parquet_files:
            df = self._load_and_preprocess_data(file_path)
            if len(df) > 0:
                all_dataframes.append(df)

        print(f"Successfully loaded {len(all_dataframes)} valid files")
        return all_dataframes

    def get_dataset_config(self) -> Dict[str, Any]:
        """Get the dataset configuration for KITTI."""
        return {
            "timestamp_col": self.timestamp_col,
            "target_cols": self.target_cols,
            "metadata_cols": self.keep_cols,
            "description": "KITTI dataset with standardized track columns"
        }