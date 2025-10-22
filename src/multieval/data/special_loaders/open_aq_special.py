"""
Special loader for OpenAQ air quality data.

This loader handles the unique structure of OpenAQ data where parameters are stored
in rows rather than columns, requiring pivoting to wide format.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any


class OpenAQSpecialLoader:
    """
    Special loader for OpenAQ air quality data.
    
    This loader handles multivariate time series data with air quality measurements
    including PM1, PM2.5, relative humidity, temperature, and other parameters.
    PM2.5 is used as the target variable, while other parameters serve as covariates.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the OpenAQ special loader.
        
        Args:
            data_path: Path to the OpenAQ data directory
        """
        self.data_path = Path(data_path)
        self.pq_files = self._collect_files()
        
    def _collect_files(self) -> List[Path]:
        """Collect and sort all parquet files in the data location."""
        pq_files = list(self.data_path.rglob("*.parquet"))
        return sorted(pq_files)
    
    def _load_and_preprocess_data(self, file_path: Path) -> pd.DataFrame:
        """Load and preprocess a single OpenAQ data file."""
        # Load the parquet file
        df = pd.read_parquet(file_path)
        
        # Convert datetime column
        df["datetimeUtc"] = pd.to_datetime(df["datetimeUtc"])
        
        # Filter to relevant parameters
        relevant_parameters = [
            "pm1",
            "pm25", 
            "relativehumidity",
            "temperature",
            "um003",
        ]
        df = df[df["parameter"].isin(relevant_parameters)]
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # Pivot the data to wide format (one column per parameter)
        df_wide = df.pivot_table(
            index=["datetimeUtc", "location_id"],
            columns="parameter",
            values="value",
            aggfunc="first",
        ).reset_index()
        
        df_wide.columns.name = None
        df_wide = df_wide.sort_values("datetimeUtc").reset_index(drop=True)
        df_wide = df_wide.rename(columns={"datetimeUtc": "datetime"})
        
        return df_wide
    
    def load_all_data(self) -> List[pd.DataFrame]:
        """
        Load all OpenAQ data files as a list of DataFrames.
        
        Returns:
            List of DataFrames with processed data
        """
        all_dataframes = []
        
        for file_path in self.pq_files:
            df = self._load_and_preprocess_data(file_path)
            if len(df) > 0:
                all_dataframes.append(df)
        
        return all_dataframes
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """
        Get the dataset configuration for OpenAQ data.
        
        Returns:
            Dictionary with dataset configuration
        """
        return {
            "timestamp_col": "datetime",
            "target_cols": ["pm25"],  # PM2.5 is the target
            "metadata_cols": [
                "pm1",
                "relativehumidity", 
                "temperature",
                "um003",
            ],
            "description": "OpenAQ air quality data with PM2.5 as target and other parameters as covariates",
            "special_loader": "open_aq_special"
        }