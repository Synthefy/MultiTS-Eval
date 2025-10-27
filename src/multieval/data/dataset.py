"""
Dataset class for managing collections of windows from parquet files.
"""

import pandas as pd
import numpy as np
import json
import re
import random
from datetime import datetime, timedelta
from typing import List, Iterator, Dict, Any, Optional, Union
from pathlib import Path
from multieval.data.window import Window

from multieval.data.special_loaders.open_aq_special import OpenAQSpecialLoader
from multieval.data.special_loaders.kitti_special import KITTISpecialLoader
from multieval.data.special_loaders.ecl_special import ECLSpecialLoader


class Dataset:
    """
    Collection of windows parsed from parquet files.
    Iterable: for window in dataset
    Aggregates results across windows.
    """
    
    
    @classmethod
    def _load_dataset_config(cls) -> Dict[str, Any]:
        """Load dataset configuration from JSON file."""
        config_path = Path(__file__).parent / "dataset_config.json"
        with open(config_path, 'r') as f:
            return json.load(f)
    
    @classmethod
    def _parse_index_based_spec(cls, spec: str, df_columns: List[str], timestamp_col: str) -> List[str]:
        """Helper function to parse index-based column specifications.
        
        Handles patterns like:
        - INDEX# (e.g., INDEX1 means second column, index 1)
        - "second column (index 1)"
        - "last non-timestamp column"
        - "all non-timestamp columns"
        """
        spec = spec.strip()
        
        if spec.startswith("INDEX"):
            # Handle INDEX# format (e.g., INDEX1 means second column, index 1)
            try:
                index_num = int(spec[5:])  # Extract number after "INDEX"
                if 0 <= index_num < len(df_columns):
                    return [df_columns[index_num]]
                else:
                    return []
            except (ValueError, IndexError):
                return []
        
        elif "second column (index 1)" in spec.lower():
            # Return the second column (index 1)
            return [df_columns[1]] if len(df_columns) > 1 else []
        
        elif "last non-timestamp column" in spec.lower():
            # Return the last non-timestamp column
            non_timestamp_cols = [col for col in df_columns if col != timestamp_col]
            return [non_timestamp_cols[-1]] if non_timestamp_cols else []
        
        elif "all non-timestamp columns" in spec.lower():
            # Return all non-timestamp columns
            return [col for col in df_columns if col != timestamp_col]
        
        return []  # No index-based pattern matched

    @classmethod
    def _parse_timestamp_col(cls, timestamp_spec: str, df_columns: List[str]) -> str:
        """Parse timestamp column specification into actual column name.
        
        Handles patterns like:
        - Direct column names (e.g., "timestamp", "datetime")
        - INDEX# format (e.g., INDEX0 means first column, index 0)
        - Special values like "COMBINE_DATETIME_COLUMNS", "AUTOMATICALLY GENERATED"
        - OR|| patterns for multiple candidates
        """
        timestamp_spec = timestamp_spec.strip()
                
        # Handle OR|| patterns
        if " OR|| " in timestamp_spec:
            candidates = timestamp_spec.split(" OR|| ")
            for candidate in candidates:
                candidate = candidate.strip()
                # Only check direct column names, not INDEX patterns
                if candidate in df_columns:
                    return candidate
            return candidates[0].strip()  # Return first candidate as fallback
        
        # Try index-based parsing
        if timestamp_spec.startswith("INDEX"):
            index_num = int(timestamp_spec[5:])
            return df_columns[index_num]
        
        # Return the spec as-is if it's a direct column name or other pattern
        return timestamp_spec

    @classmethod
    def _parse_metadata_cols(cls, metadata_spec: Union[str, List[str]], 
                           df_columns: List[str], timestamp_col: str, 
                           target_cols: List[str], current_target: Optional[str] = None) -> List[str]:
        """Parse metadata column specification into actual column names."""
        if isinstance(metadata_spec, list):
            # Handle list of specifications - process each one
            result = []
            for spec in metadata_spec:
                if isinstance(spec, str):
                    parsed = cls._parse_metadata_cols(spec, df_columns, timestamp_col, target_cols, current_target)
                    result.extend(parsed)
                else:
                    result.append(spec)
            return result
        
        metadata_spec = metadata_spec.strip()
        
        # Try index-based parsing first
        index_result = cls._parse_index_based_spec(metadata_spec, df_columns, timestamp_col)
        if index_result:
            return index_result
        
        if metadata_spec == "ALL EXCEPT TS TARGET":
            # All columns except timestamp and target columns
            if current_target:
                return [col for col in df_columns if col != timestamp_col and col != current_target]
            else:
                return [col for col in df_columns if col != timestamp_col and col not in target_cols]
        
        elif "all columns except" in metadata_spec.lower():
            # Parse exclusion pattern like "all columns except datetime, count, casual_riders_count, member_riders_count"
            exclude_pattern = metadata_spec.lower().replace("all columns except", "").strip()
            exclude_cols = [col.strip() for col in exclude_pattern.split(",")]
            return [col for col in df_columns if col not in exclude_cols]
        
        elif "all columns after" in metadata_spec.lower():
            # Parse pattern like "all columns after price column (columns 2 onwards)"
            # Find the reference column and return all columns after it
            match = re.search(r"all columns after (\w+)", metadata_spec.lower())
            if match:
                ref_col = match.group(1)
                if ref_col in df_columns:
                    ref_idx = df_columns.index(ref_col)
                    return df_columns[ref_idx + 1:]
            return []
        
        elif "columns" in metadata_spec and "to" in metadata_spec:
            # Parse pattern like "columns 2 to -2 (excluding last ERCOT column)"
            match = re.search(r"columns (\d+) to (-?\d+)", metadata_spec.lower())
            if match:
                start_idx = int(match.group(1)) - 1  # Convert to 0-based
                end_idx = int(match.group(2))
                if end_idx < 0:
                    end_idx = len(df_columns) + end_idx
                return df_columns[start_idx:end_idx]
            return []
        
        else:
            # Try to parse as comma-separated column names
            return [col.strip() for col in metadata_spec.split(",")]
    
    @classmethod
    def _parse_target_cols(cls, target_spec: Union[str, List[str]], 
                          df_columns: List[str], timestamp_col: str) -> List[str]:
        """Parse target column specification into actual column names."""
        if isinstance(target_spec, list):
            # Process each item in the list
            result = []
            for spec in target_spec:
                if isinstance(spec, str):
                    parsed = cls._parse_target_cols(spec, df_columns, timestamp_col)
                    result.extend(parsed)
                else:
                    result.append(spec)
            return result
        
        target_spec = target_spec.strip()

        # Try index-based parsing first
        index_result = cls._parse_index_based_spec(target_spec, df_columns, timestamp_col)
        if index_result:
            return index_result
        
        elif target_spec == "ALL EXCEPT TS TARGET":
            # For targets, this means all non-timestamp columns
            return [col for col in df_columns if col != timestamp_col]
        
        elif target_spec.startswith("DYNAMIC:"):
            # Handle DYNAMIC: format (e.g., "DYNAMIC: RPI, UNRATE, HOUST")
            dynamic_options = [opt.strip() for opt in target_spec[8:].split(",")]  # Remove "DYNAMIC: " prefix
            
            # Find the first option that exists in the dataframe columns
            for option in dynamic_options:
                if option in df_columns:
                    return [option]
            
            # If none found, return first non-timestamp column as fallback
            non_timestamp_cols = [col for col in df_columns if col != timestamp_col]
            return [non_timestamp_cols[0]] if non_timestamp_cols else []
        
        elif target_spec.startswith("CONTAINS:"):
            # Handle CONTAINS: format (e.g., "CONTAINS: conc")
            search_term = target_spec[9:].strip()  # Remove "CONTAINS: " prefix
            
            # Find columns that contain the search term
            matching_cols = [col for col in df_columns if search_term.lower() in col.lower()]
            
            if matching_cols:
                return matching_cols
            else:
                # If no matches found, return first non-timestamp column as fallback
                non_timestamp_cols = [col for col in df_columns if col != timestamp_col]
                return [non_timestamp_cols[0]] if non_timestamp_cols else []
        
        else:
            # just return the target spec
            return [target_spec]
    
    def __init__(self, data_path: str, history_length: int = 30, forecast_horizon: int = 1, 
                 stride: int = 1, column_config: Optional[Dict[str, Any]] = None, needs_counting: bool = False, dataset_target_count: Optional[Dict[str, int]] = None, batch_size: int = 1):
        """
        Initialize dataset from parquet files.
        
        Args:
            data_path: Path to directory containing parquet files
            history_length: Number of historical points to use for forecasting
            forecast_horizon: Number of future points to forecast
            stride: Step size between windows (default: 1 for every window)
            column_config: Dictionary specifying column configuration. If None, will auto-detect.
                          Format: {
                              'timestamp_col': 'column_name',
                              'target_cols': ['col1', 'col2'] or callable,
                              'covariate_cols': ['col3', 'col4'] or callable
                          }
            needs_counting: If True, count windows
            batch_size: Number of windows to collect in each batch (default: 1 for single window mode)
        """
        self.data_path = Path(data_path)
        self.dataset_name = self.data_path.name
        self.history_length = history_length
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self.column_config = column_config
        self.batch_size = batch_size
        self._windows: List[Window] = []
        self.target_count = len(column_config['target_cols']) if column_config is not None and 'target_cols' in column_config else 1
        

        # Lazy loading state
        self._current_parquet_index = 0
        self._current_window_index = 0
        self._parquet_files: List[Path] = []
        self._total_windows: int = 0
        self._load_parquet_paths()
        if needs_counting:
            self._count_windows()
    
    @classmethod
    def with_custom_config(cls, data_path: str, timestamp_col: str, 
                          target_cols: Union[List[str], str], 
                          metadata_cols: Union[List[str], str],
                          history_length: int = 30, forecast_horizon: int = 1, stride: int = 1, batch_size: int = 1) -> 'Dataset':
        """
        Create a dataset with custom column configuration.
        
        Args:
            data_path: Path to directory containing parquet files
            timestamp_col: Name of the timestamp column
            target_cols: List of target column names or string specification
            metadata_cols: List of metadata column names or string specification
            history_length: Number of historical points to use for forecasting
            forecast_horizon: Number of future points to forecast
            stride: Step size between windows
            batch_size: Number of windows to collect in each batch
            
        Returns:
            Dataset instance with custom configuration
        """
        column_config = {
            'timestamp_col': timestamp_col,
            'target_cols': target_cols,
            'metadata_cols': metadata_cols,
            'from_json_config': True
        }
        return cls(data_path, history_length, forecast_horizon, stride, column_config, batch_size=batch_size)
    
    @classmethod
    def from_config(cls, data_path: str, dataset_name: str, 
                   history_length: int = 30, forecast_horizon: int = 1, stride: int = 1, needs_counting: bool = True, batch_size: int = 1) -> 'Dataset':
        """
        Create a dataset using configuration from the JSON config file.
        
        Args:
            data_path: Path to directory containing parquet files
            dataset_name: Name of the dataset in the config file
            history_length: Number of historical points to use for forecasting
            forecast_horizon: Number of future points to forecast
            stride: Step size between windows
            needs_counting: If True, count windows
            batch_size: Number of windows to collect in each batch
        Returns:
            Dataset instance with configuration from JSON file
        """
        config = cls._load_dataset_config()
        if dataset_name not in config:
            raise ValueError(f"Dataset '{dataset_name}' not found in config. Available datasets: {list(config.keys())}")
        
        dataset_config = config[dataset_name]
        
        # Check if this dataset uses a special loader
        if 'special_loader' in dataset_config:
            return cls._create_with_special_loader(data_path, dataset_name, dataset_config, 
                                                 history_length, forecast_horizon, stride, needs_counting, batch_size)
        
        # Create a special config that will be processed by the generalized loader
        column_config = {
            'timestamp_col': dataset_config['timestamp_col'],
            'target_cols': dataset_config['target_cols'],
            'metadata_cols': dataset_config['metadata_cols'],
            'from_json_config': True  # Flag to indicate this comes from JSON config
        }
        
        return cls(data_path, history_length, forecast_horizon, stride, column_config, needs_counting, batch_size=batch_size)
    
    @classmethod
    def _create_with_special_loader(cls, data_path: str, dataset_name: str, dataset_config: Dict[str, Any],
                                  history_length: int, forecast_horizon: int, stride: int, needs_counting: bool = True, batch_size: int = 1) -> 'Dataset':
        """
        Create a dataset using a special loader for datasets with unique formats.
        
        Args:
            data_path: Path to the data directory
            dataset_name: Name of the dataset
            dataset_config: Configuration from JSON file
            history_length: Number of historical points to use for forecasting
            forecast_horizon: Number of future points to forecast
            stride: Step size between windows
            needs_counting: If True, count windows
            batch_size: Number of windows to collect in each batch
        Returns:
            Dataset instance with special loader
        """
        special_loader_name = dataset_config['special_loader']
        
        # Import the appropriate special loader
        if special_loader_name == 'open_aq_special':
            special_loader = OpenAQSpecialLoader(data_path)
        elif special_loader_name == 'kitti_special':
            special_loader = KITTISpecialLoader(data_path)
        elif special_loader_name == 'ecl_special':
            special_loader = ECLSpecialLoader(data_path)
        else:
            raise ValueError(f"Unknown special loader: {special_loader_name}")
            
        # Load all data using the special loader
        dataframes = special_loader.load_all_data()
        
        if not dataframes:
            raise ValueError(f"No data loaded for dataset {dataset_name}")
        
        # Create column config from special loader
        column_config = {
            'timestamp_col': dataset_config['timestamp_col'],
            'target_cols': dataset_config['target_cols'],
            'metadata_cols': dataset_config['metadata_cols'],
            'from_json_config': True,
            'special_loader': True,
            'dataframes': dataframes
        }
        
        # Create dataset instance with the dataframes
        dataset = cls(data_path, history_length, forecast_horizon, stride, column_config, needs_counting, batch_size=batch_size)
        dataset.dataset_name = dataset_name
        return dataset
    
    def _load_parquet_paths(self) -> None:
        """Load parquet file paths for lazy loading."""
        # Check if this is a special loader with DataFrames
        if self.column_config and self.column_config.get('special_loader') and 'dataframes' in self.column_config:
            self._parquet_files = self.column_config['dataframes']
            return
            
        self._parquet_files = sorted(list(self.data_path.rglob("*.parquet")))
        
        if not self._parquet_files:
            raise ValueError(f"No parquet files found in {self.data_path}")
    
    def _count_windows_in_file(self, parquet_file) -> int:
        """Count windows in a parquet file or DataFrame without creating them.
        Uses simplified logic based on dataframe length."""
        if isinstance(parquet_file, pd.DataFrame):
            df = parquet_file
        else:
            df = pd.read_parquet(parquet_file)
        data_length = len(df)
        
        # Minimum history length for forecasting (prioritize forecasting after 15 values)
        MIN_HISTORY_FORECAST_LENGTH = 15
        min_history_for_forecast = min(MIN_HISTORY_FORECAST_LENGTH, self.history_length)
        
        # Minimum total data required: 15 history + 15 forecast = 30 points
        min_total_data = MIN_HISTORY_FORECAST_LENGTH * 2
        
        # Check if dataset has minimum required data
        if data_length < min_total_data:
            return 0
        
        # Calculate the maximum start index
        max_start_index = data_length - min_history_for_forecast
        
        if max_start_index <= 0:
            return 0
        
        # Count windows with stride
        windows_count = 0
        for start_idx in range(0, max_start_index, self.stride):
            # Determine actual history length for this window
            available_data = data_length - start_idx
            
            if available_data >= self.history_length + self.forecast_horizon:
                # Full history and forecast available
                windows_count += self.target_count
            elif available_data >= MIN_HISTORY_FORECAST_LENGTH * 2:
                # Split available data equally between history and forecast
                windows_count += self.target_count
        
        return windows_count
    
    def _count_windows(self) -> None:
        """Count total windows across all parquet files.
        For datasets with >100 files, estimates by sampling first 100 files."""
        num_files = len(self._parquet_files)
        
        if num_files <= 100:
            # Count windows in all files using custom counter
            self._total_windows = 0
            for parquet_file in self._parquet_files:
                windows_count = self._count_windows_in_file(parquet_file)
                self._total_windows += windows_count

        else:
            # Estimate by sampling first 100 files using custom counter
            print(f"Large dataset {self.dataset_name} detected ({num_files} files). Estimating window count by sampling first 100 files...")
            
            sample_windows = 0
            sample_size = min(100, num_files)
            
            for i in range(sample_size):
                parquet_file = self._parquet_files[i]
                windows_count = self._count_windows_in_file(parquet_file)
                sample_windows += windows_count
            
            # Calculate average windows per file and estimate total
            avg_windows_per_file = sample_windows / sample_size
            self._total_windows = int(avg_windows_per_file * num_files)
            
            print(f"Estimated {self._total_windows} total windows ({avg_windows_per_file:.1f} avg per file)")
    
    def get_column_config(self) -> Dict[str, Any]:
        """Auto-detect column configuration based on dataset folder name."""
        config = self._load_dataset_config()
        
        # The dataset name is the name of the folder (last part of the path)
        dataset_name = self.data_path.name
        
        # Check if this dataset name exists in config
        if dataset_name in config:
            dataset_config = config[dataset_name]
            
            # Return the config - let column parsing handle validation
            return {
                'timestamp_col': dataset_config['timestamp_col'],
                'target_cols': dataset_config['target_cols'],
                'metadata_cols': dataset_config['metadata_cols'],
                'from_json_config': True
            }
        
        raise ValueError(f"Could not auto-detect dataset configuration from path {self.data_path}. Dataset name '{dataset_name}' not found in config.")

    def _clean_numeric_columns(self, df: pd.DataFrame, timestamp_col: str, target_cols: List[str], metadata_cols: List[str]) -> tuple[pd.DataFrame, List[str], List[str]]:
        """Clean numeric columns by removing commas and converting to numeric.
        
        Returns:
            Tuple of (cleaned_dataframe, updated_target_cols, updated_metadata_cols)
        """
        df_cleaned = df.copy()
        
        # Skip timestamp column, but process target and metadata columns
        check_cols = list(set(target_cols + metadata_cols))
        
        # Convert target and metadata columns to numeric, handling comma-separated numbers
        if check_cols:
            # Check if any columns contain string values (including comma-separated numbers)
            has_string_values = (
                df_cleaned[check_cols].dtypes.eq('object').any()
            )
            
            if has_string_values:
                df_cleaned[check_cols] = (
                    df_cleaned[check_cols]
                    .astype(str)
                    .replace(',', '', regex=True)
                    .apply(pd.to_numeric, errors='coerce')
                )
        
        # Remove columns that are entirely NaN
        df_cleaned = df_cleaned.dropna(axis=1, how='all')
        
        # Track which columns were dropped
        dropped_cols = set(df.columns) - set(df_cleaned.columns)
        if dropped_cols:
            pass
            # print(f"Dropped columns due to all NaN values: {dropped_cols}")
        
        # Update target and metadata column lists to remove dropped columns
        updated_target_cols = [col for col in target_cols if col in df_cleaned.columns]
        updated_metadata_cols = [col for col in metadata_cols if col in df_cleaned.columns]
        
        # Remove trailing and preceding NaNs from each column
        
        for i, col in enumerate(updated_target_cols):
            if col != timestamp_col:  # Don't modify timestamp column
                # Find first and last non-NaN values
                first_valid = df_cleaned[col].first_valid_index()
                last_valid = df_cleaned[col].last_valid_index()
                
                
                if first_valid is not None and last_valid is not None:
                    # Keep only the range from first to last valid value
                    old_shape = df_cleaned.shape
                    df_cleaned = df_cleaned.loc[first_valid:last_valid]
                    new_shape = df_cleaned.shape
                else:
                    # If column is entirely NaN, it should have been dropped above
                    pass        
        return df_cleaned, updated_target_cols, updated_metadata_cols

    def _prep_windows_from_file(self, parquet_file) -> None:
        """Load windows from a single parquet file or DataFrame."""
        if isinstance(parquet_file, pd.DataFrame):
            df = parquet_file
        else:
            df = pd.read_parquet(parquet_file)
                
        # Auto-detect column configuration if not provided
        if self.column_config is None:
            self.column_config = self._auto_detect_column_config(df)
        
        # Handle timestamp column parsing
        timestamp_spec = self.column_config.get('timestamp_col')
        parsed_timestamp_col = self._parse_timestamp_col(timestamp_spec, df.columns.tolist())
        
        # Handle automatically generated timestamps if needed
        if parsed_timestamp_col == "AUTOMATICALLY GENERATED":
            timestamp_col = "timestamp"
            if timestamp_col not in df.columns:
                df[timestamp_col] = self._generate_timestamps(len(df))
            self.column_config['used_timestamp_col'] = timestamp_col
        
        # Handle combining separate datetime columns if needed
        elif parsed_timestamp_col == "COMBINE_DATETIME_COLUMNS":
            timestamp_col = "timestamp"
            if timestamp_col not in df.columns:
                df = self._combine_datetime_columns(df, timestamp_col)
            self.column_config['used_timestamp_col'] = timestamp_col
        
        else:
            # Use the parsed timestamp column
            self.column_config['used_timestamp_col'] = parsed_timestamp_col
                
        windows = self._load_windows_general(df)
        return windows
        
    def _generate_timestamps(self, length: int) -> List[datetime]:
        """Generate realistic timestamps starting from a random date/time."""
        # Choose a random start time within the last 5 years
        now = datetime.now()
        start_time = now - timedelta(days=random.randint(30, 365 * 5))
        
        # Randomly choose time interval (minutes or hours)
        use_hours = random.choice([True, False])
        
        if use_hours:
            # Generate hourly timestamps
            interval = timedelta(hours=1)
        else:
            # Generate minute-based timestamps (5, 10, 15, 30, or 60 minutes)
            minute_intervals = [5, 10, 15, 30, 60]
            interval_minutes = random.choice(minute_intervals)
            interval = timedelta(minutes=interval_minutes)
        
        # Generate the timestamp sequence
        timestamps = []
        current_time = start_time
        
        for _ in range(length):
            timestamps.append(current_time)
            current_time += interval
        
        return timestamps
    
    def _has_separate_datetime_columns(self, df: pd.DataFrame) -> bool:
        """Check if the dataframe has separate datetime columns that can be combined."""
        datetime_columns = ['year', 'month', 'day', 'hour', 'minute', 'second']
        return all(col in df.columns for col in datetime_columns)
    
    def _combine_datetime_columns(self, df: pd.DataFrame, timestamp_col: str) -> List[datetime]:
        """Combine separate datetime columns into a single timestamp column."""
        datetime_column_names = ['year', 'month', 'day', 'hour', 'minute', 'second']
        
        # Create a dictionary with available columns
        datetime_data = {}
        for col in datetime_column_names:
            if col in df.columns:
                datetime_data[col] = df[col].values
        
        # Generate timestamps
        timestamps = []
        for i in range(len(df)):
            # Get values for this row, with defaults for missing columns
            year = datetime_data.get('year', [0] * len(df))[i]
            month = datetime_data.get('month', [0] * len(df))[i]
            day = datetime_data.get('day', [0] * len(df))[i]
            hour = datetime_data.get('hour', [0] * len(df))[i]
            minute = datetime_data.get('minute', [0] * len(df))[i]
            second = datetime_data.get('second', [0] * len(df))[i]

            # determine which columns are present
            datetime_columns = list()
            if year != 0:
                datetime_columns.append(year)
            if month != 0:
                datetime_columns.append(month)
            if day != 0:
                datetime_columns.append(day)
            if hour != 0:
                datetime_columns.append(hour)
            if minute != 0:
                datetime_columns.append(minute)
            if second != 0:
                datetime_columns.append(second)
            
            # create the timestamp
            timestamp = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
            timestamps.append(timestamp)
        
        # remove other timestamp columns
        for col in df.columns:
            if col in datetime_column_names:
                df = df.drop(col, axis=1)
        df[timestamp_col] = timestamps
        # print(df.columns, len(df), datetime_column_names, timestamp_col, [col in datetime_column_names for col in df.columns])

        return df
    
    def _create_windows_with_stride(self, target_series: np.ndarray, covariate_data: np.ndarray, timestamp_data: Optional[np.ndarray] = None) -> None:
        """Create windows with stride parameter and handle incomplete windows."""
        data_length = len(target_series)
        
        # Minimum history length for forecasting (prioritize forecasting after 15 values)
        MIN_HISTORY_FORECAST_LENGTH = 15
        min_history_for_forecast = min(MIN_HISTORY_FORECAST_LENGTH, self.history_length)
        
        # Minimum total data required: 15 history + 15 forecast = 30 points
        min_total_data = MIN_HISTORY_FORECAST_LENGTH * 2
        
        # Check if dataset has minimum required data
        if data_length < min_total_data:
            # Dataset is too short for any meaningful forecasting (need at least 30 points)
            return []
        
        # Calculate the maximum start index
        max_start_index = data_length - min_history_for_forecast
        
        if max_start_index <= 0:
            # Dataset is too short for any meaningful forecasting
            return []
        
        # Create windows with stride
        windows = []
        for start_idx in range(0, max_start_index, self.stride):
            # Determine actual history length for this window
            available_data = data_length - start_idx
            
            if available_data >= self.history_length + self.forecast_horizon:
                # Full history and forecast available
                actual_history_length = self.history_length
                history_end = start_idx + actual_history_length
                target_start = history_end
                target_end = target_start + self.forecast_horizon
                
                history = target_series[start_idx:history_end]
                target = target_series[target_start:target_end]
                covariates = covariate_data[start_idx:history_end]

                
                # Skip windows where target is completely NaN
                if np.all(np.isnan(target)) or np.all(np.isnan(history)):
                    print(f"Skipping window with completely NaN target: {start_idx}")
                    continue
                # print(f"Window with no NaN target: {start_idx} mean: {np.nanmean(target)} variance: {np.nanstd(target)} num nan: {np.isnan(target).sum()}")
                
                # Extract timestamps for this window if available
                timestamps = None
                if timestamp_data is not None:
                    timestamps = timestamp_data[start_idx:target_end]
                
                # Create padding masks (all False for individual windows)
                history_padding = np.zeros_like(history, dtype=bool)
                target_padding = np.zeros_like(target, dtype=bool)
                
                window = Window(history, target, covariates, timestamps, history_padding, target_padding)
                windows.append(window)
            elif available_data >= self.forecast_horizon * 2:
                # Split available data so that the forecast is equal to forecast length, and the history is the remainder
                actual_history_length = available_data - self.forecast_horizon
                history_end = start_idx + actual_history_length
                target_start = history_end
                target_end = target_start + self.forecast_horizon
                
                history = target_series[start_idx:history_end]
                target = target_series[target_start:target_end]
                covariates = covariate_data[start_idx:history_end]
                
                # Skip windows where target is completely NaN
                if np.all(np.isnan(target)) or np.all(np.isnan(history)):
                    print(f"Skipping window with completely NaN target: {start_idx}")
                    continue
                
                # Extract timestamps for this window if available
                timestamps = None
                if timestamp_data is not None:
                    timestamps = timestamp_data[start_idx:target_end]
                
                # Create padding masks (all False for individual windows)
                history_padding = np.zeros_like(history, dtype=bool)
                target_padding = np.zeros_like(target, dtype=bool)
                
                window = Window(history, target, covariates, timestamps, history_padding, target_padding)
                windows.append(window)
                
            elif available_data >= MIN_HISTORY_FORECAST_LENGTH * 2:

                # Split available data equally between history and forecast, but limit target to forecast_horizon
                actual_history_length = available_data // 2
                history_end = start_idx + actual_history_length
                target_start = history_end
                target_end = min(target_start + self.forecast_horizon, start_idx + available_data)
                
                history = target_series[start_idx:history_end]
                target = target_series[target_start:target_end]
                covariates = covariate_data[start_idx:history_end]
                
                # Skip windows where target is completely NaN
                if np.all(np.isnan(target)) or np.all(np.isnan(history)):
                    print(f"Skipping window with completely NaN target: {start_idx}")
                    continue
                
                # Extract timestamps for this window if available
                timestamps = None
                if timestamp_data is not None:
                    timestamps = timestamp_data[start_idx:target_end]
                
                # Create padding masks (all False for individual windows)
                history_padding = np.zeros_like(history, dtype=bool)
                target_padding = np.zeros_like(target, dtype=bool)
                
                window = Window(history, target, covariates, timestamps, history_padding, target_padding)
                windows.append(window)
            # Skip windows that don't meet minimum requirements
        return windows
    
    def _create_batched_window(self, windows: List[Window]) -> Window:
        """
        Create a single batched window from a list of individual windows.
        
        Args:
            windows: List of individual windows to batch together
            
        Returns:
            Single Window object containing batched data
        """
        if not windows:
            raise ValueError("Cannot create batched window from empty list")
        
        # Extract data from all windows
        histories = []
        targets = []
        covariates_list = []
        timestamps_list = []
        
        for window in windows:
            histories.append(window.history())
            targets.append(window.target())
            covariates_list.append(window.covariates())
            if window.has_timestamps:
                timestamps_list.append(window.timestamps())
            else:
                timestamps_list.append(None)
        
        # Handle variable shapes by padding to expected dimensions
        actual_history_lengths = [h.shape[0] for h in histories]
        actual_forecast_lengths = [t.shape[0] for t in targets]
        expected_history_length = max(actual_history_lengths)
        expected_forecast_horizon = max(actual_forecast_lengths)  # Use max forecast length in batch
        
        # Pad/truncate histories to expected length (pad at the beginning to preserve recent data)
        padded_histories = []
        history_padding_masks = []
        for history in histories:
            if history.shape[0] < expected_history_length:
                # Pad with first value (forward-fill) at the beginning
                if len(history) > 0 and not np.all(np.isnan(history)):
                    first_valid = history[~np.isnan(history)][0] if np.any(~np.isnan(history)) else 0.0
                else:
                    first_valid = 0.0
                padding_length = expected_history_length - history.shape[0]
                padding = np.full(padding_length, first_valid)
                padded_history = np.concatenate([padding, history])
                # Create padding mask (True = padded positions)
                padding_mask = np.concatenate([np.ones(padding_length, dtype=bool), np.zeros(history.shape[0], dtype=bool)])
            elif history.shape[0] > expected_history_length:
                # Truncate to expected length (keep last N values)
                padded_history = history[-expected_history_length:]
                padding_mask = np.zeros(expected_history_length, dtype=bool)
            else:
                padded_history = history
                padding_mask = np.zeros(expected_history_length, dtype=bool)
            padded_histories.append(padded_history)
            history_padding_masks.append(padding_mask)
        
        # Pad/truncate targets to expected length (pad at the end to preserve immediate predictions)
        padded_targets = []
        target_padding_masks = []
        for target in targets:
            if target.shape[0] < expected_forecast_horizon:
                # Pad with last value (forward-fill) at the end
                if len(target) > 0 and not np.all(np.isnan(target)):
                    last_valid = target[~np.isnan(target)][-1] if np.any(~np.isnan(target)) else 0.0
                else:
                    last_valid = 0.0
                padding_length = expected_forecast_horizon - target.shape[0]
                padding = np.full(padding_length, last_valid)
                padded_target = np.concatenate([target, padding])
                # Create padding mask (True = padded positions)
                padding_mask = np.concatenate([np.zeros(target.shape[0], dtype=bool), np.ones(padding_length, dtype=bool)])
            elif target.shape[0] > expected_forecast_horizon:
                # Truncate to expected length (keep first N values)
                padded_target = target[:expected_forecast_horizon]
                padding_mask = np.zeros(expected_forecast_horizon, dtype=bool)
            else:
                padded_target = target
                padding_mask = np.zeros(expected_forecast_horizon, dtype=bool)
            padded_targets.append(padded_target)
            target_padding_masks.append(padding_mask)
        
        # Pad/truncate covariates to expected length and dimensions
        padded_covariates_list = []
        
        # First, determine the maximum covariate dimension across all windows
        max_covariate_dim = 0
        for covariates in covariates_list:
            if covariates is not None and covariates.shape[1] > max_covariate_dim:
                max_covariate_dim = covariates.shape[1]
        
        # Ensure we have at least dimension 1 for covariates
        if max_covariate_dim == 0:
            max_covariate_dim = 1
        
        for covariates in covariates_list:
            if covariates is None:
                # Create zero covariates if None
                padded_covariates = np.zeros((expected_history_length, max_covariate_dim))
            else:
                # Pad/truncate time dimension
                if covariates.shape[0] < expected_history_length:
                    # Pad with first row values (forward-fill) at the beginning
                    if covariates.shape[0] > 0 and not np.all(np.isnan(covariates)):
                        first_row = covariates[0]
                        # Replace any NaN values in first row with 0
                        first_row = np.where(np.isnan(first_row), 0.0, first_row)
                    else:
                        first_row = np.zeros(covariates.shape[1])
                    padding_shape = (expected_history_length - covariates.shape[0], covariates.shape[1])
                    padding = np.tile(first_row, (padding_shape[0], 1))
                    padded_covariates = np.vstack([padding, covariates])
                elif covariates.shape[0] > expected_history_length:
                    # Truncate to expected length (keep last N values)
                    padded_covariates = covariates[-expected_history_length:]
                else:
                    padded_covariates = covariates
                
                # Pad covariate dimension if needed
                if padded_covariates.shape[1] < max_covariate_dim:
                    padding_shape = (padded_covariates.shape[0], max_covariate_dim - padded_covariates.shape[1])
                    padding = np.zeros(padding_shape)
                    padded_covariates = np.hstack([padded_covariates, padding])
            
            padded_covariates_list.append(padded_covariates)
        
        # Stack arrays to create batch dimensions
        batched_history = np.stack(padded_histories, axis=0)  # [batch_size, history_length]
        batched_target = np.stack(padded_targets, axis=0)      # [batch_size, forecast_horizon]
        batched_covariates = np.stack(padded_covariates_list, axis=0)  # [batch_size, history_length, covariate_dim]
        
        # Stack padding masks
        batched_history_padding = np.stack(history_padding_masks, axis=0)  # [batch_size, history_length]
        batched_target_padding = np.stack(target_padding_masks, axis=0)    # [batch_size, forecast_horizon]
        
        # Handle timestamps (pad/truncate to consistent length)
        batched_timestamps = None
        if all(ts is not None for ts in timestamps_list):
            # Calculate expected timestamp length (history + forecast)
            expected_timestamp_length = expected_history_length + expected_forecast_horizon
            
            # Pad/truncate timestamps to expected length
            padded_timestamps_list = []
            for timestamps in timestamps_list:
                if timestamps.shape[0] < expected_timestamp_length:
                    # For timestamps, we need to handle dtype properly
                    # Create padding with the same dtype as timestamps
                    if np.issubdtype(timestamps.dtype, np.datetime64):
                        # Use NaT (Not a Time) for datetime64
                        padding = np.full(expected_timestamp_length - timestamps.shape[0], np.datetime64('NaT'))
                    else:
                        # Use NaN for other numeric types
                        padding = np.full(expected_timestamp_length - timestamps.shape[0], np.nan)
                    padded_timestamps = np.concatenate([timestamps, padding])
                elif timestamps.shape[0] > expected_timestamp_length:
                    # Truncate to expected length (keep first N values)
                    padded_timestamps = timestamps[:expected_timestamp_length]
                else:
                    padded_timestamps = timestamps
                padded_timestamps_list.append(padded_timestamps)
            
            batched_timestamps = np.stack(padded_timestamps_list, axis=0)
        
        # Create batched window
        batched_window = Window(
            history=batched_history,
            target=batched_target,
            covariates=batched_covariates,
            timestamps=batched_timestamps,
            history_padding=batched_history_padding,
            target_padding=batched_target_padding
        )
        
        return batched_window

    def _load_windows_general(self, df: pd.DataFrame) -> List[Window]:
        """Generalized window loading function."""
        # Get column configuration
        timestamp_col = self.column_config['timestamp_col'] if 'used_timestamp_col' not in self.column_config else self.column_config['used_timestamp_col']
        
        # Parse target and metadata columns from config
        target_cols = self._parse_target_cols(
            self.column_config['target_cols'], 
            df.columns.tolist(), 
            timestamp_col
        )
        metadata_cols_spec = self.column_config['metadata_cols']
        
        # Sort by timestamp column
        df = df.sort_values(timestamp_col)
        
        # Parse metadata columns from config
        covariate_cols = self._parse_metadata_cols(
            metadata_cols_spec,
            df.columns.tolist(),
            timestamp_col,
            target_cols,
        )
        df, updated_target_cols, updated_covariate_cols = self._clean_numeric_columns(df, timestamp_col, target_cols, covariate_cols)
        # Update the column lists to reflect dropped columns
        target_cols = updated_target_cols
        covariate_cols = updated_covariate_cols
        
        # Create windows for each target column
        all_windows = []
        for target_col in target_cols:
            if target_col in covariate_cols:
                covariate_cols.remove(target_col)
            
            
            # Extract time series data
            target_series = df[target_col].values
            if covariate_cols:
                covariate_data = df[covariate_cols].values
            else:
                covariate_data = np.zeros((len(df), 1))
            
            # Extract timestamp data if available
            timestamp_data = None
            if timestamp_col in df.columns:
                timestamp_data = df[timestamp_col].values
            
            # Create sliding windows with stride and incomplete window handling
            windows = self._create_windows_with_stride(target_series, covariate_data, timestamp_data)
            all_windows.extend(windows)
        return all_windows
    
    def __iter__(self) -> Iterator[Window]:
        """Iterate over windows. If there 
        are no more windows, load the next parquet file
        until there are no more windows to load.
        """
        return self._lazy_iterator()
    
    def _lazy_iterator(self) -> Iterator[Window]:
        """Lazy iterator that loads parquet files on-demand and always creates batched windows."""
        # Reset state for fresh iteration
        self._current_parquet_index = 0
        self._current_window_index = 0
        self._windows = []
        total_windows_yielded = 0
        max_iterations = len(self._parquet_files) * 10  # Safety limit to prevent infinite loops
        iteration_count = 0
        
        while self._current_parquet_index < len(self._parquet_files) or len(self._windows) > 0:
            iteration_count += 1
            if iteration_count > max_iterations:
                print(f"Warning: Maximum iterations ({max_iterations}) reached in dataset iterator. Breaking to prevent infinite loop.")
                break
                
            # Load windows from current parquet file if not already loaded
            if not self._windows and self._current_parquet_index < len(self._parquet_files):
                parquet_file = self._parquet_files[self._current_parquet_index]
                self._windows = self._prep_windows_from_file(parquet_file)
                self._current_window_index = 0
                self._current_parquet_index += 1
                
                # If all files have been processed and no windows found, break
                if self._current_parquet_index >= len(self._parquet_files) and not self._windows:
                    break
            
            # Collect windows until we have enough for a batch or run out of windows
            batch_windows = []
            while len(batch_windows) < self.batch_size and (self._current_window_index < len(self._windows) or self._current_parquet_index < len(self._parquet_files)):
                # If we've exhausted current file's windows, load next file
                if self._current_window_index >= len(self._windows) and self._current_parquet_index < len(self._parquet_files):
                    parquet_file = self._parquet_files[self._current_parquet_index]
                    self._windows = self._prep_windows_from_file(parquet_file)
                    self._current_window_index = 0
                    self._current_parquet_index += 1
                    
                    # If the loaded file has no windows, break out of inner loop to try next file
                    if not self._windows:
                        break
                
                # Add windows from current file to batch
                remaining_in_file = len(self._windows) - self._current_window_index
                remaining_for_batch = self.batch_size - len(batch_windows)
                windows_to_add = min(remaining_in_file, remaining_for_batch)
                
                for i in range(windows_to_add):
                    batch_windows.append(self._windows[self._current_window_index + i])
                
                self._current_window_index += windows_to_add
            
            # Yield batch if we have any windows
            if batch_windows:
                batched_window = self._create_batched_window(batch_windows)
                total_windows_yielded += len(batch_windows)
                yield batched_window
            else:
                # If no batch was created and we've processed all files, break
                if self._current_parquet_index >= len(self._parquet_files):
                    break
    
    def __len__(self) -> int:
        """Return total number of windows across all parquet files."""
        return self._total_windows
    
    def num_files(self) -> int:
        """Return number of parquet files in the dataset."""
        return len(self._parquet_files)
    
    def __getitem__(self, index):
        '''
        Slice the dataset by parquet files
        '''        
        if isinstance(index, slice):
            # Handle nested slicing
            actual_index = index.start if index.start else 0
            actual_end = index.stop if index.stop else self.num_files()
            new_dataset = Dataset(self.data_path, self.history_length, self.forecast_horizon, self.stride, self.column_config, True, {}, self.batch_size)
            new_dataset._parquet_files = self._parquet_files[actual_index:actual_end]
            return new_dataset
        elif isinstance(index, int):
            actual_index = index
            if actual_index >= self.num_files():
                raise IndexError("Index out of range")
            new_dataset = Dataset(self.data_path, self.history_length, self.forecast_horizon, self.stride, self.column_config, True, {}, self.batch_size)
            new_dataset._parquet_files = [self._parquet_files[actual_index]]
            return new_dataset
        else:
            raise TypeError("Invalid index type")
    
    def get_num_parquet_files(self) -> int:
        """Return number of parquet files in the dataset."""
        return len(self._parquet_files)
    
    def reset_iterator(self) -> None:
        """Reset the iterator state to allow fresh iteration."""
        self._current_parquet_index = 0
        self._current_window_index = 0
        self._windows = []
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation for all windows (delegates to their cached results).
        
        Returns:
            Dictionary of aggregated evaluation metrics
        """
        if not self._windows:
            return {}
        
        # Check if all windows have forecasts
        windows_with_forecasts = [w for w in self._windows if w.has_forecast]
        if len(windows_with_forecasts) != len(self._windows):
            raise ValueError(f"Only {len(windows_with_forecasts)}/{len(self._windows)} windows have forecasts submitted")
        
        print(f"Evaluating {len(self._windows)} windows")
        
        # Aggregate results from all windows
        all_results = [window.evaluate() for window in self._windows]
        
        # Get metric names from the first result
        metric_names = list(all_results[0].keys())
        
        # Collect all metric vectors from all windows
        all_metric_values = {metric: [] for metric in metric_names}
        
        for result in all_results:
            for metric in metric_names:
                all_metric_values[metric].extend(result[metric])
        
        # Compute single nanmean across all individual window metrics
        aggregated = {}
        for metric in metric_names:
            aggregated[metric] = np.nanmean(all_metric_values[metric]) if all_metric_values[metric] else np.nan
            aggregated[f"{metric}_std"] = np.nanstd(all_metric_values[metric]) if all_metric_values[metric] else np.nan
        
        return aggregated
    
    def to_results_csv(self, path: str) -> None:
        """
        Validate all windows have forecasts submitted.
        Collect cached results, compute aggregates, save CSV.
        Does not recompute metrics or save forecasts.
        
        Args:
            path: Output CSV file path
        """
        # Validate all windows have forecasts
        windows_with_forecasts = [w for w in self._windows if w.has_forecast]
        if len(windows_with_forecasts) != len(self._windows):
            raise ValueError(f"Only {len(windows_with_forecasts)}/{len(self._windows)} windows have forecasts submitted")
        
        # Collect results
        results = []
        for i, window in enumerate(self._windows):
            window_results = window.evaluate()
            window_results['window_id'] = i
            results.append(window_results)
        
        # Create DataFrame and save
        results_df = pd.DataFrame(results)
        results_df.to_csv(path, index=False)
        
        # Also save aggregated results
        aggregated_results = self.evaluate()
        aggregated_df = pd.DataFrame([aggregated_results])
        aggregated_path = path.replace('.csv', '_aggregated.csv')
        aggregated_df.to_csv(aggregated_path, index=False)
