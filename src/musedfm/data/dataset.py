"""
Dataset class for managing collections of windows from parquet files.
"""

import pandas as pd
import numpy as np
import json
import re
import random
from datetime import datetime, timedelta
from typing import Iterator, Dict, Any, List, Optional, Union
from pathlib import Path
from .window import Window

from .special_loaders.open_aq_special import OpenAQSpecialLoader


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

        print("parsing target cols", target_spec, df_columns, timestamp_col)
        
        if target_spec.startswith("INDEX"):
            # Handle INDEX# format (e.g., INDEX1 means second column, index 1)
            try:
                index_num = int(target_spec[5:])  # Extract number after "INDEX"
                if 0 <= index_num < len(df_columns):
                    return [df_columns[index_num]]
                else:
                    return []
            except (ValueError, IndexError):
                return []
        
        elif "second column (index 1)" in target_spec.lower():
            # Return the second column (index 1)
            return [df_columns[1]] if len(df_columns) > 1 else []
        
        elif "last non-timestamp column" in target_spec.lower():
            # Return the last non-timestamp column
            non_timestamp_cols = [col for col in df_columns if col != timestamp_col]
            return [non_timestamp_cols[-1]] if non_timestamp_cols else []
        
        elif "all non-timestamp columns" in target_spec.lower():
            # Return all non-timestamp columns
            return [col for col in df_columns if col != timestamp_col]
        
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
                 stride: int = 1, column_config: Optional[Dict[str, Any]] = None):
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
        """
        self.data_path = Path(data_path)
        self.dataset_name = self.data_path.name
        self.history_length = history_length
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self.column_config = column_config
        self._windows: List[Window] = []
        
        # Lazy loading state
        self._current_parquet_index = 0
        self._current_window_index = 0
        self._parquet_files: List[Path] = []
        self._total_windows: int = 0
        self._load_parquet_paths()
        self._count_windows()
    
    @classmethod
    def with_custom_config(cls, data_path: str, timestamp_col: str, 
                          target_cols: Union[List[str], str], 
                          metadata_cols: Union[List[str], str],
                          history_length: int = 30, forecast_horizon: int = 1, stride: int = 1) -> 'Dataset':
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
            
        Returns:
            Dataset instance with custom configuration
        """
        column_config = {
            'timestamp_col': timestamp_col,
            'target_cols': target_cols,
            'metadata_cols': metadata_cols,
            'from_json_config': True
        }
        return cls(data_path, history_length, forecast_horizon, stride, column_config)
    
    @classmethod
    def from_config(cls, data_path: str, dataset_name: str, 
                   history_length: int = 30, forecast_horizon: int = 1, stride: int = 1) -> 'Dataset':
        """
        Create a dataset using configuration from the JSON config file.
        
        Args:
            data_path: Path to directory containing parquet files
            dataset_name: Name of the dataset in the config file
            history_length: Number of historical points to use for forecasting
            forecast_horizon: Number of future points to forecast
            stride: Step size between windows
            
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
                                                 history_length, forecast_horizon, stride)
        
        # Create a special config that will be processed by the generalized loader
        column_config = {
            'timestamp_col': dataset_config['timestamp_col'],
            'target_cols': dataset_config['target_cols'],
            'metadata_cols': dataset_config['metadata_cols'],
            'from_json_config': True  # Flag to indicate this comes from JSON config
        }
        
        return cls(data_path, history_length, forecast_horizon, stride, column_config)
    
    @classmethod
    def _create_with_special_loader(cls, data_path: str, dataset_name: str, dataset_config: Dict[str, Any],
                                  history_length: int, forecast_horizon: int, stride: int) -> 'Dataset':
        """
        Create a dataset using a special loader for datasets with unique formats.
        
        Args:
            data_path: Path to the data directory
            dataset_name: Name of the dataset
            dataset_config: Configuration from JSON file
            history_length: Number of historical points to use for forecasting
            forecast_horizon: Number of future points to forecast
            stride: Step size between windows
            
        Returns:
            Dataset instance with special loader
        """
        special_loader_name = dataset_config['special_loader']
        
        # Import the appropriate special loader
        if special_loader_name == 'open_aq_special':
            from .special_loaders.open_aq_special import OpenAQSpecialLoader
            special_loader = OpenAQSpecialLoader(data_path)
            
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
            dataset = cls(data_path, history_length, forecast_horizon, stride, column_config)
            dataset.dataset_name = dataset_name
            return dataset
            
        else:
            raise ValueError(f"Unknown special loader: {special_loader_name}")
    
    def _load_parquet_paths(self) -> None:
        """Load parquet file paths for lazy loading."""
        # Check if this is a special loader with DataFrames
        if self.column_config and self.column_config.get('special_loader') and 'dataframes' in self.column_config:
            self._parquet_files = self.column_config['dataframes']
            return
            
        self._parquet_files = list(self.data_path.rglob("*.parquet"))
        
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
        
        # Minimum history length for forecasting (prioritize forecasting after 20 values)
        MIN_HISTORY_FORECAST_LENGTH = 20
        min_history_for_forecast = min(MIN_HISTORY_FORECAST_LENGTH, self.history_length)
        
        # Minimum total data required: 20 history + 20 forecast = 40 points
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
                windows_count += 1
            elif available_data >= MIN_HISTORY_FORECAST_LENGTH * 2:
                # Split available data equally between history and forecast
                windows_count += 1
        
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
        
        print(check_cols, df.columns)
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
            print(f"Dropped columns due to all NaN values: {dropped_cols}")
        
        # Update target and metadata column lists to remove dropped columns
        updated_target_cols = [col for col in target_cols if col in df_cleaned.columns]
        updated_metadata_cols = [col for col in metadata_cols if col in df_cleaned.columns]
        
        # Remove trailing and preceding NaNs from each column
        for col in df_cleaned.columns:
            if col != timestamp_col:  # Don't modify timestamp column
                # Find first and last non-NaN values
                first_valid = df_cleaned[col].first_valid_index()
                last_valid = df_cleaned[col].last_valid_index()
                
                if first_valid is not None and last_valid is not None:
                    # Keep only the range from first to last valid value
                    df_cleaned = df_cleaned.loc[first_valid:last_valid]
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
        
        # Handle automatically generated timestamps if needed
        if self.column_config.get('timestamp_col') == "AUTOMATICALLY GENERATED":
            timestamp_col = "timestamp"
            if timestamp_col not in df.columns:
                df[timestamp_col] = self._generate_timestamps(len(df))
            self.column_config['used_timestamp_col'] = timestamp_col
        
        # Handle combining separate datetime columns if needed
        if self.column_config.get('timestamp_col') == "COMBINE_DATETIME_COLUMNS":
            timestamp_col = "timestamp"
            if timestamp_col not in df.columns:
                df = self._combine_datetime_columns(df, timestamp_col)
            self.column_config['used_timestamp_col'] = timestamp_col
        
        if self.column_config.get('timestamp_col').find(" OR|| ") != -1:
            timestamp_col_candidates = self.column_config.get('timestamp_col').split(" OR|| ")
            for timestamp_col in timestamp_col_candidates:
                if timestamp_col in df.columns:
                    self.column_config['used_timestamp_col'] = timestamp_col
                    break
                
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
    
    def _create_windows_with_stride(self, target_series: np.ndarray, covariate_data: np.ndarray) -> None:
        """Create windows with stride parameter and handle incomplete windows."""
        data_length = len(target_series)
        
        # Minimum history length for forecasting (prioritize forecasting after 20 values)
        MIN_HISTORY_FORECAST_LENGTH = 20
        min_history_for_forecast = min(MIN_HISTORY_FORECAST_LENGTH, self.history_length)
        
        # Minimum total data required: 20 history + 20 forecast = 40 points
        min_total_data = MIN_HISTORY_FORECAST_LENGTH * 2
        
        # Check if dataset has minimum required data
        if data_length < min_total_data:
            # Dataset is too short for any meaningful forecasting (need at least 40 points)
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
                
                window = Window(history, target, covariates)
                windows.append(window)
                
            elif available_data >= MIN_HISTORY_FORECAST_LENGTH * 2:
                # Split available data equally between history and forecast
                actual_history_length = available_data // 2
                history_end = start_idx + actual_history_length
                target_start = history_end
                target_end = start_idx + available_data
                
                history = target_series[start_idx:history_end]
                target = target_series[target_start:target_end]
                covariates = covariate_data[start_idx:history_end]
                
                window = Window(history, target, covariates)
                windows.append(window)
            else:
                print(f"Skipping window with available data: {available_data}")                
            # Skip windows that don't meet minimum requirements
        return windows

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
            
            # Create sliding windows with stride and incomplete window handling
            windows = self._create_windows_with_stride(target_series, covariate_data)
            all_windows.extend(windows)
        return all_windows
    
    def __iter__(self) -> Iterator[Window]:
        """Iterate over windows. If there 
        are no more windows, load the next parquet file
        until there are no more windows to load.
        """
        return self._lazy_iterator()
    
    def _lazy_iterator(self) -> Iterator[Window]:
        """Lazy iterator that loads parquet files on-demand."""
        # Reset state for fresh iteration
        self._current_parquet_index = 0
        self._current_window_index = 0
        self._windows = []
        
        while self._current_parquet_index < len(self._parquet_files):
            # Load windows from current parquet file if not already loaded
            if not self._windows:
                parquet_file = self._parquet_files[self._current_parquet_index]
                self._windows = self._prep_windows_from_file(parquet_file)
                self._current_window_index = 0
            
            # Yield windows from current parquet file
            while self._current_window_index < len(self._windows):
                print(self._windows[self._current_window_index].history())
                yield self._windows[self._current_window_index]
                self._current_window_index += 1
            
            # Move to next parquet file
            self._current_parquet_index += 1
            self._windows = []  # Clear windows to force reload of next file
    
    def __len__(self) -> int:
        """Return total number of windows across all parquet files."""
        return self._total_windows
    
    def get_num_parquet_files(self) -> int:
        """Return number of parquet files in the dataset."""
        return len(self._parquet_files)
    
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
        
        # Compute mean metrics
        aggregated = {}
        for metric in all_results[0].keys():
            values = [result[metric] for result in all_results]
            aggregated[metric] = np.mean(values)
            aggregated[f"{metric}_std"] = np.std(values)
        
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
