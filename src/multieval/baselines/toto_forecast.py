"""
DataDog ToTo forecasting model for MUSED-FM evaluation.

ToTo is a Time-Series-Optimized Transformer for Observability, developed by DataDog.
It's a foundation model for multivariate time series forecasting with a focus on 
observability metrics.

RUN INSTRUCTIONS:
================

1. SERVER-BASED USAGE (Recommended):
   ```bash
   # Start the ToTo server
   cd toto-server
   uv run python src/toto_server/server_main.py
   
   # Set environment variable for server URL
   export TOTO_SERVER_URL="http://localhost:8000"
   ```

2. USAGE IN MUSED-FM:
   ```python
   # Programmatic usage
   from multieval.baselines.toto_forecast import TotoForecast
   
   # Initialize ToTo model (will connect to server)
   model = TotoForecast()
   
   # Generate forecast
   history = np.random.randn(1, 100)  # [batch_size, history_length]
   forecast = model.forecast(history, forecast_horizon=10)
   ```
"""

import numpy as np
import requests
import os
from typing import Optional, Dict, Any
from .utils import handle_nans, standard_normalize

class TotoForecast:
    """
    DataDog ToTo forecasting model wrapper for MUSED-FM evaluation.
    
    This implementation uses DataDog's ToTo model via a server API.
    """
    
    def __init__(self, server_url: Optional[str] = None, num_samples: int = 8, max_chunk_size: int = 8):
        """
        Initialize DataDog ToTo forecast model.
        
        Args:
            server_url: URL of the ToTo server (defaults to environment variable TOTO_SERVER_URL or localhost:8000)
            num_samples: Number of samples for probabilistic forecasting
            max_chunk_size: Maximum batch size per chunk to avoid server payload limits
        """
        self.server_url = server_url or os.getenv("TOTO_SERVER_URL", "http://localhost:8000")
        self.server_url = self.server_url.rstrip("/")
        self.num_samples = num_samples
        self.max_chunk_size = max_chunk_size  # Maximum batch size per chunk
        self._check_server_health()
    
    def _check_server_health(self):
        """Check if the ToTo server is running and healthy."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            response.raise_for_status()
            print(f"✓ ToTo server health check passed: {response.json().get('status', 'UNKNOWN')}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to ToTo server at {self.server_url}: {e}")
    
    def _validate_timestamps(self, timestamps: np.ndarray) -> None:
        """Validate that timestamps are evenly spaced."""
        if len(timestamps) < 2:
            return
        
        # Convert timestamps to numeric - handle various types
        try:
            # Try direct conversion to float first (for numeric types)
            timestamps_numeric = timestamps.astype(np.float64)
        except (ValueError, TypeError):
            # If that fails, try converting from string/datetime
            try:
                # First convert to datetime64, then to float64
                timestamps_datetime = np.array(timestamps, dtype='datetime64')
                timestamps_numeric = timestamps_datetime.astype(np.float64)
            except (ValueError, TypeError):
                # If all conversion fails, skip validation
                return
        
        time_deltas = np.diff(timestamps_numeric)
        if not np.allclose(time_deltas, time_deltas[0]):
            print(f"Warning: ToTo assumes evenly spaced timestamps. Found varying time deltas: {time_deltas}")
    
    def _convert_timestamps_to_seconds(self, timestamps: np.ndarray) -> np.ndarray:
        """Convert timestamps to seconds since epoch."""
        # Try direct conversion first (for numeric types)
        try:
            return timestamps.astype("datetime64[s]").astype(np.float64)
        except (ValueError, TypeError):
            # If that fails, try converting from string/datetime
            try:
                # First convert to datetime64, then to seconds
                timestamps_datetime = np.array(timestamps, dtype='datetime64')
                return timestamps_datetime.astype("datetime64[s]").astype(np.float64)
            except (ValueError, TypeError):
                # If all conversion fails, create dummy timestamps
                return np.arange(len(timestamps), dtype=np.float64)
    
    def _get_time_delta_seconds(self, timestamps_in_seconds: np.ndarray) -> float:
        """Get time delta in seconds from timestamps."""
        if len(timestamps_in_seconds) < 2:
            return 1.0  # Default 1 second
        return float(np.diff(timestamps_in_seconds)[-1])
    
    def forecast(self, history: np.ndarray, covariates: Optional[np.ndarray] = None, forecast_horizon: Optional[int] = None, timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate forecast from historical data using DataDog ToTo via server API.
        
        Args:
            history: Historical time series data (shape: [batch_size, history_length])
            covariates: Optional covariate data (shape: [batch_size, history_length, covariate_dim])
            forecast_horizon: Number of future points to forecast (default: 1)
            timestamps: Optional timestamp data (shape: [batch_size, history_length])
            
        Returns:
            Forecast values (shape: [batch_size, forecast_horizon])
        """
        if forecast_horizon is None:
            forecast_horizon = 1
        
        batch_size = history.shape[0]
        history_length = history.shape[1]
        forecasts = np.zeros((batch_size, forecast_horizon))
        
        
        # Check if we have sufficient data
        if history.shape[1] < 1:
            return forecasts
        
        # Normalize the data using standardized utils functions
        history_normalized, history_mean, history_std = standard_normalize(
            history, 
            axis=1, 
            keepdims=True,
            epsilon=1e-6
        )
        # Handle NaN values - replace with 0.0 for processing
        history_processed = np.nan_to_num(history_normalized, nan=0.0)
        
        # Use provided timestamps or create dummy timestamps
        if timestamps is not None:
            # Ensure timestamps match history length
            if timestamps.shape[1] >= history_length:
                timestamps_batch = timestamps[0][:history_length]  # Use first batch element's timestamps, truncated to history length
            else:
                # If timestamps are shorter than history, create dummy timestamps
                timestamps_batch = np.arange(history_length, dtype=np.float64)
        else:
            timestamps_batch = np.arange(history_length, dtype=np.float64)
        
        # Validate timestamps
        self._validate_timestamps(timestamps_batch)
        
        # Convert timestamps to seconds
        timestamps_seconds = self._convert_timestamps_to_seconds(timestamps_batch)
        time_delta_seconds = self._get_time_delta_seconds(timestamps_seconds)
        
        # Determine if we have covariates (multivariate) or not (univariate)
        if covariates is not None:
            # Multivariate case: combine history and covariates
            covariates_processed = handle_nans(covariates, method="zero")
            
            # Calculate number of variates
            num_variates = 1 + covariates.shape[2] if len(covariates.shape) == 3 else 1
            
            # Vectorized preparation of input series
            # Stack history and covariates along the variate dimension: [batch_size, num_variates, T]
            if len(covariates.shape) == 3:
                # Transpose covariates from [batch_size, T, covariate_dim] to [batch_size, covariate_dim, T]
                covariates_transposed = np.transpose(covariates_processed, (0, 2, 1))  # [batch_size, covariate_dim, T]
                
                # Combine history and covariates: [batch_size, num_variates, T]
                input_series_array = np.concatenate([
                    history_processed[:, np.newaxis, :],  # [batch_size, 1, T]
                    covariates_transposed  # [batch_size, covariate_dim, T]
                ], axis=1)  # [batch_size, num_variates, T]
            else:
                input_series_array = history_processed[:, np.newaxis, :]  # [batch_size, 1, T]
            
            # Convert to list format for API
            input_series_batch = input_series_array.astype(np.float32).tolist()
            
            # Create timestamp and time_interval arrays once
            timestamp_seconds_list = timestamps_seconds.astype(np.float32).tolist()
            time_interval_seconds_list = [time_delta_seconds] * num_variates
            
            # Vectorized creation of timestamp and time_interval batches
            timestamp_seconds_batch = [[timestamp_seconds_list] * num_variates for _ in range(batch_size)]
            time_interval_seconds_batch = [time_interval_seconds_list for _ in range(batch_size)]
        else:
            # Univariate case: only history
            # Vectorized preparation: [batch_size, 1, T]
            input_series_array = history_processed[:, np.newaxis, :]  # [batch_size, 1, T]
            input_series_batch = input_series_array.astype(np.float32).tolist()
            
            # Create timestamp and time_interval arrays once
            timestamp_seconds_list = timestamps_seconds.astype(np.float32).tolist()
            time_interval_seconds_list = [time_delta_seconds]
            
            # Vectorized creation of timestamp and time_interval batches
            timestamp_seconds_batch = [[timestamp_seconds_list] for _ in range(batch_size)]
            time_interval_seconds_batch = [time_interval_seconds_list for _ in range(batch_size)]
        
        # Ensure num_samples is divisible by batch_size (sampling_batch_size)
        adjusted_num_samples = ((self.num_samples + batch_size - 1) // batch_size) * batch_size
        
        payload: Dict[str, Any] = {
            "input_series": input_series_batch,
            "timestamp_seconds": timestamp_seconds_batch,
            "time_interval_seconds": time_interval_seconds_batch,
            "prediction_length": int(forecast_horizon),
            "num_samples": int(adjusted_num_samples),
            "samples_per_batch": int(batch_size),
        }
        
        # Process in chunks if batch is too large
        if batch_size <= self.max_chunk_size:
            # Single API call for small batches
            response = requests.post(
                f"{self.server_url}/forecast", 
                json=payload, 
                timeout=60
            )
            response.raise_for_status()
            
            result_json = response.json()
            api_result = np.asarray(result_json["result"], dtype=np.float32)
            
            # Extract forecasts - take only the first variate (target series)
            forecasts = api_result[:, 0, :]  # Extract [batch_size, forecast_horizon]
            
        else:
            # Process in chunks for large batches
            forecasts = np.zeros((batch_size, forecast_horizon))
            
            for chunk_start in range(0, batch_size, self.max_chunk_size):
                chunk_end = min(chunk_start + self.max_chunk_size, batch_size)
                chunk_size = chunk_end - chunk_start
                
                # Ensure num_samples is divisible by chunk_size for this chunk
                chunk_adjusted_num_samples = ((self.num_samples + chunk_size - 1) // chunk_size) * chunk_size
                
                # Create chunk payload
                chunk_payload = {
                    "input_series": payload["input_series"][chunk_start:chunk_end],
                    "timestamp_seconds": payload["timestamp_seconds"][chunk_start:chunk_end],
                    "time_interval_seconds": payload["time_interval_seconds"][chunk_start:chunk_end],
                    "prediction_length": payload["prediction_length"],
                    "num_samples": int(chunk_adjusted_num_samples),
                    "samples_per_batch": chunk_size,
                }
                
                # Call API for this chunk
                response = requests.post(
                    f"{self.server_url}/forecast", 
                    json=chunk_payload, 
                    timeout=60
                )
                response.raise_for_status()
                
                result_json = response.json()
                api_result = np.asarray(result_json["result"], dtype=np.float32)
                
                # Extract forecasts for this chunk
                chunk_forecasts = api_result[:, 0, :]  # Extract [chunk_size, forecast_horizon]
                forecasts[chunk_start:chunk_end] = chunk_forecasts
        
        # Ensure correct length for all forecasts
        if forecasts.shape[1] != forecast_horizon:
            if forecasts.shape[1] > forecast_horizon:
                forecasts = forecasts[:, :forecast_horizon]
            else:
                # Pad with last values
                last_values = forecasts[:, -1:] if forecasts.shape[1] > 0 else np.zeros((batch_size, 1))
                padding_length = forecast_horizon - forecasts.shape[1]
                padding = np.tile(last_values, (1, padding_length))
                forecasts = np.concatenate([forecasts, padding], axis=1)
        
        return forecasts
