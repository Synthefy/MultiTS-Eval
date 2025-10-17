"""
Window class representing a single evaluation unit.
"""

import numpy as np
from typing import Dict, Optional, List
from ..metrics import evaluate_metrics


class Window:
    """
    Represents a single evaluation unit (history, target, covariates).
    Stores ground truth and submitted forecast.
    Always operates in batch mode with shape [batch_size, ...].
    """
    
    def __init__(self, history: np.ndarray, target: np.ndarray, covariates: np.ndarray, timestamps: Optional[np.ndarray] = None, 
                 history_padding: Optional[np.ndarray] = None, target_padding: Optional[np.ndarray] = None):
        """
        Initialize a window with history, target, covariates, and timestamps.
        
        Args:
            history: Historical data for forecasting (shape: [batch_size, history_length])
            target: Ground truth target values (shape: [batch_size, forecast_horizon])
            covariates: Additional covariate data (shape: [batch_size, history_length, covariate_dim])
            timestamps: Timestamp data for the window (optional, shape: [batch_size, timestamp_length])
            history_padding: Padding mask for history (shape: [batch_size, history_length], True = padded)
            target_padding: Padding mask for target (shape: [batch_size, forecast_horizon], True = padded)
        """
        self._history = history.copy()
        self._target = target.copy()
        self._covariates = covariates.copy()
        self._timestamps = timestamps.copy() if timestamps is not None else None
        self._forecast: Optional[np.ndarray] = None
        self._evaluation_results: Optional[Dict[str, np.ndarray]] = None
        
        # Padding information
        self._history_padding = history_padding.copy() if history_padding is not None else np.zeros_like(history, dtype=bool)
        self._target_padding = target_padding.copy() if target_padding is not None else np.zeros_like(target, dtype=bool)
        
        # For univariate forecasts
        self._univariate_forecast: Optional[np.ndarray] = None
        self._univariate_evaluation_results: Optional[Dict[str, np.ndarray]] = None
        self._has_univariate = False
    
    def history(self) -> np.ndarray:
        """Return the historical data."""
        return self._history.copy()
    
    def target(self) -> np.ndarray:
        """Return the target values."""
        return self._target.copy()
    
    def covariates(self) -> np.ndarray:
        """Return the covariate data."""
        return self._covariates.copy()
    
    def timestamps(self) -> Optional[np.ndarray]:
        """Return the timestamp data."""
        return self._timestamps.copy() if self._timestamps is not None else None
    
    def history_padding(self) -> np.ndarray:
        """Return the history padding mask (True = padded positions)."""
        return self._history_padding.copy()
    
    def target_padding(self) -> np.ndarray:
        """Return the target padding mask (True = padded positions)."""
        return self._target_padding.copy()
    
    def get_unpadded_target(self) -> np.ndarray:
        """Return target values excluding padded positions."""
        unpadded_target = self._target.copy().astype(float)
        unpadded_target[self._target_padding] = np.nan
        return unpadded_target
    
    def get_unpadded_forecast(self) -> Optional[np.ndarray]:
        """Return forecast values excluding padded positions."""
        if self._forecast is None:
            return None
        unpadded_forecast = self._forecast.copy().astype(float)
        unpadded_forecast[self._target_padding] = np.nan
        return unpadded_forecast
    
    def submit_forecast(self, multivariate_forecast: Optional[np.ndarray]=None, univariate_forecast: Optional[np.ndarray] = None) -> None:
        """
        Store forecast and trigger evaluation once for this window.
        
        Args:
            multivariate_forecast: Predicted values (multivariate forecast, shape: [batch_size, forecast_horizon])
            univariate_forecast: Optional univariate forecast for comparison (shape: [batch_size, forecast_horizon])
        """
        # Store multivariate forecast
        if multivariate_forecast is not None:
            # Ensure multivariate forecast is properly batched
            if multivariate_forecast.ndim == 1:
                # Convert 1D array to 2D by adding batch dimension
                multivariate_forecast = multivariate_forecast.reshape(1, -1)
            
            # Validate dimensions match target
            if multivariate_forecast.shape != self._target.shape:
                raise ValueError(f"Multivariate forecast shape {multivariate_forecast.shape} does not match target shape {self._target.shape}")
            
            self._forecast = multivariate_forecast.copy()
            # Evaluate the entire batch at once using unpadded data and store raw vectors
            unpadded_target = self.get_unpadded_target()
            unpadded_forecast = self.get_unpadded_forecast()
            self._evaluation_results = evaluate_metrics(unpadded_target, unpadded_forecast)
        
        # Store univariate forecast if provided
        if univariate_forecast is not None:
            # Ensure univariate forecast is properly batched
            if univariate_forecast.ndim == 1:
                # Convert 1D array to 2D by adding batch dimension
                univariate_forecast = univariate_forecast.reshape(1, -1)
            
            # Validate dimensions match target
            if univariate_forecast.shape != self._target.shape:
                raise ValueError(f"Univariate forecast shape {univariate_forecast.shape} does not match target shape {self._target.shape}")
            
            self._univariate_forecast = univariate_forecast.copy()
            self._has_univariate = True
            # Evaluate univariate forecast using unpadded data and store raw vectors
            unpadded_target = self.get_unpadded_target()
            unpadded_univariate_forecast = self._univariate_forecast.copy().astype(float)
            unpadded_univariate_forecast[self._target_padding] = np.nan
            self._univariate_evaluation_results = evaluate_metrics(unpadded_target, unpadded_univariate_forecast)
        else:
            self._has_univariate = False
    
    def evaluate(self, forecast_type: str = "multivariate") -> Dict[str, np.ndarray]:
        """
        Run metrics and return cached results.
        
        Args:
            forecast_type: Type of forecast to evaluate - "multivariate" or "univariate"
        
        Returns:
            Dictionary of evaluation metric vectors (one per batch element)
        """
        if forecast_type == "multivariate":
            if self._evaluation_results is None:
                raise ValueError("No multivariate forecast submitted yet. Call submit_forecast() first.")
            results = self._evaluation_results.copy()
            return results
        elif forecast_type == "univariate":
            if not self._has_univariate or not hasattr(self, '_univariate_evaluation_results'):
                raise ValueError("No univariate forecast submitted yet. Call submit_forecast() with univariate_forecast parameter.")
            results = self._univariate_evaluation_results.copy()
            return results
        else:
            raise ValueError(f"Invalid forecast_type: {forecast_type}. Must be 'multivariate' or 'univariate'")
    
    @property
    def has_forecast(self) -> bool:
        """Check if forecast has been submitted."""
        return self._forecast is not None
    
    @property
    def has_univariate_forecast(self) -> bool:
        """Check if univariate forecast has been submitted."""
        return getattr(self, '_has_univariate', False)
    
    @property
    def has_multivariate_forecast(self) -> bool:
        """Check if multivariate forecast has been submitted."""
        return self._forecast is not None
    
    @property
    def batch_size(self) -> int:
        """Return the batch size."""
        return self._target.shape[0]
    
    @property
    def has_timestamps(self) -> bool:
        """Check if timestamps are available."""
        return self._timestamps is not None
    
    def replace_nan_with_nanmean(self) -> None:
        """
        Replace NaN values with nanmean in history, forecast, and covariates.
        
        This method modifies the internal data arrays in-place by replacing
        any NaN values with the nanmean of the respective array.
        """
        batch_size = self.batch_size
        
        # Replace NaN values in history
        if np.any(np.isnan(self._history)):
            for i in range(batch_size):
                if np.any(np.isnan(self._history[i])):
                    nanmean_history = np.nanmean(self._history[i])
                    self._history[i] = np.where(np.isnan(self._history[i]), nanmean_history, self._history[i])
        
        # Replace NaN values in covariates
        if np.any(np.isnan(self._covariates)):
            for i in range(batch_size):
                if np.any(np.isnan(self._covariates[i])):
                    nanmean_covariates = np.nanmean(self._covariates[i])
                    self._covariates[i] = np.where(np.isnan(self._covariates[i]), nanmean_covariates, self._covariates[i])
        
        # Replace NaN values in forecast if it exists
        if self._forecast is not None and np.any(np.isnan(self._forecast)):
            for i in range(batch_size):
                if np.any(np.isnan(self._forecast[i])):
                    nanmean_forecast = np.nanmean(self._forecast[i])
                    self._forecast[i] = np.where(np.isnan(self._forecast[i]), nanmean_forecast, self._forecast[i])
        
        # Replace NaN values in timestamps if they exist and are numeric
        if self._timestamps is not None and np.issubdtype(self._timestamps.dtype, np.number):
            if np.any(np.isnan(self._timestamps)):
                for i in range(batch_size):
                    if np.any(np.isnan(self._timestamps[i])):
                        nanmean_timestamps = np.nanmean(self._timestamps[i])
                        self._timestamps[i] = np.where(np.isnan(self._timestamps[i]), nanmean_timestamps, self._timestamps[i])