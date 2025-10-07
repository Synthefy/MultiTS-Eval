"""
Window class representing a single evaluation unit.
"""

import numpy as np
from typing import Dict, Optional
from ..metrics import evaluate_metrics


class Window:
    """
    Represents a single evaluation unit (history, target, covariates).
    Stores ground truth and submitted forecast.
    """
    
    def __init__(self, history: np.ndarray, target: np.ndarray, covariates: np.ndarray, timestamps: Optional[np.ndarray] = None):
        """
        Initialize a window with history, target, covariates, and timestamps.
        
        Args:
            history: Historical data for forecasting
            target: Ground truth target values
            covariates: Additional covariate data
            timestamps: Timestamp data for the window (optional)
        """
        self._history = history.copy()
        self._target = target.copy()
        self._covariates = covariates.copy()
        self._timestamps = timestamps.copy() if timestamps is not None else None
        self._forecast: Optional[np.ndarray] = None
        self._evaluation_results: Optional[Dict[str, float]] = None
    
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
    
    def submit_forecast(self, multivariate_forecast: Optional[np.ndarray]=None, univariate_forecast: Optional[np.ndarray] = None) -> None:
        """
        Store forecast and trigger evaluation once for this window.
        
        Args:
            multivariate_forecast: Predicted values (multivariate forecast)
            univariate_forecast: Optional univariate forecast for comparison
        """
        # Store multivariate forecast
        self._forecast = multivariate_forecast.copy() if multivariate_forecast is not None else None
        
        # Store univariate forecast if provided
        if univariate_forecast is not None:
            self._univariate_forecast = univariate_forecast.copy()
            self._has_univariate = True
        else:
            self._has_univariate = False
        
        # Trigger evaluation immediately for both forecasts
        self._evaluation_results = evaluate_metrics(self._target, self._forecast) if self._forecast is not None else None
        
        self._univariate_evaluation_results = evaluate_metrics(self._target, self._univariate_forecast) if self._univariate_forecast is not None else None
    
    def evaluate(self, forecast_type: str = "multivariate") -> Dict[str, float]:
        """
        Run metrics and return cached results.
        
        Args:
            forecast_type: Type of forecast to evaluate - "multivariate" or "univariate"
        
        Returns:
            Dictionary of evaluation metrics
        """
        if forecast_type == "multivariate":
            if self._evaluation_results is None:
                raise ValueError("No multivariate forecast submitted yet. Call submit_forecast() first.")
            results = self._evaluation_results.copy()
            results['univariate'] = False
            return results
        elif forecast_type == "univariate":
            if not self._has_univariate or not hasattr(self, '_univariate_evaluation_results'):
                raise ValueError("No univariate forecast submitted yet. Call submit_forecast() with univariate_forecast parameter.")
            results = self._univariate_evaluation_results.copy()
            results['univariate'] = True
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
    def is_univariate(self) -> bool:
        """Check if the primary forecast is univariate."""
        return getattr(self, '_is_univariate', False)
    
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
        # Replace NaN values in history
        if np.any(np.isnan(self._history)):
            nanmean_history = np.nanmean(self._history)
            self._history = np.where(np.isnan(self._history), nanmean_history, self._history)
        
        # Replace NaN values in covariates
        if np.any(np.isnan(self._covariates)):
            nanmean_covariates = np.nanmean(self._covariates)
            self._covariates = np.where(np.isnan(self._covariates), nanmean_covariates, self._covariates)
        
        # Replace NaN values in forecast if it exists
        if self._forecast is not None and np.any(np.isnan(self._forecast)):
            nanmean_forecast = np.nanmean(self._forecast)
            self._forecast = np.where(np.isnan(self._forecast), nanmean_forecast, self._forecast)
        
        # Replace NaN values in timestamps if they exist and are numeric
        if self._timestamps is not None and np.issubdtype(self._timestamps.dtype, np.number):
            if np.any(np.isnan(self._timestamps)):
                nanmean_timestamps = np.nanmean(self._timestamps)
                self._timestamps = np.where(np.isnan(self._timestamps), nanmean_timestamps, self._timestamps)