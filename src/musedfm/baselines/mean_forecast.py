"""
Mean forecast baseline for MUSED-FM evaluation.
"""

import numpy as np
import pandas as pd
from typing import Optional
from musedfm.baselines.base_forecaster import BaseForecaster


class MeanForecast(BaseForecaster):
    """
    Simple baseline that forecasts the mean of historical data.
    """
    
    def forecast(self, history: np.ndarray, covariates: Optional[np.ndarray] = None, forecast_horizon: Optional[int] = None, timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forecast the mean of historical data using vectorized operations.
        
        Args:
            history: Historical time series data (shape: [batch_size, history_length])
            covariates: Optional covariate data (ignored)
            forecast_horizon: Number of future points to forecast (default: 1)
            timestamps: Optional timestamp data (ignored)
            
        Returns:
            Array of mean values with length equal to forecast horizon (shape: [batch_size, forecast_horizon])
        """
        if forecast_horizon is None:
            forecast_horizon = 1
        
        # Convert to numeric, skipping non-numeric values (vectorized)
        history_numeric = pd.to_numeric(history.flatten(), errors='coerce').reshape(history.shape)
        
        # Calculate mean for each batch element, ignoring NaN values
        batch_means = np.nanmean(history_numeric, axis=1)
        
        # Check for batch elements with no valid data
        valid_counts = np.sum(~np.isnan(history_numeric), axis=1)
        empty_batches = valid_counts == 0
        
        # Initialize result array
        forecasts = np.zeros((history.shape[0], forecast_horizon))
        
        # For batches with valid data, use the mean
        valid_mask = ~empty_batches
        if np.any(valid_mask):
            forecasts[valid_mask] = batch_means[valid_mask, np.newaxis]
        
        # Batches with no valid data remain zeros (already initialized)
        return forecasts
