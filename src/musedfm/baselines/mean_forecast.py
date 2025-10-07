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
    
    def forecast(self, history: np.ndarray, covariates: Optional[np.ndarray] = None, forecast_horizon: Optional[int] = None) -> np.ndarray:
        """
        Forecast the mean of historical data.
        
        Args:
            history: Historical time series data
            covariates: Optional covariate data (ignored)
            forecast_horizon: Number of future points to forecast (default: 1)
            
        Returns:
            Array of mean values with length equal to forecast horizon
        """
        if forecast_horizon is None:
            forecast_horizon = 1
        
        # Convert to numeric, skipping non-numeric values
        try:
            # Try to convert the entire array to numeric
            history_numeric = pd.to_numeric(history, errors='coerce')
            # Remove NaN values (including non-numeric values that became NaN)
            history_clean = history_numeric[~np.isnan(history_numeric)]
            
            if len(history_clean) == 0:
                # If no valid numeric data, return zeros
                return np.zeros(forecast_horizon)
            
            mean_value = np.mean(history_clean)
            # Return array with mean value repeated for forecast horizon
            return np.full(forecast_horizon, mean_value)
        except (ValueError, TypeError):
            # If conversion fails entirely, return zeros
            return np.zeros(forecast_horizon)
