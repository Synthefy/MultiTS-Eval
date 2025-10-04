"""
Linear trend baseline for MUSED-FM evaluation.
"""

import numpy as np
from typing import Optional
from .base_forecaster import BaseForecaster


class LinearTrend(BaseForecaster):
    """
    Simple linear trend baseline that fits a linear trend to historical data.
    """
    
    def forecast(self, history: np.ndarray, covariates: Optional[np.ndarray] = None, forecast_horizon: Optional[int] = None) -> np.ndarray:
        """
        Forecast using linear trend extrapolation.
        
        Args:
            history: Historical time series data
            covariates: Optional covariate data (ignored)
            forecast_horizon: Number of future points to forecast (default: 1)
            
        Returns:
            Linear trend forecast
        """
        if forecast_horizon is None:
            forecast_horizon = 1
        
        # Fit linear trend
        x = np.arange(len(history))
        coeffs = np.polyfit(x, history, 1)
        
        # Extrapolate for forecast horizon
        forecasts = []
        for i in range(forecast_horizon):
            next_x = len(history) + i
            forecast_value = coeffs[0] * next_x + coeffs[1]
            forecasts.append(forecast_value)
        
        return np.array(forecasts)
