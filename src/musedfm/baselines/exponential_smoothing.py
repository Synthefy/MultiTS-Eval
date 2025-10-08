"""
Exponential smoothing baseline for MUSED-FM evaluation.
"""

import numpy as np
from typing import Optional
from musedfm.baselines.base_forecaster import BaseForecaster


class ExponentialSmoothing(BaseForecaster):
    """
    Simple exponential smoothing baseline.
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Initialize exponential smoothing forecaster.
        
        Args:
            alpha: Smoothing parameter (0 < alpha <= 1)
        """
        self.alpha = alpha
    
    def forecast(self, history: np.ndarray, covariates: Optional[np.ndarray] = None, forecast_horizon: Optional[int] = None) -> np.ndarray:
        """
        Generate exponential smoothing forecast.
        
        Args:
            history: Historical time series data
            covariates: Optional covariate data (ignored)
            forecast_horizon: Number of future points to forecast (default: 1)
            
        Returns:
            Exponential smoothing forecast
        """
        if forecast_horizon is None:
            forecast_horizon = 1
        
        # filter history with nanmean
        history[np.isnan(history)] = np.nanmean(history)

        if len(history) < 2:
            return np.full(forecast_horizon, history[0])
        
        # Simple exponential smoothing
        smoothed = history[0]
        for value in history[1:]:
            smoothed = self.alpha * value + (1 - self.alpha) * smoothed
        
        # Return smoothed value repeated for forecast horizon
        return np.full(forecast_horizon, smoothed)
