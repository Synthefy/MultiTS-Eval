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
    
    def forecast(self, history: np.ndarray, covariates: Optional[np.ndarray] = None, forecast_horizon: Optional[int] = None, timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate exponential smoothing forecast.
        
        Args:
            history: Historical time series data (shape: [batch_size, history_length])
            covariates: Optional covariate data (ignored)
            forecast_horizon: Number of future points to forecast (default: 1)
            timestamps: Optional timestamp data (ignored)
            forecast_horizon: Number of future points to forecast (default: 1)
            
        Returns:
            Exponential smoothing forecast (shape: [batch_size, forecast_horizon])
        """
        if forecast_horizon is None:
            forecast_horizon = 1
        
        batch_size = history.shape[0]
        forecasts = []
        
        for i in range(batch_size):
            history_i = history[i].copy()
            
            # filter history with nanmean
            history_i[np.isnan(history_i)] = np.nanmean(history_i)

            if len(history_i) < 2:
                forecast_i = np.full(forecast_horizon, history_i[0])
            else:
                # Simple exponential smoothing
                smoothed = history_i[0]
                for value in history_i[1:]:
                    smoothed = self.alpha * value + (1 - self.alpha) * smoothed
                
                # Return smoothed value repeated for forecast horizon
                forecast_i = np.full(forecast_horizon, smoothed)
            
            forecasts.append(forecast_i)
        
        return np.stack(forecasts, axis=0)  # [batch_size, forecast_horizon]
