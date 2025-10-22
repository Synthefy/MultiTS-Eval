"""
Linear trend baseline for MUSED-FM evaluation.
"""

import numpy as np
from typing import Optional
from multieval.baselines.base_forecaster import BaseForecaster


class LinearTrend(BaseForecaster):
    """
    Simple linear trend baseline that fits a linear trend to historical data.
    """
    
    def forecast(self, history: np.ndarray, covariates: Optional[np.ndarray] = None, forecast_horizon: Optional[int] = None, timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forecast using linear trend extrapolation.
        
        Args:
            history: Historical time series data (shape: [batch_size, history_length])
            covariates: Optional covariate data (ignored)
            forecast_horizon: Number of future points to forecast (default: 1)
            timestamps: Optional timestamp data (ignored)
            forecast_horizon: Number of future points to forecast (default: 1)
            
        Returns:
            Linear trend forecast (shape: [batch_size, forecast_horizon])
        """
        if forecast_horizon is None:
            forecast_horizon = 1
        
        batch_size = history.shape[0]
        forecasts = []
        
        for i in range(batch_size):
            history_i = history[i].copy()
            
            # filter history with nanmean
            history_i[np.isnan(history_i)] = np.nanmean(history_i)

            # Fit linear trend
            x = np.arange(len(history_i))
            coeffs = np.polyfit(x, history_i, 1)
            
            # Extrapolate for forecast horizon
            forecast_i = []
            for j in range(forecast_horizon):
                next_x = len(history_i) + j
                forecast_value = coeffs[0] * next_x + coeffs[1]
                forecast_i.append(forecast_value)
            
            forecasts.append(np.array(forecast_i))
        
        return np.stack(forecasts, axis=0)  # [batch_size, forecast_horizon]
