"""
Historical inertia baseline for MUSED-FM evaluation.
"""

import numpy as np
from typing import Optional
from museval.baselines.base_forecaster import BaseForecaster


class HistoricalInertia(BaseForecaster):
    """
    Historical inertia baseline that uses the last observed value.
    Based on the concept from https://arxiv.org/pdf/2103.16349
    """
    
    def forecast(self, history: np.ndarray, covariates: Optional[np.ndarray] = None, forecast_horizon: Optional[int] = None, timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forecast using historical inertia (last observed values).
        
        Args:
            history: Historical time series data (shape: [batch_size, history_length])
            covariates: Optional covariate data (ignored)
            forecast_horizon: Number of future points to forecast (default: 1)
            timestamps: Optional timestamp data (ignored)
            forecast_horizon: Number of future points to forecast (default: 1)
            
        Returns:
            Array with last observed values for forecast horizon (shape: [batch_size, forecast_horizon])
        """
        if forecast_horizon is None:
            forecast_horizon = 1
        
        batch_size = history.shape[0]
        forecasts = []
        
        for i in range(batch_size):
            # filter history with nanmean
            history_i = history[i].copy()
            history_i[np.isnan(history_i)] = np.nanmean(history_i)

            # Use minimum of forecast horizon or history length
            inertia_length = min(forecast_horizon, len(history_i))
            
            # Get the last inertia_length values from history
            last_values = history_i[-inertia_length:]
            
            # If forecast horizon is longer than history, repeat the last value
            if forecast_horizon > len(history_i):
                # Pad with the last value
                forecast_i = np.concatenate([
                    last_values,
                    np.full(forecast_horizon - len(history_i), last_values[-1])
                ])
            else:
                forecast_i = last_values
            
            forecasts.append(forecast_i)
        
        return np.stack(forecasts, axis=0)  # [batch_size, forecast_horizon]
