"""
Historical inertia baseline for MUSED-FM evaluation.
"""

import numpy as np
from typing import Optional
from musedfm.baselines.base_forecaster import BaseForecaster


class HistoricalInertia(BaseForecaster):
    """
    Historical inertia baseline that uses the last observed value.
    Based on the concept from https://arxiv.org/pdf/2103.16349
    """
    
    def forecast(self, history: np.ndarray, covariates: Optional[np.ndarray] = None, forecast_horizon: Optional[int] = None) -> np.ndarray:
        """
        Forecast using historical inertia (last observed values).
        
        Args:
            history: Historical time series data
            covariates: Optional covariate data (ignored)
            forecast_horizon: Number of future points to forecast (default: 1)
            
        Returns:
            Array with last observed values for forecast horizon
        """
        if forecast_horizon is None:
            forecast_horizon = 1
        
        # filter history with nanmean
        history[np.isnan(history)] = np.nanmean(history)

        # Use minimum of forecast horizon or history length
        inertia_length = min(forecast_horizon, len(history))
        
        # Get the last inertia_length values from history
        last_values = history[-inertia_length:]
        
        # If forecast horizon is longer than history, repeat the last value
        if forecast_horizon > len(history):
            # Pad with the last value
            forecast = np.concatenate([
                last_values,
                np.full(forecast_horizon - len(history), last_values[-1])
            ])
        else:
            forecast = last_values
        
        return forecast
