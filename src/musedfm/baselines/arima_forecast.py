"""
ARIMA forecast baseline for MUSED-FM evaluation.
"""

import numpy as np
from typing import Optional
from .base_forecaster import BaseForecaster


class ARIMAForecast(BaseForecaster):
    """
    ARIMA baseline forecasting method.
    """
    
    def __init__(self, order: tuple = (1, 1, 1)):
        """
        Initialize ARIMA forecaster.
        
        Args:
            order: ARIMA order (p, d, q)
        """
        self.order = order
        self._model = None
    
    def forecast(self, history: np.ndarray, covariates: Optional[np.ndarray] = None, forecast_horizon: Optional[int] = None) -> np.ndarray:
        """
        Generate ARIMA forecast.
        
        Args:
            history: Historical time series data
            covariates: Optional covariate data (ignored)
            forecast_horizon: Number of future points to forecast (default: 1)
            
        Returns:
            ARIMA forecast values
        """
        if forecast_horizon is None:
            forecast_horizon = 1
        
        from statsmodels.tsa.arima.model import ARIMA
        
        # Fit ARIMA model
        model = ARIMA(history, order=self.order)
        fitted_model = model.fit()
        
        # Generate forecast
        forecast = fitted_model.forecast(steps=forecast_horizon)
        
        # Handle different return types from statsmodels
        if hasattr(forecast, 'iloc'):
            return forecast.iloc[:forecast_horizon].values
        elif hasattr(forecast, '__getitem__'):
            return np.array([forecast[i] for i in range(forecast_horizon)])
        else:
            return np.array([float(forecast)] * forecast_horizon)
