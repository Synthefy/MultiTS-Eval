"""
ARIMA forecast baseline for MUSED-FM evaluation.
"""

import numpy as np
from typing import Optional
from museval.baselines.base_forecaster import BaseForecaster


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
    
    def forecast(self, history: np.ndarray, covariates: Optional[np.ndarray] = None, forecast_horizon: Optional[int] = None, timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate ARIMA forecast.
        
        Args:
            history: Historical time series data (shape: [batch_size, history_length])
            covariates: Optional covariate data (ignored)
            forecast_horizon: Number of future points to forecast (default: 1)
            timestamps: Optional timestamp data (ignored)
            forecast_horizon: Number of future points to forecast (default: 1)
            
        Returns:
            ARIMA forecast values (shape: [batch_size, forecast_horizon])
        """
        if forecast_horizon is None:
            forecast_horizon = 1
        
        batch_size = history.shape[0]
        forecasts = []
        
        for i in range(batch_size):
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            
            # filter history with nanmean
            history_i = history[i].copy()
            history_i[np.isnan(history_i)] = np.nanmean(history_i)

            history_std, history_mean = np.std(history_i), np.mean(history_i)
            history_i = (history_i - history_mean) / max(history_std, 1e-10)
            if np.std(history_i) < 1e-2:
                # if the variance is too low, just return the mean
                forecast_i = np.full(forecast_horizon, history_mean)
            else:
                # Fit ARIMA model
                model = SARIMAX(history_i, order=self.order, enforce_stationarity=False, enforce_invertibility=False)
                fitted_model = model.fit(method='lbfgs', maxiter=200, disp=False)
                
                # Generate forecast
                forecast_i = fitted_model.forecast(steps=forecast_horizon)
                forecast_i = forecast_i * history_std + history_mean
                
                # Handle different return types from statsmodels
                if hasattr(forecast_i, 'iloc'):
                    forecast_i = forecast_i.iloc[:forecast_horizon].values
                elif hasattr(forecast_i, '__getitem__'):
                    forecast_i = np.array([forecast_i[j] for j in range(forecast_horizon)])
                else:
                    forecast_i = np.array([float(forecast_i)] * forecast_horizon)
            
            forecasts.append(forecast_i)
        
        return np.stack(forecasts, axis=0)  # [batch_size, forecast_horizon]
