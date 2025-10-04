"""
Base forecaster abstract class for MUSED-FM baselines.
"""

import numpy as np
from typing import Optional
from abc import ABC, abstractmethod


class BaseForecaster(ABC):
    """Abstract base class for forecasting methods."""
    
    @abstractmethod
    def forecast(self, history: np.ndarray, covariates: Optional[np.ndarray] = None, forecast_horizon: Optional[int] = None) -> np.ndarray:
        """
        Generate forecast from historical data.
        
        Args:
            history: Historical time series data
            covariates: Optional covariate data
            forecast_horizon: Number of future points to forecast (if None, will be determined from context)
            
        Returns:
            Forecast values
        """
        pass
