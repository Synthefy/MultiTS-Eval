"""
Baseline forecasting methods for MUSED-FM evaluation.
"""

from .base_forecaster import BaseForecaster
from .mean_forecast import MeanForecast
from .historical_inertia import HistoricalInertia
from .arima_forecast import ARIMAForecast
from .linear_trend import LinearTrend
from .exponential_smoothing import ExponentialSmoothing

__all__ = [
    'BaseForecaster',
    'MeanForecast',
    'HistoricalInertia',
    'ARIMAForecast',
    'LinearTrend',
    'ExponentialSmoothing'
]
