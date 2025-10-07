"""
Baseline forecasting methods for MUSED-FM evaluation.
"""

from musedfm.baselines.base_forecaster import BaseForecaster
from musedfm.baselines.mean_forecast import MeanForecast
from musedfm.baselines.historical_inertia import HistoricalInertia
from musedfm.baselines.arima_forecast import ARIMAForecast
from musedfm.baselines.linear_trend import LinearTrend
from musedfm.baselines.exponential_smoothing import ExponentialSmoothing

__all__ = [
    'BaseForecaster',
    'MeanForecast',
    'HistoricalInertia',
    'ARIMAForecast',
    'LinearTrend',
    'ExponentialSmoothing'
]
