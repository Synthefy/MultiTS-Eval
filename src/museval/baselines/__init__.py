"""
Baseline forecasting methods for MUSED-FM evaluation.
"""

from museval.baselines.base_forecaster import BaseForecaster
from museval.baselines.mean_forecast import MeanForecast
from museval.baselines.historical_inertia import HistoricalInertia
from museval.baselines.arima_forecast import ARIMAForecast
from museval.baselines.linear_trend import LinearTrend
from museval.baselines.exponential_smoothing import ExponentialSmoothing
from museval.baselines.linear_regression import LinearRegressionForecast

__all__ = [
    'BaseForecaster',
    'MeanForecast',
    'HistoricalInertia',
    'ARIMAForecast',
    'LinearTrend',
    'ExponentialSmoothing',
    'LinearRegressionForecast'
]
