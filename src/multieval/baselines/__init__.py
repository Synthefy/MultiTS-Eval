"""
Baseline forecasting methods for MUSED-FM evaluation.
"""

from multieval.baselines.base_forecaster import BaseForecaster
from multieval.baselines.mean_forecast import MeanForecast
from multieval.baselines.historical_inertia import HistoricalInertia
from multieval.baselines.arima_forecast import ARIMAForecast
from multieval.baselines.linear_trend import LinearTrend
from multieval.baselines.exponential_smoothing import ExponentialSmoothing
from multieval.baselines.linear_regression import LinearRegressionForecast

__all__ = [
    'BaseForecaster',
    'MeanForecast',
    'HistoricalInertia',
    'ARIMAForecast',
    'LinearTrend',
    'ExponentialSmoothing',
    'LinearRegressionForecast'
]
