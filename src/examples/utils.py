"""
Utility functions for MUSED-FM examples.

This module contains helper functions for:
- Model parsing and configuration
- Metrics aggregation
- NaN value tracking and reporting
- Results aggregation by different levels
"""

import numpy as np
from typing import Dict, List, Tuple, Any


def get_available_models():
    """Get dictionary of available forecasting models.
    
    To add your own custom model:
    1. Create a class that implements the forecast() method
    2. Add it to this dictionary with a descriptive name
    3. The forecast() method should accept: history, covariates, forecast_length
    4. It should return a numpy array of forecasts
    
    Example custom model:
    class MyCustomModel:
        def __init__(self, param1=1.0, param2=2.0):
            self.param1 = param1
            self.param2 = param2
        
        def forecast(self, history, covariates, forecast_length):
            # Your custom forecasting logic here
            # history: numpy array of historical values
            # covariates: numpy array of covariate values (can be None)
            # forecast_length: number of steps to forecast
            return np.array([np.mean(history) * self.param1] * forecast_length)
    
    Then add to the dictionary:
    "my_custom": MyCustomModel(param1=1.5, param2=3.0)
    """
    from musedfm.baselines import (
        MeanForecast, 
        HistoricalInertia, 
        ARIMAForecast, 
        LinearTrend, 
        ExponentialSmoothing
    )
    from musedfm.baselines.linear_regression import LinearRegressionForecast
    
    return {
        "mean": {"model": MeanForecast(), "univariate": True},
        "historical_inertia": {"model": HistoricalInertia(), "univariate": True},
        "linear_trend": {"model": LinearTrend(), "univariate": True},
        "exponential_smoothing": {"model": ExponentialSmoothing(), "univariate": True},
        "arima": {"model": ARIMAForecast(order=(1, 1, 1)), "univariate": True},
        "linear_regression": {"model": LinearRegressionForecast(), "univariate": False}
        # Add your custom models here:
        # "my_custom": {"model": MyCustomModel(), "univariate": False},
        # "another_model": {"model": AnotherModel(param1=value1, param2=value2), "univariate": True}
    }


def parse_models(model_string: str) -> Dict[str, Any]:
    """Parse model string and return list of model instances."""
    available_models = get_available_models()
    
    if model_string.lower() == "all":
        return available_models
    
    model_names = [name.strip().lower() for name in model_string.split(",")]
    selected_models = {}
    
    for name in model_names:
        if name in available_models:
            selected_models[name] = available_models[name]
        else:
            print(f"Warning: Unknown model '{name}'. Available models: {list(available_models.keys())}")
    
    return selected_models


def _aggregate_metrics(dataset_results: List[Dict], metric_names: List[str] = ['MAPE', 'MAE', 'RMSE', 'NMAE']) -> Tuple[Dict[str, float], int, int]:
    """Helper function to aggregate metrics from dataset results."""
    if not dataset_results:
        return {}, 0, 0
    
    total_windows = sum(result['window_count'] for result in dataset_results)
    dataset_count = len(dataset_results)
    
    avg_metrics = {}
    for metric in metric_names:
        values = [result['metrics'][metric] for result in dataset_results if metric in result['metrics']]
        avg_metrics[metric] = np.mean(values) if values else np.nan
    
    return avg_metrics, total_windows, dataset_count


def _aggregate_results_by_level(results: Dict, models: Dict, benchmark, level_name: str, level_attr: str) -> None:
    """Helper function to aggregate results by category or domain level."""
    for model_name in models.keys():
        # Collect dataset results for this level
        level_results = []
        for dataset_result in results[model_name]['dataset_results']:
            if level_name in dataset_result['dataset_name']:
                level_results.append(dataset_result)
        
        if level_results:
            avg_metrics, total_windows, dataset_count = _aggregate_metrics(level_results)
            results[model_name][f'{level_attr}_results'][level_name] = {
                'metrics': avg_metrics,
                'window_count': total_windows,
                'dataset_count': dataset_count
            }
        
        # Process univariate results for multivariate models
        if not models[model_name]["univariate"]:
            univariate_model_name = f"{model_name}_univariate"
            univariate_level_results = []
            for dataset_result in results[univariate_model_name]['dataset_results']:
                if level_name in dataset_result['dataset_name']:
                    univariate_level_results.append(dataset_result)
            
            if univariate_level_results:
                avg_metrics, total_windows, dataset_count = _aggregate_metrics(univariate_level_results)
                results[univariate_model_name][f'{level_attr}_results'][level_name] = {
                    'metrics': avg_metrics,
                    'window_count': total_windows,
                    'dataset_count': dataset_count
                }


