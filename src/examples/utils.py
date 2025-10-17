"""
Utility functions for MUSED-FM examples.

This module contains helper functions for:
- Model parsing and configuration
- Metrics aggregation
- NaN value tracking and reporting
- Results aggregation by different levels
- Timeout handling for model forecasting
"""

import numpy as np
import signal
from typing import Dict, List, Tuple, Any
from contextlib import contextmanager


def _aggregate_metrics(dataset_results: List[Dict], metric_names: List[str] = ['MAPE', 'MAE', 'RMSE', 'NMAE']) -> Tuple[Dict[str, float], int, int]:
    """Helper function to aggregate metrics from dataset results."""
    if not dataset_results:
        return {}, 0, 0
    
    total_windows = sum(result['window_count'] for result in dataset_results)
    dataset_count = len(dataset_results)
    
    avg_metrics = {}
    for metric in metric_names:
        values = [result['metrics'][metric] for result in dataset_results if metric in result['metrics']]
        if values:
            # Convert to numpy array and use nanmean to skip NaN values
            values_array = np.array(values)
            avg_metrics[metric] = np.nanmean(values_array)
        else:
            avg_metrics[metric] = np.nan
    
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

@contextmanager
def timeout_handler(seconds):
    """Context manager for timeout handling in model forecasting.
    
    Args:
        seconds: Timeout duration in seconds
        
    Raises:
        TimeoutError: If the operation exceeds the timeout duration
    """
    def timeout_signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old signal handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
