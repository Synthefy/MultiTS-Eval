"""
Debugging functionality for MUSED-FM examples.

This module contains functions for:
- NaN value tracking, reporting and statistics
- High MAPE detection and plotting
- Debug output and diagnostics
"""

import numpy as np
from typing import Dict, Any
from musedfm.plotting import plot_window_forecasts


def _initialize_nan_tracking() -> Dict[str, int]:
    """Initialize NaN tracking statistics for a dataset."""
    return {
        'total_windows': 0,
        'windows_with_nan_history': 0,
        'windows_with_nan_target': 0,
        'windows_with_nan_covariates': 0,
        'windows_with_any_nan': 0,
        'history_nan_count': 0,
        'target_nan_count': 0,
        'covariates_nan_count': 0
    }


def _update_nan_tracking(nan_stats: Dict[str, int], window_nan_stats: Dict[str, Any]) -> None:
    """Update NaN tracking statistics with window-level data."""
    nan_stats['total_windows'] += 1
    
    if window_nan_stats['history_nans'] > 0:
        nan_stats['windows_with_nan_history'] += 1
        nan_stats['history_nan_count'] += window_nan_stats['history_nans']
    
    if window_nan_stats['target_nans'] > 0:
        nan_stats['windows_with_nan_target'] += 1
        nan_stats['target_nan_count'] += window_nan_stats['target_nans']
    
    if window_nan_stats['covariates_nans'] > 0:
        nan_stats['windows_with_nan_covariates'] += 1
        nan_stats['covariates_nan_count'] += window_nan_stats['covariates_nans']
    
    if window_nan_stats['any_nans']:
        nan_stats['windows_with_any_nan'] += 1


def _check_window_nan_values(window) -> Dict[str, Any]:
    """Check for NaN values in a single window and return statistics."""
    # Check history for NaN values
    history = window.history()
    history_nans = np.isnan(history).sum()
    
    # Check target for NaN values
    target = window.target()
    target_nans = np.isnan(target).sum()
    
    # Check covariates for NaN values
    covariates = window.covariates()
    covariates_nans = np.isnan(covariates).sum()
    
    return {
        'history_nans': history_nans,
        'target_nans': target_nans,
        'covariates_nans': covariates_nans,
        'any_nans': history_nans > 0 or target_nans > 0 or covariates_nans > 0
    }


def _report_nan_statistics(nan_stats: Dict[str, int]) -> None:
    """Report NaN statistics for a dataset."""
    if nan_stats['windows_with_any_nan'] > 0:
        print("  ⚠ NaN values detected:")
        print(f"    Windows with NaN: {nan_stats['windows_with_any_nan']}/{nan_stats['total_windows']}")
        if nan_stats['windows_with_nan_history'] > 0:
            print(f"    History NaN: {nan_stats['windows_with_nan_history']} windows, {nan_stats['history_nan_count']} values")
        if nan_stats['windows_with_nan_target'] > 0:
            print(f"    Target NaN: {nan_stats['windows_with_nan_target']} windows, {nan_stats['target_nan_count']} values")
        if nan_stats['windows_with_nan_covariates'] > 0:
            print(f"    Covariates NaN: {nan_stats['windows_with_nan_covariates']} windows, {nan_stats['covariates_nan_count']} values")
    else:
        print(f"  ✓ No NaN values detected in {nan_stats['total_windows']} windows")


def plot_high_mape_windows(model_name: str, dataset_name: str, dataset, model: Dict, 
                          dataset_avg_metrics: Dict[str, float], max_windows: int = 3, save_path: str = "",
                          forecast_type: str = "multivariate") -> None:
    """Plot sample windows for models with high MAPE values.
    
    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        dataset: Dataset object to iterate through
        model: Model dictionary containing model and univariate flag
        dataset_avg_metrics: Dictionary containing average metrics
        max_windows: Maximum number of windows to plot
        save_path: Path to save the plots
        forecast_type: Type of forecast to generate ("multivariate" or "univariate")
    """
    if dataset_avg_metrics.get('MAPE', np.nan) > 1000 or np.isnan(dataset_avg_metrics.get('MAPE', np.nan)):
        print(f"  🔍 High MAPE detected for {model_name} on {dataset_name} - plotting sample windows to /{save_path}/{model_name}_{dataset_name}.png")
        # Plot first few windows to see what's happening
        for plot_idx, window in enumerate(dataset):
            if plot_idx >= max_windows:  # Only plot first few windows
                break
            
            # Get the forecast for this specific window
            target_length = len(window.target())
            
            if forecast_type == "univariate":
                sample_forecast = model["model"].forecast(window.history(), None, target_length)
            else:  # multivariate
                if model["univariate"]:
                    sample_forecast = model["model"].forecast(window.history(), None, target_length)
                else:
                    sample_forecast = model["model"].forecast(window.history(), window.covariates(), target_length)
            
            if sample_forecast is not None:
                plot_window_forecasts(
                    window, 
                    {model_name: sample_forecast}, 
                    f"{model_name} - {dataset_name} - Window {plot_idx+1}", 
                    figsize=(12, 6), 
                    save_path=f"{save_path}/{model_name}_{dataset_name}_window_{plot_idx+1}.png"
                )


def debug_model_performance(model_name: str, dataset_avg_metrics: Dict[str, float], 
                          model_dataset_windows: Dict[str, int], model_elapsed_time: float) -> None:
    """Print debug information for model performance."""
    mape_value = dataset_avg_metrics.get('MAPE', 'N/A')
    nmae_value = dataset_avg_metrics.get('NMAE', 'N/A')
    if isinstance(mape_value, (int, float)) and not np.isnan(mape_value):
        mape_str = f"{mape_value:.2f}%"
    else:   
        mape_str = "N/A"
    if isinstance(nmae_value, (int, float)) and not np.isnan(nmae_value):
        nmae_str = f"{nmae_value:.4f}"
    else:
        nmae_str = "N/A"
    
    print(f"    {model_name}: {model_dataset_windows[model_name]} windows in {model_elapsed_time:.2f}s with MAPE {mape_str} and NMAE {nmae_str}")


def debug_univariate_performance(univariate_model_name: str, univariate_avg_metrics: Dict[str, float],
                                model_dataset_windows: Dict[str, int], model_elapsed_time: float, model_name: str) -> None:
    """Print debug information for univariate model performance."""
    mape_value = univariate_avg_metrics.get('MAPE', 'N/A')
    if isinstance(mape_value, (int, float)) and not np.isnan(mape_value):
        mape_str = f"{mape_value:.2f}%"
    else:
        mape_str = "N/A"
    
    print(f"    {univariate_model_name}: {model_dataset_windows[model_name]} windows in {model_elapsed_time:.2f}s with MAPE {mape_str}")


def debug_forecast_failure(model_name: str, forecast_type: str) -> None:
    """Print debug information when forecast generation fails."""
    print(f"  ⚠ {model_name} {forecast_type} forecast returned None")


def debug_forecast_length_mismatch(model_name: str, forecast_length: int, target_length: int) -> None:
    """Print debug information when forecast length doesn't match target length."""
    print(f"  ⚠ Forecast length mismatch: model '{model_name}' returned {forecast_length} values, but target has {target_length} values")


def debug_model_summary(model_name: str, results: Dict[str, Any], models: Dict[str, Any]) -> None:
    """Print debug summary for a model."""
    overall_avg_metrics = results[model_name]['metrics']
    
    # Determine model type for display
    if model_name.endswith('_univariate'):
        model_type = 'Univariate'
    else:
        model_type = 'Univariate' if models[model_name]['univariate'] else 'Multivariate'
    
    print(f"\n{model_name} Summary:")
    print(f"  Total windows: {results[model_name]['windows']}")
    print(f"  Total time: {results[model_name]['time']:.2f}s")
    print(f"  Model type: {model_type}")
    if overall_avg_metrics:
        print(f"  Average MAPE: {overall_avg_metrics['MAPE']:.2f}%")
        print(f"  Average MAE: {overall_avg_metrics['MAE']:.4f}")
        print(f"  Average RMSE: {overall_avg_metrics['RMSE']:.4f}")
        print(f"  Average NMAE: {overall_avg_metrics['NMAE']:.4f}")
    
    # Display category and domain results
    for level_name, level_data in [('Category', results[model_name]['category_results']), 
                                  ('Domain', results[model_name]['domain_results'])]:
        if level_data:
            print(f"  {level_name} Results:")
            for name, data in level_data.items():
                metrics = data['metrics']
                print(f"    {name}: {data['dataset_count']} datasets, {data['window_count']} windows")
                if metrics:
                    print(f"      MAPE: {metrics['MAPE']:.2f}%, MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, NMAE: {metrics['NMAE']:.4f}")
