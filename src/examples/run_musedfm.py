"""
Example demonstrating how to run multiple forecasting models with MUSED-FM.

This example shows how to:
1. Load datasets from the MUSED-FM benchmark
2. Run multiple baseline forecasting models
3. Compare model performance across different metrics
4. Generate visualizations and export results

Usage:
    python run_musedfm.py --models mean,arima,linear --data-path /path/to/dataset
    python run_musedfm.py --models all --windows 50
    python run_musedfm.py --help
"""

import time
import warnings
import numpy as np
import argparse
from pathlib import Path

# Import ConvergenceWarning for proper suppression
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
except ImportError:
    # Fallback if ConvergenceWarning is not available
    ConvergenceWarning = Warning
from musedfm.data import Benchmark
from musedfm.baselines import (
    MeanForecast, 
    HistoricalInertia, 
    ARIMAForecast, 
    LinearTrend, 
    ExponentialSmoothing
)

from musedfm.baselines.linear_regression import LinearRegressionForecast
from musedfm.plotting import export_metrics_to_csv
import warnings as w

# Suppress specific statsmodels warnings about ARIMA parameter initialization
# These warnings occur when ARIMA models encounter non-invertible MA parameters or non-stationary AR parameters
# during initialization. statsmodels automatically handles this by using zeros as starting parameters,
# which is a standard fallback approach. The warnings are informational and don't affect model performance.
warnings.filterwarnings("ignore", 
                       message="Non-invertible starting MA parameters found. Using zeros as starting parameters.",
                       category=UserWarning,
                       module="statsmodels.tsa.statespace.sarimax")

warnings.filterwarnings("ignore", 
                       message="Non-stationary starting autoregressive parameters found. Using zeros as starting parameters.",
                       category=UserWarning,
                       module="statsmodels.tsa.statespace.sarimax")

# Suppress statsmodels convergence warnings
warnings.filterwarnings("ignore", 
                       message="Maximum Likelihood optimization failed to",
                       category=Warning,
                       module="statsmodels")

# Alternative: suppress all warnings from statsmodels.base.model module
warnings.filterwarnings("ignore", 
                       module="statsmodels.base.model")

# Additional suppression for ConvergenceWarning
warnings.filterwarnings("ignore", 
                       message="Maximum Likelihood optimization failed to converge. Check mle_retvals",
                       category=Warning)

# Suppress all ConvergenceWarning instances
warnings.filterwarnings("ignore", 
                       message=".*converge.*",
                       category=Warning,
                       module="statsmodels")

# Nuclear option: suppress all statsmodels warnings
warnings.filterwarnings("ignore", 
                       module="statsmodels")

# Additional targeted suppression for the specific warning
warnings.filterwarnings("ignore", 
                       message="Maximum Likelihood optimization failed to converge.\ Check mle_retvals",
                       category=ConvergenceWarning)

# Suppress ConvergenceWarning from statsmodels.base.model specifically
warnings.filterwarnings("ignore", 
                       category=ConvergenceWarning,
                       module="statsmodels.base.model")

# Additional comprehensive suppression
warnings.filterwarnings("ignore", 
                       message="Maximum Likelihood optimization failed to converge. Check mle_retvals",
                       category=ConvergenceWarning,
                       module="statsmodels.base.model")

# Suppress all warnings from the specific line that's causing issues
warnings.filterwarnings("ignore", 
                       message="Maximum Likelihood optimization failed to converge. Check mle_retvals",
                       category=Warning)

# Suppress Chronos prediction length warnings
warnings.filterwarnings("ignore", 
                       message="We recommend keeping prediction length <= 64. The quality of longer predictions may degrade since the model is not optimized for it.",
                       category=UserWarning,
                       module="chronos")

# Suppress all Chronos warnings
warnings.filterwarnings("ignore", 
                       module="chronos")

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


def parse_models(model_string):
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


def _aggregate_metrics(dataset_results, metric_names=['MAPE', 'MAE', 'RMSE', 'NMAE']):
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


def _initialize_nan_tracking():
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


def _update_nan_tracking(nan_stats, window_nan_stats):
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


def _report_nan_statistics(nan_stats):
    """Report NaN statistics for a dataset."""
    if nan_stats['windows_with_any_nan'] > 0:
        print(f"  ⚠ NaN values detected:")
        print(f"    Windows with NaN: {nan_stats['windows_with_any_nan']}/{nan_stats['total_windows']}")
        if nan_stats['windows_with_nan_history'] > 0:
            print(f"    History NaN: {nan_stats['windows_with_nan_history']} windows, {nan_stats['history_nan_count']} values")
        if nan_stats['windows_with_nan_target'] > 0:
            print(f"    Target NaN: {nan_stats['windows_with_nan_target']} windows, {nan_stats['target_nan_count']} values")
        if nan_stats['windows_with_nan_covariates'] > 0:
            print(f"    Covariates NaN: {nan_stats['windows_with_nan_covariates']} windows, {nan_stats['covariates_nan_count']} values")
    else:
        print(f"  ✓ No NaN values detected in {nan_stats['total_windows']} windows")


def _check_window_nan_values(window):
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


def _aggregate_results_by_level(results, models, benchmark, level_name, level_attr):
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


def run_models_on_benchmark(benchmark_path: str, models: dict, max_windows: int = 100, 
                           collections: str = None, domains: str = None, datasets: str = None,
                           collect_plot_data: bool = False, history_length: int = 512, 
                           forecast_horizon: int = 128, stride: int = 256, load_cached_counts: bool = False):
    """Run multiple forecasting models on a benchmark and compare their performance."""
    print("=" * 60)
    print("Running Multiple Models on Benchmark")
    print("=" * 60)
    
    benchmark = Benchmark(benchmark_path, history_length=history_length, forecast_horizon=forecast_horizon, stride=stride, load_cached_counts=load_cached_counts)
    print(f"Loaded benchmark with {len(benchmark)} categories")
    print(f"Running {len(models)} models: {list(models.keys())}")
    
    results = {}
    plot_data = []  # Store windows and forecasts for plotting
    
    # Initialize results for each model
    for model_name in models.keys():
        results[model_name] = {
            'metrics': {},
            'dataset_results': [],
            'category_results': {},
            'domain_results': {},
            'time': 0.0,
            'windows': 0
        }
        
        # Initialize univariate results only for non-univariate models
        if not models[model_name]["univariate"]:
            univariate_model_name = f"{model_name}_univariate"
            results[univariate_model_name] = {
                'metrics': {},
                'dataset_results': [],
                'category_results': {},
                'domain_results': {},
                'time': 0.0,
                'windows': 0
            }
    
    # Iterate through benchmark structure: category -> domain -> dataset (outer loop)
    dataset_count = 0
    skip_datasets = 0  # DEBUG: Change this to skip the first N datasets for debugging
    
    for category in benchmark:
        for domain in category:
            for dataset in domain:
                # Apply filters if specified
                if collections is not None and category.category_path.name not in collections:
                    continue
                if domains is not None and domain.domain_path.name not in domains:
                    continue
                if datasets is not None and dataset.data_path.name not in datasets:
                    continue
                
                # Skip first few datasets for debugging
                if dataset_count < skip_datasets:
                    dataset_count += 1
                    print(f"Skipping dataset {dataset_count}: {dataset.data_path.name}")
                    continue
                
                # Get full dataset name from benchmark path
                dataset_name = str(dataset.data_path.relative_to(benchmark.benchmark_path))
                print(f"\nProcessing dataset {dataset_count + 1}: {dataset_name}")
                dataset_count += 1
                
                # Process all models for this dataset (inner loop)
                print(f"  Processing {max_windows if max_windows is not None else 'all'} windows with {len(models)} models...")
                
                # Initialize per-model tracking for this dataset
                model_dataset_windows = {model_name: 0 for model_name in models.keys()}
                model_dataset_results = {model_name: [] for model_name in models.keys()}
                # Add univariate results tracking only for non-univariate models
                for model_name, model in models.items():
                    if not model["univariate"]:
                        model_dataset_results[f"{model_name}_univariate"] = []
                model_start_times = {model_name: time.time() for model_name in models.keys()}
                
                # Initialize NaN tracking for this dataset
                nan_stats = _initialize_nan_tracking()
                
                for i, window in enumerate(dataset):
                    # Check max_windows per dataset, not overall
                    if max_windows is not None and i >= max_windows:
                        break
                    
                    # Check for NaN values in this window
                    window_nan_stats = _check_window_nan_values(window)
                    _update_nan_tracking(nan_stats, window_nan_stats)
                    
                    # Process this window with all models
                    for model_name, model in models.items():
                        target_length = len(window.target())
                        
                        # Generate multivariate forecast if model supports it (do this first for proper training)
                        multivariate_forecast = None
                        if not model["univariate"]:
                            multivariate_forecast = model["model"].forecast(window.history(), window.covariates(), target_length)
                        
                        # Generate univariate forecast (all models)
                        univariate_forecast = model["model"].forecast(window.history(), None, target_length)
                        
                        # Validate forecast length matches target length
                        if len(univariate_forecast) != target_length:
                            raise ValueError(f"Forecast length mismatch: model '{model_name}' returned {len(univariate_forecast)} values, but target has {target_length} values")
                        
                        # Plot linear regression fit for first window of linear models (temporary debugging)
                        # Do this after multivariate forecast to ensure model is trained with covariates
                        if i == 0 and 'linear' in model_name.lower() and hasattr(model["model"], 'plot_linear_fit'):
                            print(f"  📊 Plotting linear regression fit for {model_name}...")
                            try:
                                model["model"].plot_linear_fit(
                                    window.history(), 
                                    window.covariates(),
                                    save_path=f"/tmp/{model_name}_linear_fit_window_{i}.png"
                                )
                            except Exception as e:
                                print(f"  ⚠ Could not plot linear regression fit: {e}")
                        
                        # Submit both forecasts
                        window.submit_forecast(multivariate_forecast, univariate_forecast)
                        
                        # Get evaluation results for multivariate forecast if submitted
                        if multivariate_forecast is not None:
                            multivariate_results = window.evaluate("multivariate")
                            model_dataset_results[model_name].append(multivariate_results)
                        
                        # Store univariate results
                        if univariate_forecast is not None:
                            univariate_results = window.evaluate("univariate")
                            # For univariate models, store in main results; for multivariate models, store in _univariate
                            if model["univariate"]:
                                model_dataset_results[model_name].append(univariate_results)
                            else:
                                model_dataset_results[f"{model_name}_univariate"].append(univariate_results)
                        
                        # Collect data for plotting if requested
                        if collect_plot_data and results[model_name]['windows'] < 3:  # Only collect first 3 windows for plotting
                            plot_data.append({
                                'window': window,
                                'forecast': multivariate_forecast,
                                'model_name': model_name,
                                'window_index': results[model_name]['windows'],
                                'dataset_name': dataset.data_path.name
                            })
                        
                        model_dataset_windows[model_name] += 1
                        results[model_name]['windows'] += 1
                        
                        # Also count windows for univariate results if available
                        if not model["univariate"] and univariate_forecast is not None:
                            univariate_model_name = f"{model_name}_univariate"
                            results[univariate_model_name]['windows'] += 1
                
                # Calculate average metrics and store results for each model
                for model_name in models.keys():
                    # Calculate average metrics for this model on this dataset
                    if model_dataset_results[model_name]:
                        dataset_avg_metrics = {}
                        for metric in model_dataset_results[model_name][0].keys():
                            values = [result[metric] for result in model_dataset_results[model_name]]
                            dataset_avg_metrics[metric] = np.mean(values)
                    else:
                        dataset_avg_metrics = {}
                    
                    # Store dataset results for this model
                    results[model_name]['dataset_results'].append({
                        'dataset_name': dataset_name,
                        'metrics': dataset_avg_metrics,
                        'window_count': model_dataset_windows[model_name]
                    })
                    
                    model_elapsed_time = time.time() - model_start_times[model_name]
                    results[model_name]['time'] += model_elapsed_time
                    
                    print(f"    {model_name}: {model_dataset_windows[model_name]} windows in {model_elapsed_time:.2f}s")
                
                # Process univariate results for multivariate models only
                for model_name, model in models.items():
                    if not model["univariate"] and f"{model_name}_univariate" in model_dataset_results:
                        univariate_model_name = f"{model_name}_univariate"
                        
                        # Calculate average metrics for univariate version
                        if model_dataset_results[univariate_model_name]:
                            univariate_avg_metrics = {}
                            for metric in model_dataset_results[univariate_model_name][0].keys():
                                values = [result[metric] for result in model_dataset_results[univariate_model_name]]
                                univariate_avg_metrics[metric] = np.mean(values)
                        else:
                            univariate_avg_metrics = {}
                        
                        # Store univariate dataset results
                        results[univariate_model_name]['dataset_results'].append({
                            'dataset_name': dataset_name,
                            'metrics': univariate_avg_metrics,
                            'window_count': model_dataset_windows[model_name]  # Same window count as main model
                        })
                        
                        # Use same elapsed time as main model
                        results[univariate_model_name]['time'] += model_elapsed_time
                        
                        print(f"    {univariate_model_name}: {model_dataset_windows[model_name]} windows in {model_elapsed_time:.2f}s")
                
                # Report NaN statistics for this dataset
                _report_nan_statistics(nan_stats)
                
                print(f"  Completed dataset {dataset_name}")
        
        # Aggregate domain-level metrics for this domain
        _aggregate_results_by_level(results, models, benchmark, domain.domain_name, 'domain')
    
    # Aggregate category-level metrics for each category
    for category in benchmark:
        _aggregate_results_by_level(results, models, benchmark, category.category, 'category')
        
    # Calculate overall average metrics across all datasets for each model
    all_model_names = list(models.keys())
    # Add univariate model names for multivariate models only
    for model_name, model in models.items():
        if not model["univariate"]:
            all_model_names.append(f"{model_name}_univariate")
    
    for model_name in all_model_names:
        if results[model_name]['dataset_results']:
            overall_avg_metrics, _, _ = _aggregate_metrics(results[model_name]['dataset_results'])
        else:
            overall_avg_metrics = {}
        
        results[model_name]['metrics'] = overall_avg_metrics
        
        # Determine model type for display
        if model_name.endswith('_univariate'):
            model_type = 'Univariate'
            base_model_name = model_name.replace('_univariate', '')
        else:
            model_type = 'Univariate' if models[model_name]['univariate'] else 'Multivariate'
            base_model_name = model_name
        
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
    
    if collect_plot_data:
        results['_plot_data'] = plot_data
    
    return results


def generate_forecast_plots(results: dict, output_dir: str = "/tmp"):
    """Generate forecast metrics CSV using pre-computed data from run_models_on_benchmark."""
    print("\n" + "=" * 60)
    print("Generating Forecast Metrics CSV")
    print("=" * 60)
    
    if not results or '_plot_data' not in results:
        print("Warning: No results data available")
        return False
    
    print("Generating metrics CSV from results")
    
    # Export metrics to CSV using the full results dictionary
    success = export_metrics_to_csv(results, output_dir)
    
    if success:
        print("Generated metrics CSV files")
        print(f"CSV files saved to {output_dir}/")
    
    return success




def _clean_output_directories(output_dir: str):
    """Clean output directories before generating new CSV files."""
    import os
    import shutil
    
    directories_to_clean = [
        os.path.join(output_dir, "categories"),
        os.path.join(output_dir, "domains"),
        os.path.join(output_dir, "datasets")
    ]
    
    for directory in directories_to_clean:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"✓ Cleaned directory: {directory}")
        os.makedirs(directory, exist_ok=True)


def export_hierarchical_results_to_csv(results: dict, output_dir: str = "/tmp"):
    """Export results organized by category, domain, and dataset levels."""
    import os
    import pandas as pd
    
    print("\n" + "=" * 60)
    print("Exporting Hierarchical Results to CSV")
    print("=" * 60)
    
    # Clean output directories first
    _clean_output_directories(output_dir)
    
    # Create output directories
    category_dir = os.path.join(output_dir, "categories")
    domain_dir = os.path.join(output_dir, "domains") 
    dataset_dir = os.path.join(output_dir, "datasets")
    
    os.makedirs(category_dir, exist_ok=True)
    os.makedirs(domain_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Export category-level results
    for model_name, model_results in results.items():
        if 'category_results' in model_results and model_results['category_results']:
            category_data = []
            for category_name, category_info in model_results['category_results'].items():
                row = {
                    'model': model_name,
                    'category': category_name,
                    'datasets': category_info['dataset_count'],
                    'windows': category_info['window_count'],
                    **category_info['metrics']
                }
                category_data.append(row)
            
            if category_data:
                df = pd.DataFrame(category_data)
                df.to_csv(os.path.join(category_dir, f"{model_name}_category_results.csv"), index=False)
                print(f"✓ Category results saved: {model_name}_category_results.csv")
    
    # Export domain-level results
    for model_name, model_results in results.items():
        if 'domain_results' in model_results and model_results['domain_results']:
            domain_data = []
            for domain_name, domain_info in model_results['domain_results'].items():
                row = {
                    'model': model_name,
                    'domain': domain_name,
                    'datasets': domain_info['dataset_count'],
                    'windows': domain_info['window_count'],
                    **domain_info['metrics']
                }
                domain_data.append(row)
            
            if domain_data:
                df = pd.DataFrame(domain_data)
                df.to_csv(os.path.join(domain_dir, f"{model_name}_domain_results.csv"), index=False)
                print(f"✓ Domain results saved: {model_name}_domain_results.csv")
    
    # Export dataset-level results
    for model_name, model_results in results.items():
        if 'dataset_results' in model_results and model_results['dataset_results']:
            dataset_data = []
            for dataset_info in model_results['dataset_results']:
                row = {
                    'model': model_name,
                    'dataset': dataset_info['dataset_name'],
                    'windows': dataset_info['window_count'],
                    **dataset_info['metrics']
                }
                dataset_data.append(row)
            
            if dataset_data:
                df = pd.DataFrame(dataset_data)
                df.to_csv(os.path.join(dataset_dir, f"{model_name}_dataset_results.csv"), index=False)
                print(f"✓ Dataset results saved: {model_name}_dataset_results.csv")
    
    print(f"\nHierarchical CSV files saved to:")
    print(f"  Categories: {category_dir}/")
    print(f"  Domains: {domain_dir}/")
    print(f"  Datasets: {dataset_dir}/")

def export_results_to_csv(benchmark_path: str, models: dict, max_windows: int = None, output_dir: str = "/tmp",
                         collections: str = None, domains: str = None, datasets: str = None,
                         history_length: int = 512, forecast_horizon: int = 128, stride: int = 256, load_cached_counts: bool = False):
    """Export forecast results to CSV files."""
    import os
    import glob
    
    print("\n" + "=" * 60)
    print("Exporting Results to CSV")
    print("=" * 60)
    
    # Clean existing CSV files in output directory
    csv_patterns = [
        os.path.join(output_dir, "musedfm_results*.csv"),
        os.path.join(output_dir, "*_results.csv")
    ]
    
    for pattern in csv_patterns:
        for csv_file in glob.glob(pattern):
            try:
                os.remove(csv_file)
                print(f"✓ Cleaned CSV file: {os.path.basename(csv_file)}")
            except OSError as e:
                print(f"Warning: Could not remove {csv_file}: {e}")
    
    benchmark = Benchmark(benchmark_path, history_length=history_length, forecast_horizon=forecast_horizon, stride=stride, load_cached_counts=load_cached_counts)
    
    # Process windows with the first model (for CSV export, we only need one model's forecasts)
    first_model = next(iter(models.values()))["model"]
    model_name = next(iter(models.keys()))
    
    print(f"Processing {max_windows if max_windows is not None else 'all'} windows per dataset with {model_name} for CSV export...")
    
    total_windows = 0
    
    # Iterate through benchmark structure: category -> domain -> dataset
    for category in benchmark:
        for domain in category:
            for dataset in domain:
                # Apply filters if specified
                if collections is not None and category.category_path.name not in collections:
                    continue
                if domains is not None and domain.domain_path.name not in domains:
                    continue
                if datasets is not None and dataset.data_path.name not in datasets:
                    continue
                
                # Get full dataset name from benchmark path
                dataset_name = str(dataset.data_path.relative_to(benchmark.benchmark_path))
                print(f"  Processing dataset: {dataset_name}")
                dataset_windows = 0
                
                for i, window in enumerate(dataset):
                    # Check max_windows per dataset, not overall
                    if max_windows is not None and dataset_windows >= max_windows:
                        break
                    
                    target_length = len(window.target())
                    forecast = first_model.forecast(window.history(), window.covariates(), target_length)
                    
                    # Validate forecast length matches target length
                    if len(forecast) != target_length:
                        raise ValueError(f"Forecast length mismatch: model '{model_name}' returned {len(forecast)} values, but target has {target_length} values")
                    
                    # Get univariate flag for this model
                    is_univariate = models[model_name]["univariate"]
                    window.submit_forecast(forecast, univariate=is_univariate)
                    dataset_windows += 1
                    total_windows += 1
                
                print(f"    Processed {dataset_windows} windows from {dataset_name}")
    
    # Export CSV using benchmark's method
    output_path = f"{output_dir}/musedfm_results.csv"
    benchmark.to_results_csv(output_path)
    
    # Check if files were created
    if Path(output_path).exists():
        print(f"✓ Results CSV created: {output_path}")
    else:
        print("✗ Results CSV not created")
    
    aggregated_path = f"{output_dir}/musedfm_results_aggregated.csv"
    if Path(aggregated_path).exists():
        print(f"✓ Aggregated CSV created: {aggregated_path}")
    else:
        print("✗ Aggregated CSV not created")
    
    return True


def compare_model_performance(results: dict):
    """Compare and display model performance metrics."""
    print("\n" + "=" * 60)
    print("Model Performance Comparison")
    print("=" * 60)
    
    # Filter out plot data from results
    model_results = {k: v for k, v in results.items() if k != '_plot_data'}
    
    if not model_results:
        print("No model results to compare")
        return
    
    # Create comparison table
    print(f"{'Model':<20} {'MAPE (%)':<10} {'MAE':<10} {'RMSE':<10} {'NMAE':<10} {'Time (s)':<10}")
    print("-" * 80)
    
    for model_name, result in model_results.items():
        metrics = result['metrics']
        # Handle missing metrics
        mape = metrics.get('MAPE', np.nan)
        mae = metrics.get('MAE', np.nan)
        rmse = metrics.get('RMSE', np.nan)
        nmae = metrics.get('NMAE', np.nan)
        print(f"{model_name:<20} {mape:<10.2f} {mae:<10.4f} "
              f"{rmse:<10.4f} {nmae:<10.4f} {result['time']:<10.2f}")
    
    # Find best performing models (only for models with valid metrics)
    valid_models = {k: v for k, v in model_results.items() if v['metrics'] and 'MAPE' in v['metrics']}
    if valid_models:
        best_mape = min(valid_models.items(), key=lambda x: x[1]['metrics']['MAPE'])
        best_mae = min(valid_models.items(), key=lambda x: x[1]['metrics']['MAE'])
        best_rmse = min(valid_models.items(), key=lambda x: x[1]['metrics']['RMSE'])
        fastest = min(model_results.items(), key=lambda x: x[1]['time'])
        
        print("\nBest Performance:")
        print(f"  Lowest MAPE: {best_mape[0]} ({best_mape[1]['metrics']['MAPE']:.2f}%)")
        print(f"  Lowest MAE:  {best_mae[0]} ({best_mae[1]['metrics']['MAE']:.4f})")
        print(f"  Lowest RMSE: {best_rmse[0]} ({best_rmse[1]['metrics']['RMSE']:.4f})")
        print(f"  Fastest:     {fastest[0]} ({fastest[1]['time']:.2f}s)")
    else:
        print("\nBest Performance: No valid models found")


def main():
    """Main function demonstrating multiple model usage with MUSED-FM."""
    parser = argparse.ArgumentParser(
        description="MUSED-FM Example: Run multiple forecasting models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_musedfm.py --models mean,arima --benchmark-path /path/to/benchmark
  python run_musedfm.py --models all --windows 50 --plots --csv
  python run_musedfm.py --models linear_trend,exponential_smoothing --windows 20
  python run_musedfm.py --models all --collections Traditional --domains Energy
  python run_musedfm.py --models mean --datasets al_daily,bitcoin_price --plots
        """
    )
    
    parser.add_argument(
        "--models", 
        type=str, 
        default="mean,linear_trend",
        help="Comma-separated list of models to run, or 'all' for all available models"
    )
    
    parser.add_argument(
        "--benchmark-path",
        type=str,
        default="/home/caleb/musedfm_data",
        help="Path to the benchmark directory containing collections"
    )
    
    parser.add_argument(
        "--collections",
        type=str,
        help="Comma-separated list of collections to filter by (e.g., 'Traditional,Synthetic')"
    )
    
    parser.add_argument(
        "--domains",
        type=str,
        help="Comma-separated list of domains to filter by (e.g., 'Energy,Finance')"
    )
    
    parser.add_argument(
        "--datasets",
        type=str,
        help="Comma-separated list of datasets to filter by (e.g., 'al_daily,bitcoin_price')"
    )
    
    parser.add_argument(
        "--windows",
        type=int,
        default=None,
        help="Maximum number of windows to process per dataset (default: None for all windows)"
    )
    
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate forecast plots"
    )
    
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Export results to CSV files"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp",
        help="Output directory for plots and CSV files (default: /tmp)"
    )
    
    parser.add_argument(
        "--history-length",
        type=int,
        default=512,
        help="History length for windows (default: 512)"
    )
    
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=128,
        help="Forecast horizon for windows (default: 128)"
    )
    
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
        help="Stride between windows (default: 256)"
    )
    
    parser.add_argument(
        "--load-cached-counts",
        action="store_true",
        help="Load window counts from cached JSON files instead of generating"
    )
    
    args = parser.parse_args()
    
    print("MUSED-FM Example: Multiple Model Forecasting")
    print("=" * 60)
    print(f"Models: {args.models}")
    print(f"Benchmark path: {args.benchmark_path}")
    print(f"Collections: {args.collections or 'All'}")
    print(f"Domains: {args.domains or 'All'}")
    print(f"Datasets: {args.datasets or 'All'}")
    print(f"Max windows per dataset: {args.windows}")
    print(f"Generate plots: {args.plots}")
    print(f"Export CSV: {args.csv}")
    print(f"Output directory: {args.output_dir}")
    
    # Parse models
    models = parse_models(args.models)
    if not models:
        print("Error: No valid models specified")
        return 1
    
    # Parse filter arguments
    collections = args.collections.split(',') if args.collections else None
    domains = args.domains.split(',') if args.domains else None
    datasets = args.datasets.split(',') if args.datasets else None
    
    start_time = time.time()
    
    # Run models on benchmark
    results = run_models_on_benchmark(args.benchmark_path, models, args.windows, 
                                     collections=collections, domains=domains, datasets=datasets,
                                     collect_plot_data=args.plots, history_length=args.history_length,
                                     forecast_horizon=args.forecast_horizon, stride=args.stride, load_cached_counts=args.load_cached_counts)
    
    # Compare performance
    compare_model_performance(results)
    # Export hierarchical CSV results
    export_hierarchical_results_to_csv(results, output_dir=args.output_dir)
    
    # Generate plots if requested
    if args.plots and '_plot_data' in results:
        generate_forecast_plots(results, output_dir=args.output_dir)
    
    # Export CSV if requested
    if args.csv:
        export_results_to_csv(args.benchmark_path, models, max_windows=args.windows, output_dir=args.output_dir,
                             collections=collections, domains=domains, datasets=datasets,
                             history_length=args.history_length, forecast_horizon=args.forecast_horizon, stride=args.stride, load_cached_counts=args.load_cached_counts)
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print("Example completed successfully!")
    
    return 0


if __name__ == "__main__":
    exit(main())