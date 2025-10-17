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
import os
import warnings
import numpy as np
import argparse
from tqdm import tqdm
import copy

# Set environment variable to suppress warnings at the system level
os.environ['PYTHONWARNINGS'] = 'ignore'

# Set warnings to ignore at the module level
warnings.simplefilter("ignore")

from musedfm.data import Benchmark
from musedfm.plotting import plot_window_forecasts

# Import utility and debug functions
from examples.utils import (
    _aggregate_metrics, _aggregate_results_by_level
)
from examples.model_handling import (
    parse_models
)
from examples.debug import (
    _initialize_nan_tracking, _update_nan_tracking, _check_window_nan_values,
    _report_nan_statistics, plot_high_mape_windows,
    debug_model_performance, debug_univariate_performance,
    debug_model_summary,
    debug_low_variance_window, debug_low_variance_summary,
    plot_low_variance_dataset_summary, print_final_low_variance_statistics,
    create_individual_window
)
from examples.export_csvs import (
    export_hierarchical_results_to_csv
)
from examples.eval_musedfm import (
    SaveManager
)
from examples.save_submission import save_submission

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

# Suppress maximum likelihood convergence warnings - multiple approaches
warnings.filterwarnings("ignore", 
                       message="Maximum Likelihood optimization failed to converge. Check mle_retvals",
                       category=Warning)
# Suppress by module and filename
warnings.filterwarnings("ignore", 
                       module="statsmodels.base.model")

warnings.filterwarnings("ignore", 
                       module="statsmodels")

# Suppress by message patterns
warnings.filterwarnings("ignore", 
                       message=".*convergence.*",
                       category=Warning)

warnings.filterwarnings("ignore", 
                       message=".*optimization.*",
                       category=Warning)

warnings.filterwarnings("ignore", 
                       message=".*failed to converge.*",
                       category=Warning)

warnings.filterwarnings("ignore", 
                       message=".*mle_retvals.*",
                       category=Warning)

# Suppress all warnings from statsmodels
warnings.filterwarnings("ignore", 
                       module="statsmodels")

# Nuclear option - suppress all warnings
warnings.filterwarnings("ignore")

# Additional aggressive suppression
import contextlib

@contextlib.contextmanager
def suppress_all_warnings():
    """Context manager to suppress all warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _process_window_with_models(window, models, model_dataset_results, model_dataset_windows, 
                               results, collect_plot_data, plot_data, dataset_name, num_plots_to_keep=1,
                               save_managers_multivariate=None, save_managers_univariate=None, category="", domain=""):
    """Process a single window with all models."""
    # Check for NaN values in this window
    window_nan_stats = _check_window_nan_values(window)
    
    # Process this window with all models
    for model_name, model in models.items():
        
        target_length = window.target().shape[1]  # Get forecast horizon from batched target
        
        # Generate multivariate forecast if model supports it (do this first for proper training)
        multivariate_forecast = None
        if not model["univariate"]:
            with suppress_all_warnings():
                multivariate_forecast = model["model"].forecast(window.history(), window.covariates(), target_length, window.timestamps())

        # debugging failed forecasts
        if save_managers_multivariate[model_name] is not None:
            save_managers_multivariate[model_name].save_forecasts_interval(multivariate_forecast, category.category, domain.domain_name, dataset_name)
        
        # Generate univariate forecast
        with suppress_all_warnings():
            univariate_forecast = model["model"].forecast(window.history(), None, target_length, window.timestamps())

        # debugging failed forecasts
        if save_managers_univariate[model_name] is not None:
            save_managers_univariate[model_name].save_forecasts_interval(univariate_forecast, category.category, domain.domain_name, dataset_name)

        # Validate forecast length matches target length
        if univariate_forecast.shape[1] != target_length:
            raise ValueError(f"Forecast length mismatch: model '{model_name}' returned {univariate_forecast.shape[1]} values, but target has {target_length} values")

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
        if collect_plot_data and model_dataset_windows[model_name] < num_plots_to_keep:  # Only collect first 3 windows for plotting
            plot_data.append({
                'window': window,
                'forecast': multivariate_forecast,
                'univariate_forecast': univariate_forecast,
                'model_name': model_name,
                'window_index': results[model_name]['windows'],
                'dataset_name': dataset_name
            })
        
        model_dataset_windows[model_name] += window.batch_size
        results[model_name]['windows'] += window.batch_size
        
        # Also count windows for univariate results if available
        if not model["univariate"] and univariate_forecast is not None:
            univariate_model_name = f"{model_name}_univariate"
            results[univariate_model_name]['windows'] += window.batch_size
        
        # # Restore stdout if it was disabled
        # if model_name != "chronos":
        #     sys.stdout.close()
        #     sys.stdout = sys.__stdout__
    
    return window_nan_stats


def _calculate_dataset_metrics(model_dataset_results, model_name):
    """Calculate average metrics for a model on a dataset."""
    if model_dataset_results[model_name]:
        dataset_avg_metrics = {}
        
        # Get metric names from the first result
        metric_names = list(model_dataset_results[model_name][0].keys())
        
        # Collect all metric vectors from all batches
        all_metric_values = {metric: [] for metric in metric_names}
        
        for result in model_dataset_results[model_name]:
            # Each result now contains metric vectors instead of scalars
            for metric in metric_names:
                all_metric_values[metric].extend(result[metric])
        
        # Calculate single nanmean across all individual window metrics
        for metric in metric_names:
            dataset_avg_metrics[metric] = np.nanmean(all_metric_values[metric]) if all_metric_values[metric] else np.nan
    else:
        # No valid results - set all metrics to NaN
        dataset_avg_metrics = {'MAPE': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'NMAE': np.nan}
    
    return dataset_avg_metrics


def _process_univariate_results(model_name, model, model_dataset_results, model_dataset_windows, 
                               model_elapsed_time, dataset_name, results):
    """Process univariate results for multivariate models."""
    if not model["univariate"] and f"{model_name}_univariate" in model_dataset_results:
        univariate_model_name = f"{model_name}_univariate"
        
        # Calculate average metrics for univariate version
        univariate_avg_metrics = _calculate_dataset_metrics(model_dataset_results, univariate_model_name)
        
        # Store univariate dataset results
        results[univariate_model_name]['dataset_results'].append({
            'dataset_name': dataset_name,
            'metrics': univariate_avg_metrics,
            'window_count': model_dataset_windows[model_name]  # Same window count as main model
        })
        
        # Use same elapsed time as main model
        results[univariate_model_name]['time'] += model_elapsed_time
        
        debug_univariate_performance(univariate_model_name, univariate_avg_metrics, model_dataset_windows, model_elapsed_time, model_name)
        return univariate_model_name, univariate_avg_metrics
    
    return None, None


def run_models_on_benchmark(benchmark_path: str, models: dict, max_windows: int = 100, 
                           categories: str = None, domains: str = None, datasets: str = None,
                           collect_plot_data: bool = False, history_length: int = 512, 
                           forecast_horizon: int = 128, stride: int = 256, load_cached_counts: bool = False,
                           num_plots_to_keep: int = 1, debug_mode: bool = False, chunk_size: int = 1048576, forecast_save_path: str = "", output_dir: str = "", batch_size: int = 1):
    """Run multiple forecasting models on a benchmark and compare their performance."""
    print("=" * 60)
    print("Running Multiple Models on Benchmark")
    print("=" * 60)
    
    benchmark = Benchmark(benchmark_path, history_length=history_length, forecast_horizon=forecast_horizon, stride=stride, load_cached_counts=load_cached_counts, batch_size=batch_size)
    print(f"Loaded benchmark with {len(benchmark)} categories")
    print(f"Running {len(models)} models: {list(models.keys())}")
    
    results = {}
    plot_data = []  # Store windows and forecasts for plotting
    
    # Initialize global statistics collection
    all_dataset_low_variance_stats = {}
    total_windows_processed = 0
    total_files_processed = 0
    dataset_seen_files = {}  # Track unique files seen during processing
    low_variance_windows = {}
    
    # Initialize results for each model
    results_base_dict = {
        'metrics': {},
        'dataset_results': [],
        'category_results': {},
        'domain_results': {},
        'time': 0.0,
        'windows': 0
    }

    # create category names without ALL_DATASETS
    new_category_names = copy.deepcopy(benchmark.category_names)
    for category in new_category_names:
        del new_category_names[category]["ALL_DATASETS"]
    model_save_managers_multivariate = {}
    model_save_managers_univariate = {}
    for model_name in models.keys():
        results[model_name] = copy.deepcopy(results_base_dict)
        
        # Initialize univariate results only for non-univariate models
        if not models[model_name]["univariate"]:
            univariate_model_name = f"{model_name}_univariate"
            results[univariate_model_name] = copy.deepcopy(results_base_dict)
            # if forecast_save_path is not empty, save forecasts to the forecast_save_path through save managers
            if forecast_save_path != "":
                save_manager_multivariate = SaveManager(forecast_save_path, new_category_names, model_name, stride, history_length, forecast_horizon, chunk_size)
                save_manager_univariate = SaveManager(forecast_save_path, new_category_names, model_name, stride, history_length, forecast_horizon, chunk_size)
                model_save_managers_univariate[model_name] = save_manager_univariate
                model_save_managers_multivariate[model_name] = save_manager_multivariate
            else:
                save_manager_multivariate = None
                save_manager_univariate = None
                model_save_managers_univariate[model_name] = None
                model_save_managers_multivariate[model_name] = None
        else:
            if forecast_save_path != "":
                model_save_managers_univariate[model_name] = SaveManager(forecast_save_path, new_category_names, model_name, stride, history_length, forecast_horizon, chunk_size)
            else:
                model_save_managers_univariate[model_name] = None
            model_save_managers_multivariate[model_name] = None
    
    # Iterate through benchmark structure: category -> domain -> dataset (outer loop)
    dataset_count = 0
    skip_datasets_debug = 0  # DEBUG: Change this to skip the first N datasets for debugging
    last_dataset_debug = -1
    
    # Count total datasets for progress bar
    total_datasets = 0
    for category in benchmark:
        if categories is not None and category.category_path.name not in categories:
            continue
        for domain in category:
            if domains is not None and domain.domain_path.name not in domains:
                continue
            for dataset in domain:
                if datasets is not None and dataset.data_path.name not in datasets:
                    continue
                total_datasets += 1
    
    # Main processing loop with progress bar
    dataset_progress = tqdm(total=total_datasets, desc="Processing datasets", unit="dataset")
    
    for category in benchmark:
        # Apply filters if specified
        if categories is not None and category.category_path.name not in categories:
            continue
        for domain in category:
            if domains is not None and domain.domain_path.name not in domains:
                continue
            for dataset in domain:
                if datasets is not None and dataset.data_path.name not in datasets:
                    continue
                
                # Skip first few datasets for debugging
                if dataset_count < skip_datasets_debug or (last_dataset_debug > 0 and dataset_count >= last_dataset_debug):
                    dataset_count += 1
                    dataset_progress.set_postfix_str(f"Skipping: {dataset.data_path.name}")
                    dataset_progress.update(1)
                    continue
                
                # Get full dataset name from benchmark path
                dataset_name = str(dataset.dataset_name)
                dataset_seen_files[dataset_name] = set()
                dataset_progress.set_postfix_str(f"Processing: {dataset_name}")
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
                
                # Initialize low variance tracking for debugging
                low_variance_windows[dataset_name] = list()
                
                # Initialize NaN tracking for this dataset
                nan_stats = _initialize_nan_tracking()
                
                # Determine number of windows to process
                num_windows_estimate = min(len(dataset), max_windows) if max_windows is not None else len(dataset)
                num_iters_estimate = int(np.ceil(num_windows_estimate / batch_size))
                window_progress = tqdm(total=num_iters_estimate, desc=f"Windows in {dataset_name}", unit="iters", leave=False)
                num_windows = 0
                last_parquet_index = 0
                
                for i, window in enumerate(dataset):
                    # Check max_windows per dataset, not overall
                    if num_iters_estimate is not None and i >= num_iters_estimate:
                        break
                    
                    window_progress.set_postfix_str(f"Window {i+1}/{len(dataset)}")
                    
                    # Track files as we see them
                    added_files = list()
                    if hasattr(dataset, '_parquet_files') and hasattr(dataset, '_current_parquet_index'):
                        for j in range(last_parquet_index, min(dataset._current_parquet_index, len(dataset._parquet_files))):
                            current_file = str(dataset._parquet_files[j])
                            added_files.append(current_file)
                            dataset_seen_files[dataset_name].add(current_file)
                        last_parquet_index = min(dataset._current_parquet_index, len(dataset._parquet_files))
                        # last_parquet_index = dataset._current_parquet_index
                        # current_file = str(dataset._parquet_files[dataset._current_parquet_index])
                        # dataset_seen_files[dataset_name].add(current_file)
                    else:
                        raise ValueError(f"Dataset {dataset_name} has no parquet files")
                    
                    # Debug low variance targets if in debug mode
                    if debug_mode:
                        target = window.target()  # Get target once for all models
                        batch_size = target.shape[0]
                        
                        # Check each individual window in the batch separately
                        for batch_idx in range(batch_size):
                            # Create individual window for this batch item
                            individual_window = create_individual_window(window, batch_idx)
                            
                            # Call debug function for this individual window
                            is_low_variance = debug_low_variance_window(
                                individual_window, model_name, dataset_name, f"{i}_{batch_idx}", 
                                save_path=output_dir, dataset=dataset
                            )
                            
                            if is_low_variance:
                                individual_target = target[batch_idx]
                                low_variance_windows[dataset_name].append({
                                    'dataset_name': dataset_name,
                                    'window_idx': i * batch_size + batch_idx,
                                    'batch_idx': batch_idx,
                                    'iter_idx': i,
                                    'file_path': current_file,
                                    'model_name': model_name,
                                    'target_std': np.std(individual_target),
                                    'target_range': np.max(individual_target) - np.min(individual_target),
                                    'target_mean': np.mean(individual_target),
                                    'is_constant': (np.max(individual_target) - np.min(individual_target)) < 1e-6,
                                    'is_near_zero': abs(np.mean(individual_target)) < 1e-3,
                                    'file_name': os.path.basename(current_file), # when batching, this is just the best guess
                                })
                    
                    # Process this window with all models
                    window_nan_stats = _process_window_with_models(
                        window, models, model_dataset_results, model_dataset_windows, 
                        results, collect_plot_data, plot_data, dataset_name, num_plots_to_keep=num_plots_to_keep,
                        save_managers_multivariate=model_save_managers_multivariate, save_managers_univariate=model_save_managers_univariate,
                        category=category, domain=domain
                    )
                    if debug_mode:
                        _update_nan_tracking(nan_stats, window_nan_stats)
                    
                    window_progress.update(1)
                    num_windows += 1
                
                for model_name, save_manager in model_save_managers_multivariate.items():
                    if save_manager is not None:
                        save_manager.flush_saving()
                for model_name, save_manager in model_save_managers_univariate.items():
                    if save_manager is not None:
                        save_manager.flush_saving()
                
                # Calculate average metrics and store results for each model
                for model_name in models.keys():
                    # Calculate average metrics for this model on this dataset
                    dataset_avg_metrics = _calculate_dataset_metrics(model_dataset_results, model_name)
                    
                    # Store dataset results for this model
                    results[model_name]['dataset_results'].append({
                        'dataset_name': dataset_name,
                        'metrics': dataset_avg_metrics,
                        'window_count': model_dataset_windows[model_name]
                    })
                    
                    model_elapsed_time = time.time() - model_start_times[model_name]
                    results[model_name]['time'] += model_elapsed_time
                    
                    if debug_mode:
                        debug_model_performance(model_name, dataset_avg_metrics, model_dataset_windows, model_elapsed_time)
                        plot_high_mape_windows(model_name, dataset_name, dataset, models[model_name], dataset_avg_metrics, save_path=output_dir)
                
                # Process univariate results for multivariate models only
                for model_name, model in models.items():
                    univariate_model_name, univariate_avg_metrics = _process_univariate_results(
                        model_name, model, model_dataset_results, model_dataset_windows, 
                        model_elapsed_time, dataset_name, results
                    )
                    
                    if univariate_model_name is not None and debug_mode:
                        plot_high_mape_windows(univariate_model_name, dataset_name, dataset, model, univariate_avg_metrics, forecast_type="univariate", save_path=output_dir)
                
                # Report NaN statistics for this dataset
                if debug_mode:
                    _report_nan_statistics(nan_stats)
                    # Convert dataset-keyed structure to model-keyed structure for summary
                    model_low_variance_windows = {}
                    for dataset_name, windows in low_variance_windows.items():
                        for window_info in windows:
                            model_name = window_info.get('model_name', 'unknown')
                            if model_name not in model_low_variance_windows:
                                model_low_variance_windows[model_name] = []
                            model_low_variance_windows[model_name].append(window_info)
                    debug_low_variance_summary(model_low_variance_windows)
                    
                    # Create dataset summary plots only once per dataset (not per model)
                    all_low_variance_windows = []
                    for windows in low_variance_windows[dataset_name]:
                        all_low_variance_windows.append(windows)
                                        
                    if all_low_variance_windows:
                        # Generate summary plot only once for the dataset
                        print(f"  📊 Generating dataset summary plot...")
                        plot_low_variance_dataset_summary(dataset_name, all_low_variance_windows, output_dir)
                        
                        # Simple summary instead of complex plotting
                        unique_files = set(w['file_path'] for w in all_low_variance_windows)
                        low_variance_dir = os.path.join(output_dir, "low_variance", dataset_name)
                        print(f"  📊 Found {len(all_low_variance_windows)} low variance windows across {len(unique_files)} files")
                        print(f"  📊 Files: {', '.join(os.path.basename(f) for f in unique_files)}")
                        print(f"  📊 Low variance plots saved to: {low_variance_dir}")
                        print(f"  📊 Low variance analysis completed for {dataset_name}")
                        
                        # Collect statistics for final summary
                        unique_files = set(w['file_path'] for w in all_low_variance_windows)
                        constant_windows = sum(1 for w in all_low_variance_windows if w.get('is_constant', False))
                        near_zero_windows = sum(1 for w in all_low_variance_windows if w.get('is_near_zero', False))
                        
                        all_dataset_low_variance_stats[dataset_name] = {
                            'total_low_variance_windows': len(all_low_variance_windows),
                            'total_low_variance_files': len(unique_files),
                            'constant_low_variance_windows': constant_windows,
                            'near_zero_low_variance_windows': near_zero_windows,
                            'low_variance_files': sorted([os.path.basename(f) for f in unique_files]),
                            'total_windows': num_windows,
                            'total_files': len(dataset_seen_files[dataset_name])
                        }
                
                # Update global totals
                total_windows_processed += num_windows
                total_files_processed += len(dataset_seen_files[dataset_name])  # Count unique files seen so far
                
                # Close window progress bar and update dataset progress
                window_progress.close()
                dataset_progress.update(1)
                print(f"  Completed dataset {dataset_name} with {model_dataset_windows[model_name]} windows")
        
        # Aggregate domain-level metrics for this domain
        _aggregate_results_by_level(results, models, benchmark, domain.domain_name, 'domain')
    
    # Close dataset progress bar
    dataset_progress.close()
    
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
        if debug_mode:
            debug_model_summary(model_name, results, models)
    
    if collect_plot_data:
        results['_plot_data'] = plot_data
    
    # Print final low variance statistics summary
    if debug_mode and all_dataset_low_variance_stats:
        print_final_low_variance_statistics(all_dataset_low_variance_stats, total_windows_processed, total_files_processed, dataset_seen_files)
    
    return results


def generate_forecast_plots(results: dict, output_dir: str = "/tmp", limit_windows: int = 10):
    """Generate forecast metrics CSV using pre-computed data from run_models_on_benchmark."""
    print("\n" + "=" * 60)
    print("Generating Forecast Metrics CSV")
    print("=" * 60)
    
    if not results or '_plot_data' not in results:
        print("Warning: No results data available")
        return False
    
    # Create individual plots for each window
    print("Creating individual forecast plots...")
    
    # Create plots directory
    plot_data = results['_plot_data']
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(list(plot_data[0].keys()))
    
    current_dataset_window_forecasts = ("", -1, {})
    last_window = None
    window_count = 0
    for i, data in enumerate(plot_data):  # Plot first 6 windows
        window = data['window']
        forecast = data['forecast']
        univariate_forecast = data['univariate_forecast']
        dataset_name = data['dataset_name']
        window_index = data['window_index']
        model_name = data['model_name']
        to_plot = False
        if current_dataset_window_forecasts[0] != dataset_name or current_dataset_window_forecasts[1] != window_index:
            to_plot = current_dataset_window_forecasts[0] != ""
            last_dataset_window_forecasts = copy.deepcopy(current_dataset_window_forecasts)
            current_dataset_window_forecasts = (dataset_name, window_index, {})
            print(f"Started Processing {dataset_name} - Window {window_count}")
            window_count += 1
        if forecast is not None:
            current_dataset_window_forecasts[2][model_name] = forecast
        if univariate_forecast is not None:
            current_dataset_window_forecasts[2][f"{model_name}_univariate"] = univariate_forecast
        if to_plot:
            # Create title
            forecasts = last_dataset_window_forecasts[2]
            title = f"{last_dataset_window_forecasts[0]} - Window {last_dataset_window_forecasts[1]}\n"
            
            # Plot the window
            plt = plot_window_forecasts(
                window=last_window,
                forecasts=forecasts,
                title=title,
                figsize=(12, 6),
                save_path=os.path.join(plots_dir, f"baseline_forecast_{window_count-1}.png"),
                show_plot=True
            )
            
            print(f"Saved plot {i+1}: {dataset_name} - Window {window_count-1} to {os.path.join(plots_dir, f'baseline_forecast_{window_count-1}.png')}")
        if limit_windows > 0 and window_count > limit_windows:
            break
        last_window = window
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
  python run_musedfm.py --models linear_trend,exponential_smoothing --windows 20 --batch-size 4
  python run_musedfm.py --models all --categories Traditional --domains Energy --batch-size 8
  python run_musedfm.py --models mean --datasets al_daily,bitcoin_price --plots --batch-size 2
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
        help="Path to the benchmark directory containing categories"
    )
    
    parser.add_argument(
        "--categories",
        type=str,
        help="Comma-separated list of categories to filter by (e.g., 'Traditional,Synthetic')"
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
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of windows to collect in each batch (default: 1)"
    )

    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Enable debug mode"
    )

    parser.add_argument(
        "--forecast-save-path",
        type=str,
        default="",
        help="Path to save the forecasts (default: empty, no saving)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=131072,
        help="Chunk size for saving forecasts (default: 1048576)"
    )
    
    args = parser.parse_args()
    
    print("MUSED-FM Example: Multiple Model Forecasting")
    print("=" * 60)
    print(f"Models: {args.models}")
    print(f"Benchmark path: {args.benchmark_path}")
    print(f"categories: {args.categories or 'All'}")
    print(f"Domains: {args.domains or 'All'}")
    print(f"Datasets: {args.datasets or 'All'}")
    print(f"Max windows per dataset: {args.windows}")
    print(f"Batch size: {args.batch_size}")
    print(f"Generate plots: {args.plots}")
    print(f"Output directory: {args.output_dir}")
    
    # Parse models
    models = parse_models(args.models)
    if not models:
        print("Error: No valid models specified")
        return 1
    
    # Parse filter arguments
    categories = args.categories.split(',') if args.categories else None
    domains = args.domains.split(',') if args.domains else None
    datasets = args.datasets.split(',') if args.datasets else None
    
    start_time = time.time()
    
    # Run models on benchmark
    results = run_models_on_benchmark(args.benchmark_path, models, args.windows, 
                                     categories=categories, domains=domains, datasets=datasets,
                                     collect_plot_data=args.plots, history_length=args.history_length,
                                     forecast_horizon=args.forecast_horizon, stride=args.stride, load_cached_counts=args.load_cached_counts,
                                     debug_mode=args.debug_mode, 
                                     chunk_size=args.chunk_size, forecast_save_path=args.forecast_save_path, output_dir=args.output_dir, batch_size=args.batch_size)
    
    # Compare performance
    compare_model_performance(results)
    # Export hierarchical CSV results
    export_hierarchical_results_to_csv(results, output_dir=args.output_dir)
    
    # Generate plots if requested
    if args.plots and '_plot_data' in results:
        generate_forecast_plots(results, output_dir=args.output_dir, limit_windows = -1)
    
    # Save submission files
    submission_dir = os.path.join(args.output_dir, "submissions")
    save_submission(results, submission_dir)
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print("Example completed successfully!")
    
    return 0


if __name__ == "__main__":
    exit(main())