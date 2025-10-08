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
import copy

# Set environment variable to suppress warnings at the system level
os.environ['PYTHONWARNINGS'] = 'ignore'

# Set warnings to ignore at the module level
warnings.simplefilter("ignore")

from musedfm.data import Benchmark
from musedfm.plotting import plot_window_forecasts

# Import utility and debug functions
from examples.utils import (
    parse_models, _aggregate_metrics, _aggregate_results_by_level
)
from examples.debug import (
    _initialize_nan_tracking, _update_nan_tracking, _check_window_nan_values,
    _report_nan_statistics, plot_high_mape_windows, plot_high_mape_univariate_windows,
    debug_model_performance, debug_univariate_performance, debug_forecast_failure,
    debug_forecast_length_mismatch, debug_model_summary
)
from examples.export_csvs import (
    export_hierarchical_results_to_csv, export_results_to_csv
)

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
                               results, collect_plot_data, plot_data, dataset_name, num_plots_to_keep=1):
    """Process a single window with all models."""
    # Check for NaN values in this window
    window_nan_stats = _check_window_nan_values(window)
    
    # Process this window with all models
    for model_name, model in models.items():
        target_length = len(window.target())
        
        # Generate multivariate forecast if model supports it (do this first for proper training)
        multivariate_forecast = None
        if not model["univariate"]:
            with suppress_all_warnings():
                multivariate_forecast = model["model"].forecast(window.history(), window.covariates(), target_length)

            # debugging failed forecasts
            if multivariate_forecast is None:
                debug_forecast_failure(model_name, "multivariate")
                multivariate_forecast = np.zeros(target_length)  # Fallback to zeros
        
        # Generate univariate forecast
        with suppress_all_warnings():
            univariate_forecast = model["model"].forecast(window.history(), None, target_length)

        # debugging failed forecasts
        if univariate_forecast is None:
            debug_forecast_failure(model_name, "univariate")
            univariate_forecast = np.zeros(target_length)  # Fallback to zeros

        # Validate forecast length matches target length
        if len(univariate_forecast) != target_length:
            debug_forecast_length_mismatch(model_name, len(univariate_forecast), target_length)
            raise ValueError(f"Forecast length mismatch: model '{model_name}' returned {len(univariate_forecast)} values, but target has {target_length} values")

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
        
        model_dataset_windows[model_name] += 1
        results[model_name]['windows'] += 1
        
        # Also count windows for univariate results if available
        if not model["univariate"] and univariate_forecast is not None:
            univariate_model_name = f"{model_name}_univariate"
            results[univariate_model_name]['windows'] += 1
    
    return window_nan_stats


def _calculate_dataset_metrics(model_dataset_results, model_name):
    """Calculate average metrics for a model on a dataset."""
    if model_dataset_results[model_name]:
        dataset_avg_metrics = {}
        for metric in model_dataset_results[model_name][0].keys():
            values = [result[metric] for result in model_dataset_results[model_name]]
            # Check if values list is not empty before calling nanmean
            if values:
                dataset_avg_metrics[metric] = np.nanmean(values)
            else:
                dataset_avg_metrics[metric] = np.nan
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
                           num_plots_to_keep: int = 1):
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
    results_base_dict = {
        'metrics': {},
        'dataset_results': [],
        'category_results': {},
        'domain_results': {},
        'time': 0.0,
        'windows': 0
    }
    for model_name in models.keys():
        results[model_name] = copy.deepcopy(results_base_dict)
        
        # Initialize univariate results only for non-univariate models
        if not models[model_name]["univariate"]:
            univariate_model_name = f"{model_name}_univariate"
            results[univariate_model_name] = copy.deepcopy(results_base_dict)        
    
    # Iterate through benchmark structure: category -> domain -> dataset (outer loop)
    dataset_count = 0
    skip_datasets_debug = 0  # DEBUG: Change this to skip the first N datasets for debugging
    last_dataset_debug = -1
    
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
                    
                    # Process this window with all models
                    window_nan_stats = _process_window_with_models(
                        window, models, model_dataset_results, model_dataset_windows, 
                        results, collect_plot_data, plot_data, dataset_name, num_plots_to_keep=num_plots_to_keep
                    )
                    _update_nan_tracking(nan_stats, window_nan_stats)
                
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
                    
                    debug_model_performance(model_name, dataset_avg_metrics, model_dataset_windows, model_elapsed_time)
                    plot_high_mape_windows(model_name, dataset_name, dataset, models[model_name], dataset_avg_metrics)
                
                # Process univariate results for multivariate models only
                for model_name, model in models.items():
                    univariate_model_name, univariate_avg_metrics = _process_univariate_results(
                        model_name, model, model_dataset_results, model_dataset_windows, 
                        model_elapsed_time, dataset_name, results
                    )
                    
                    if univariate_model_name is not None:
                        plot_high_mape_univariate_windows(univariate_model_name, dataset_name, dataset, model, univariate_avg_metrics)
                
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
        debug_model_summary(model_name, results, models)
    
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
    
    # Create individual plots for each window
    print("Creating individual forecast plots...")
    
    # Create plots directory
    plot_data = results['_plot_data']
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(list(plot_data[0].keys()))
    
    current_dataset_window_forecasts = ("", -1, {})
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
            current_dataset_window_forecasts = (dataset_name, window_index, {})
            print(f"Started Processing {dataset_name} - Window {window_index}")
        if forecast is not None:
            current_dataset_window_forecasts[2][model_name] = forecast
        if univariate_forecast is not None:
            current_dataset_window_forecasts[2][f"{model_name}_univariate"] = univariate_forecast
        if to_plot:
            # Create title
            forecasts = current_dataset_window_forecasts[2]
            title = f"{current_dataset_window_forecasts[0]} - Window {current_dataset_window_forecasts[1]}\nModels: {', '.join(forecasts.keys())}"
            
            # Plot the window
            plot_window_forecasts(
                window=window,
                forecasts=forecasts,
                title=title,
                figsize=(12, 6),
                save_path=os.path.join(plots_dir, f"forecast_plot_{i+1}.png")
            )
            
            print(f"Saved plot {i+1}: {dataset_name} - Window {window_index} to {os.path.join(plots_dir, f'forecast_plot_{i+1}.png')}")
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
  python run_musedfm.py --models all --categories Traditional --domains Energy
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
    print(f"categories: {args.categories or 'All'}")
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
    categories = args.categories.split(',') if args.categories else None
    domains = args.domains.split(',') if args.domains else None
    datasets = args.datasets.split(',') if args.datasets else None
    
    start_time = time.time()
    
    # Run models on benchmark
    results = run_models_on_benchmark(args.benchmark_path, models, args.windows, 
                                     categories=categories, domains=domains, datasets=datasets,
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
                             categories=categories, domains=domains, datasets=datasets,
                             history_length=args.history_length, forecast_horizon=args.forecast_horizon, stride=args.stride, load_cached_counts=args.load_cached_counts)
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print("Example completed successfully!")
    
    return 0


if __name__ == "__main__":
    exit(main())