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
import glob
import warnings
import numpy as np
import argparse
import copy
import pandas as pd
from musedfm.data import Benchmark
from musedfm.plotting import plot_window_forecasts

# Import utility and debug functions
from examples.utils import (
    _aggregate_metrics, _aggregate_results_by_level, timeout_handler
)
from examples.model_handling import (
    parse_models
)
from examples.debug import (
    _initialize_nan_tracking, _update_nan_tracking, _check_window_nan_values,
    _report_nan_statistics, plot_high_mape_windows,
    debug_model_performance, debug_univariate_performance, debug_forecast_failure,
    debug_forecast_length_mismatch, debug_model_summary
)
from examples.export_csvs import (
    export_hierarchical_results_to_csv
)
from examples.save_submission import save_submission
# Additional aggressive suppression
import contextlib

@contextlib.contextmanager
def suppress_all_warnings():
    """Context manager to suppress all warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield

class SaveManager:
    def __init__(self, save_path, category_names, model_name, stride=256, history_length=512, forecast_horizon=128, chunk_size=65536):
        self.save_path = save_path
        self.category_names = category_names
        self.model_name = model_name
        self.stride = stride
        self.history_length = history_length
        self.forecast_horizon = forecast_horizon
        self.chunk_size = chunk_size
        self.forecast_block = []
        self.forecast_block_idx = 0
        self.last_block_save_path = None
    
    def save_forecasts_interval(self, forecast, category, domain, dataset):
        self.forecast_block.append(forecast)
        self.last_block_save_path = f"{self.save_path}/forecasts/{self.model_name}/{category}/{domain}/{dataset}/s{self.stride}_w{self.history_length}_f{self.forecast_horizon}_b{self.forecast_block_idx}.parquet"
        if len(self.forecast_block) * self.forecast_horizon > self.chunk_size:
            # take the product of all the blocks in self.forecast_block
            total_size = sum(np.prod(block.shape) for block in self.forecast_block)

            self.run_saving()
            self.forecast_block = []
            self.forecast_block_idx += 1

    def reset_saving(self):
        self.forecast_block = []
        self.forecast_block_idx = 0
        self.last_block_save_path = None
    
    def flush_saving(self):
        self.run_saving(target_save_path=self.last_block_save_path)
        self.reset_saving()
    
    def run_saving(self, target_save_path=None):
        if target_save_path is None:
            target_save_path = self.last_block_save_path
        # create an index for each forecast window
        window_idx = list()
        for i, forecast in enumerate(self.forecast_block):
            window_idx.extend([i] * len(forecast))
        window_idx = np.array(window_idx)
        forecast_block_df = pd.DataFrame(np.concatenate(self.forecast_block, axis=0))
        forecast_block_df['window_idx'] = window_idx
        os.makedirs(os.path.dirname(target_save_path), exist_ok=True)
        forecast_block_df.to_parquet(target_save_path)
    
    def load_forecasts(self, save_path, model_name):
        '''returns an iterator that yields all of the forecasts for a given model'''
        def forecast_generator():
            for category in self.category_names:
                for domain in self.category_names[category]:
                    for dataset in self.category_names[category][domain]:
                        # find all files in the folder and  sort by forecast_block_idx
                        files = sorted(glob.glob(f"{self.save_path}/forecasts/{self.model_name}/{category}/{domain}/{dataset}/s{self.stride}_w{self.history_length}_f{self.forecast_horizon}_b*.parquet"), key=lambda x: int(x.split('_')[-1].split('.')[0][1:]))
                        for file in files:
                            forecast_block_df = pd.read_parquet(file)

                            # for each window in the forecast, where window is defined by the forecast_block_idx, yield a forecast dict
                            unique_window_idx = forecast_block_df['window_idx'].unique()

                            for i, window in enumerate(unique_window_idx):
                                window_forecast = forecast_block_df[forecast_block_df['window_idx'] == window]
                                output_dict = {
                                    'category': category,
                                    'domain': domain,
                                    'dataset': dataset,
                                    'window_idx': window,
                                    'forecast': window_forecast.values
                                }
                                yield output_dict
        return forecast_generator()

def _forecast_window_with_models(window, models, model_dataset_results, model_dataset_windows, 
                               results, save_managers_multivariate, save_managers_univariate, category, domain, dataset):
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
            if multivariate_forecast is None:
                multivariate_forecast = np.zeros(target_length)  # Fallback to zeros
        
        # Generate univariate forecast
        with suppress_all_warnings():
            univariate_forecast = model["model"].forecast(window.history(), None, target_length)
        if univariate_forecast is None:
            univariate_forecast = np.zeros(target_length)  # Fallback to zeros

        # Get evaluation results for multivariate forecast if submitted
        if multivariate_forecast is not None:
            save_managers_multivariate[model_name].save_forecasts_interval(multivariate_forecast, category.category, domain.domain_name, dataset.dataset_name)
        
        # Store univariate results
        if univariate_forecast is not None:
            # For univariate models, store in main results; for multivariate models, store in _univariate
            save_managers_univariate[model_name].save_forecasts_interval(univariate_forecast, category.category, domain.domain_name, dataset.dataset_name)

        model_dataset_windows[model_name] += 1
        results[model_name]['windows'] += 1
        
        # Also count windows for univariate results if available
        if not model["univariate"] and univariate_forecast is not None:
            univariate_model_name = f"{model_name}_univariate"
            results[univariate_model_name]['windows'] += 1
    
    return window_nan_stats

def forecast_models_on_benchmark(benchmark_path: str, models: dict, max_windows: int = 100, 
                           categories: str = None, domains: str = None, datasets: str = None,
                           collect_plot_data: bool = False, history_length: int = 512, 
                           forecast_horizon: int = 128, stride: int = 256, load_cached_counts: bool = False,
                           num_plots_to_keep: int = 1, debug_mode: bool = False, chunk_size: int = 65536, forecast_save_path: str = "./"):
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
            save_manager_multivariate = SaveManager(forecast_save_path, new_category_names, model_name, stride, history_length, forecast_horizon, chunk_size)
            model_save_managers_multivariate[model_name] = save_manager_multivariate
            save_manager_univariate = SaveManager(forecast_save_path, new_category_names, univariate_model_name, stride, history_length, forecast_horizon, chunk_size)
            model_save_managers_univariate[model_name] = save_manager_univariate
        else:
            save_manager_univariate = SaveManager(forecast_save_path, new_category_names, model_name, stride, history_length, forecast_horizon, chunk_size)
            model_save_managers_univariate[model_name] = save_manager_univariate
            model_save_managers_multivariate[model_name] = None
    
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
                    window_nan_stats = _forecast_window_with_models(
                        window, models, model_dataset_results, model_dataset_windows, 
                        results, model_save_managers_multivariate, model_save_managers_univariate, category, domain, dataset
                    )
                for model_name, save_manager in model_save_managers_multivariate.items():
                    if save_manager is not None:
                        save_manager.flush_saving()
                for model_name, save_manager in model_save_managers_univariate.items():
                    if save_manager is not None:
                        save_manager.flush_saving()
    return results

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
        "--forecast-save-path",
        type=str,
        default="/tmp",
        help="Path to save the forecasts (default: /tmp)"
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
    results = forecast_models_on_benchmark(args.benchmark_path, models, args.windows, 
                                     categories=categories, domains=domains, datasets=datasets,
                                     history_length=args.history_length,
                                     forecast_horizon=args.forecast_horizon, stride=args.stride, load_cached_counts=args.load_cached_counts,
                                     chunk_size=args.chunk_size, forecast_save_path=args.forecast_save_path)

    # Save submission files
    submission_dir = os.path.join(args.output_dir, "submissions")
    save_submission(results, submission_dir)
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print("Example completed successfully!")
    
    return 0


if __name__ == "__main__":
    exit(main())