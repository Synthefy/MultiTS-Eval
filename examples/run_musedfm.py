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
import numpy as np
import argparse
from pathlib import Path
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


def run_models_on_benchmark(benchmark_path: str, models: dict, max_windows: int = 100, 
                           collections: str = None, domains: str = None, datasets: str = None,
                           collect_plot_data: bool = False):
    """Run multiple forecasting models on a benchmark and compare their performance."""
    print("=" * 60)
    print("Running Multiple Models on Benchmark")
    print("=" * 60)
    
    benchmark = Benchmark(benchmark_path, history_length=512, forecast_horizon=128, stride=256)
    print(f"Loaded benchmark with {len(benchmark)} categories")
    print(f"Running {len(models)} models: {list(models.keys())}")
    
    results = {}
    plot_data = []  # Store windows and forecasts for plotting
    
    # Initialize results for each model
    for model_name in models.keys():
        results[model_name] = {
            'metrics': {},
            'dataset_results': [],
            'time': 0.0,
            'windows': 0
        }
        
        # Initialize univariate results for all models
        univariate_model_name = f"{model_name}_univariate"
        results[univariate_model_name] = {
            'metrics': {},
            'dataset_results': [],
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
                # Add univariate results tracking for all models
                for model_name, model in models.items():
                    model_dataset_results[f"{model_name}_univariate"] = []
                model_start_times = {model_name: time.time() for model_name in models.keys()}
                
                for i, window in enumerate(dataset):
                    # Check max_windows per dataset, not overall
                    if max_windows is not None and i >= max_windows:
                        break
                    
                    # Process this window with all models
                    for model_name, model in models.items():
                        # Generate univariate forecast (all models)
                        target_length = len(window.target())
                        univariate_forecast = model["model"].forecast(window.history(), None, target_length)
                        
                        # Validate forecast length matches target length
                        if len(univariate_forecast) != target_length:
                            raise ValueError(f"Forecast length mismatch: model '{model_name}' returned {len(univariate_forecast)} values, but target has {target_length} values")
                        
                        # Generate multivariate forecast if model supports it
                        multivariate_forecast = None
                        if not model["univariate"]:
                            multivariate_forecast = model["model"].forecast(window.history(), window.covariates(), target_length)
                        
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
                
                print(f"  Completed dataset {dataset_name}")
        
    # Calculate overall average metrics across all datasets for each model
    all_model_names = list(models.keys())
    # Add univariate model names for multivariate models only
    for model_name, model in models.items():
        if not model["univariate"]:
            all_model_names.append(f"{model_name}_univariate")
    
    for model_name in all_model_names:
        if results[model_name]['dataset_results']:
            overall_avg_metrics = {}
            for metric in ['MAPE', 'MAE', 'RMSE', 'NMAE']:
                values = [result['metrics'][metric] for result in results[model_name]['dataset_results'] if metric in result['metrics']]
                if values:
                    overall_avg_metrics[metric] = np.mean(values)
                else:
                    overall_avg_metrics[metric] = np.nan
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




def export_results_to_csv(benchmark_path: str, models: dict, max_windows: int = None, output_dir: str = "/tmp",
                         collections: str = None, domains: str = None, datasets: str = None):
    """Export forecast results to CSV files."""
    print("\n" + "=" * 60)
    print("Exporting Results to CSV")
    print("=" * 60)
    
    benchmark = Benchmark(benchmark_path, history_length=10, forecast_horizon=5, stride=1)
    
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
                                     collect_plot_data=args.plots)
    
    # Compare performance
    compare_model_performance(results)
    
    # Generate plots if requested
    if args.plots and '_plot_data' in results:
        generate_forecast_plots(results, output_dir=args.output_dir)
    
    # Export CSV if requested
    if args.csv:
        export_results_to_csv(args.benchmark_path, models, max_windows=args.windows, output_dir=args.output_dir,
                             collections=collections, domains=domains, datasets=datasets)
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print("Example completed successfully!")
    
    return 0


if __name__ == "__main__":
    exit(main())
