"""
Compute metrics over MUSEval datasets using SaveManagers.

This module provides functionality to compute forecasting metrics
over datasets using saved forecasts from SaveManagers.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Iterator, Tuple
from pathlib import Path

from museval.data import Benchmark
from museval.metrics import evaluate_metrics
from museval.examples.eval_museval import SaveManager
from museval.examples.utils import _aggregate_metrics, _aggregate_results_by_level
from museval.examples.export_csvs import export_hierarchical_results_to_csv


def compute_metrics_from_saved_forecasts(
    forecast_save_path: str,
    benchmark_path: str,
    model_name: str,
    stride: int = 256,
    history_length: int = 512,
    forecast_horizon: int = 128,
    categories: Optional[List[str]] = None,
    domains: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for a model using saved forecasts from SaveManager.
    
    Args:
        forecast_save_path: Path where forecasts were saved
        benchmark_path: Path to the benchmark directory
        model_name: Name of the model to evaluate
        stride: Stride used for window generation
        history_length: History length used for windows
        forecast_horizon: Forecast horizon used
        categories: List of categories to filter by (None for all)
        domains: List of domains to filter by (None for all)
        datasets: List of datasets to filter by (None for all)
        
    Returns:
        Dictionary with metrics for each dataset
    """
    # Load benchmark to get ground truth data
    benchmark = Benchmark(
        benchmark_path, 
        history_length=history_length, 
        forecast_horizon=forecast_horizon, 
        stride=stride
    )
    
    # Get category names for SaveManager
    category_names = benchmark.category_names.copy()
    if "ALL_DATASETS" in category_names:
        del category_names["ALL_DATASETS"]
    
    # Create SaveManager to load forecasts
    save_manager = SaveManager(
        forecast_save_path, 
        category_names, 
        model_name, 
        stride, 
        history_length, 
        forecast_horizon
    )
    
    # Get forecast generator
    forecast_generator = save_manager.load_forecasts(forecast_save_path, model_name)
    
    # Create dataset iterator
    dataset_iterator = _create_dataset_iterator(benchmark, categories, domains, datasets)
    
    # Dictionary to store metrics for each dataset
    dataset_metrics = {}
    
    # Process forecasts and datasets in parallel using two iterators
    forecast_data = next(forecast_generator, None)
    dataset_data = next(dataset_iterator, None)
    
    while forecast_data is not None and dataset_data is not None:
        forecast_category = forecast_data['category']
        forecast_domain = forecast_data['domain']
        forecast_dataset = forecast_data['dataset']
        forecast_window_idx = forecast_data['window_idx']
        forecast_values = forecast_data['forecast']
        
        dataset_category, dataset_domain, dataset_name, dataset_obj = dataset_data
        
        # Check if forecast matches current dataset
        if (forecast_category == dataset_category and 
            forecast_domain == dataset_domain and 
            forecast_dataset == dataset_name):
            
            # Get the specific window from dataset
            target_window = _get_window_by_index(dataset_obj, forecast_window_idx)
            
            if target_window is not None:
                # Compute metrics for this window using existing metrics function
                target_values = target_window.target()
                
                # Ensure forecast and target have same length
                min_length = min(len(forecast_values), len(target_values))
                forecast_values = forecast_values[:min_length]
                target_values = target_values[:min_length]
                
                # Use existing metrics function
                window_metrics = evaluate_metrics(target_values, forecast_values)
                
                # Store metrics
                dataset_key = f"{dataset_category}/{dataset_domain}/{dataset_name}"
                if dataset_key not in dataset_metrics:
                    dataset_metrics[dataset_key] = {
                        'metrics': [],
                        'windows': 0
                    }
                
                dataset_metrics[dataset_key]['metrics'].append(window_metrics)
                dataset_metrics[dataset_key]['windows'] += 1
            
            # Move to next forecast
            forecast_data = next(forecast_generator, None)
        else:
            # Move to next dataset
            dataset_data = next(dataset_iterator, None)
    
    # Compute average metrics for each dataset using existing aggregation function
    final_metrics = {}
    for dataset_key, data in dataset_metrics.items():
        if data['metrics']:
            avg_metrics, total_windows, dataset_count = _aggregate_metrics(data['metrics'])
            final_metrics[dataset_key] = avg_metrics
            final_metrics[dataset_key]['windows'] = total_windows
        else:
            final_metrics[dataset_key] = {
                'MAPE': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'NMAE': np.nan,
                'windows': 0
            }
    
    return final_metrics


def _create_dataset_iterator(
    benchmark: Benchmark, 
    categories: Optional[List[str]] = None,
    domains: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None
) -> Iterator[Tuple[str, str, str, object]]:
    """
    Create an iterator over datasets in the benchmark.
    
    Args:
        benchmark: Benchmark object
        categories: List of categories to filter by
        domains: List of domains to filter by  
        datasets: List of datasets to filter by
        
    Yields:
        Tuple of (category, domain, dataset_name, dataset_object)
    """
    for category in benchmark:
        # Apply category filter
        if categories and category.category not in categories:
            continue
            
        for domain in category:
            # Apply domain filter
            if domains and domain.domain_name not in domains:
                continue
                
            for dataset in domain:
                # Apply dataset filter
                if datasets and dataset.dataset_name not in datasets:
                    continue
                    
                yield (
                    category.category_path.name,
                    domain.domain_name, 
                    dataset.dataset_name,
                    dataset
                )


def _get_window_by_index(dataset_obj, window_idx: int):
    """
    Get a specific window by index from a dataset.
    
    Args:
        dataset_obj: Dataset object
        window_idx: Index of the window to retrieve
        
    Returns:
        Window object or None if not found
    """
    window_count = 0
    for window in dataset_obj:
        if window_count == window_idx:
            return window
        window_count += 1
    return None



def _process_single_model(
    model_name: str,
    forecast_save_path: str,
    benchmark_path: str,
    stride: int,
    history_length: int,
    forecast_horizon: int,
    output_path: str,
    categories: Optional[List[str]] = None,
    domains: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None
) -> None:
    """Process metrics computation for a single model."""
    print(f"\nComputing metrics for model: {model_name}")
    print(f"Forecast save path: {forecast_save_path}")
    print(f"Benchmark path: {benchmark_path}")
    
    # Compute metrics
    dataset_metrics = compute_metrics_from_saved_forecasts(
        forecast_save_path=forecast_save_path,
        benchmark_path=benchmark_path,
        model_name=model_name,
        stride=stride,
        history_length=history_length,
        forecast_horizon=forecast_horizon,
        categories=categories,
        domains=domains,
        datasets=datasets
    )
    
    # Convert dataset metrics to format expected by existing aggregation functions
    results_for_export = {
        model_name: {
            'dataset_results': [],
            'category_results': {},
            'domain_results': {}
        }
    }
    
    # Convert dataset metrics
    for dataset_key, metrics in dataset_metrics.items():
        results_for_export[model_name]['dataset_results'].append({
            'dataset_name': dataset_key,
            'window_count': metrics.get('windows', 0),
            'metrics': {k: v for k, v in metrics.items() if k != 'windows'}
        })
    
    # Use existing aggregation function to compute overall metrics
    overall_results = results_for_export[model_name]['dataset_results']
    if overall_results:
        avg_metrics, total_windows, dataset_count = _aggregate_metrics(overall_results)
        
        # Print summary
        print(f"\nOverall Metrics for {model_name}:")
        for metric, value in avg_metrics.items():
            print(f"  {metric}: {value:.4f}")
        print(f"  Windows: {total_windows}")
    
    # Export to CSV using existing function
    export_hierarchical_results_to_csv(results_for_export, output_path)


def main():
    """Example usage of the compute_metrics_from_saved_forecasts function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute metrics from saved MUSEval forecasts")
    parser.add_argument("--forecast-save-path", required=True, help="Path where forecasts were saved")
    parser.add_argument("--benchmark-path", required=True, help="Path to benchmark directory")
    parser.add_argument("--model-name", help="Name of a single model to process")
    parser.add_argument("--model-names", help="Comma-separated list of model names to process")
    parser.add_argument("--stride", type=int, default=256, help="Stride used")
    parser.add_argument("--history-length", type=int, default=512, help="History length")
    parser.add_argument("--forecast-horizon", type=int, default=128, help="Forecast horizon")
    parser.add_argument("--output-path", default="./metrics", help="Output path for CSV files")
    parser.add_argument("--categories", help="Comma-separated list of categories")
    parser.add_argument("--domains", help="Comma-separated list of domains")
    parser.add_argument("--datasets", help="Comma-separated list of datasets")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model_name and not args.model_names:
        parser.error("Either --model-name or --model-names must be specified")
    
    if args.model_name and args.model_names:
        parser.error("Cannot specify both --model-name and --model-names")
    
    # Parse filter arguments
    categories = args.categories.split(',') if args.categories else None
    domains = args.domains.split(',') if args.domains else None
    datasets = args.datasets.split(',') if args.datasets else None
    
    # Determine which models to process
    if args.model_name:
        model_names = [args.model_name]
    else:
        model_names = [name.strip() for name in args.model_names.split(',')]
    
    print(f"Processing {len(model_names)} model(s): {', '.join(model_names)}")
    print(f"Forecast save path: {args.forecast_save_path}")
    print(f"Benchmark path: {args.benchmark_path}")
    print(f"Output path: {args.output_path}")
    
    # Process each model
    for i, model_name in enumerate(model_names, 1):
        print(f"\n{'='*60}")
        print(f"Processing model {i}/{len(model_names)}: {model_name}")
        print(f"{'='*60}")
        
        _process_single_model(
            model_name=model_name,
            forecast_save_path=args.forecast_save_path,
            benchmark_path=args.benchmark_path,
            stride=args.stride,
            history_length=args.history_length,
            forecast_horizon=args.forecast_horizon,
            output_path=args.output_path,
            categories=categories,
            domains=domains,
            datasets=datasets
        )
        print(f"✓ Successfully processed {model_name}")
    
    print(f"\n{'='*60}")
    print("All models processed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
