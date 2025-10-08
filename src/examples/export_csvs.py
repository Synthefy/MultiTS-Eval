"""
CSV export functionality for MUSED-FM examples.

This module contains functions for:
- Exporting results to CSV files
- Hierarchical CSV export by category, domain, and dataset
- Cleaning output directories
"""

import os
import glob
import pandas as pd
from pathlib import Path
from musedfm.data import Benchmark


def _clean_output_directories(output_dir: str) -> None:
    """Clean output directories before generating new CSV files."""
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
    
    print("\nHierarchical CSV files saved to:")
    print(f"  Categories: {category_dir}/")
    print(f"  Domains: {domain_dir}/")
    print(f"  Datasets: {dataset_dir}/")


def export_results_to_csv(benchmark_path: str, models: dict, max_windows: int = None, output_dir: str = "/tmp",
                         categories: str = None, domains: str = None, datasets: str = None,
                         history_length: int = 512, forecast_horizon: int = 128, stride: int = 256, load_cached_counts: bool = False):
    """Export forecast results to CSV files."""
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
        if categories is not None and category.category_path.name not in categories:
            continue
        for domain in category:
            if domains is not None and domain.domain_path.name not in domains:
                continue
            for dataset in domain:
                # Apply filters if specified
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
                    
                    # Submit forecast based on model type
                    is_univariate = models[model_name]["univariate"]
                    if is_univariate:
                        # For univariate models, pass the forecast as univariate_forecast
                        window.submit_forecast(univariate_forecast=forecast)
                    else:
                        # For multivariate models, pass the forecast as multivariate_forecast
                        window.submit_forecast(multivariate_forecast=forecast)
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
