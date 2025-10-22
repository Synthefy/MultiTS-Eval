#!/usr/bin/env python3
"""
Check which datasets are present in a benchmark directory and validate their columns.
This script helps identify which datasets are being skipped and why.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multieval.data import Benchmark, Dataset
from multieval.data.dataset import Dataset as DatasetClass
import json

def check_datasets(benchmark_path: str):
    """Check which datasets are being loaded and which are skipped."""
    print("=" * 80)
    print("CHECKING DATASETS IN BENCHMARK")
    print("=" * 80)
    
    # Load the dataset config
    config_path = Path(__file__).parent.parent / "src/multieval/data/dataset_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Available datasets in config: {len(config)}")
    print("Config datasets:", list(config.keys())[:10], "..." if len(config) > 10 else "")
    
    # Load benchmark
    benchmark = Benchmark(benchmark_path, history_length=10, forecast_horizon=5, stride=1)
    print(f"\nLoaded benchmark with {len(benchmark)} categories")
    
    total_datasets = 0
    loaded_datasets = 0
    skipped_datasets = []
    
    # Iterate through benchmark structure
    for category in benchmark:
        print(f"\nCategory: {category.category_path.name}")
        for domain in category:
            print(f"  Domain: {domain.domain_path.name}")
            for dataset in domain:
                total_datasets += 1
                dataset_name = str(dataset.data_path.relative_to(benchmark.benchmark_path))
                print(f"    Dataset: {dataset_name}")
                
                # Check if this dataset has windows
                try:
                    windows = list(dataset)
                    if len(windows) > 0:
                        loaded_datasets += 1
                        print(f"      ✓ Loaded {len(windows)} windows")
                    else:
                        skipped_datasets.append((dataset_name, "No windows generated"))
                        print(f"      ✗ No windows generated")
                except Exception as e:
                    skipped_datasets.append((dataset_name, str(e)))
                    print(f"      ✗ Error: {e}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total datasets found: {total_datasets}")
    print(f"Successfully loaded: {loaded_datasets}")
    print(f"Skipped/failed: {len(skipped_datasets)}")
    
    if skipped_datasets:
        print("\nSkipped datasets:")
        for dataset_name, reason in skipped_datasets:
            print(f"  {dataset_name}: {reason}")
    
    # Check for potential config mismatches
    print("\n" + "=" * 80)
    print("CONFIG ANALYSIS")
    print("=" * 80)
    
    # Find all dataset directories
    benchmark_path_obj = Path(benchmark_path)
    all_dataset_dirs = []
    
    for category_dir in benchmark_path_obj.iterdir():
        if category_dir.is_dir():
            for domain_dir in category_dir.iterdir():
                if domain_dir.is_dir():
                    for dataset_dir in domain_dir.iterdir():
                        if dataset_dir.is_dir() and any(dataset_dir.glob("*.parquet")):
                            dataset_name = str(dataset_dir.relative_to(benchmark_path_obj))
                            all_dataset_dirs.append(dataset_name)
    
    print(f"Found {len(all_dataset_dirs)} dataset directories in filesystem")
    
    # Check which ones have config entries
    config_matches = []
    config_mismatches = []
    
    for dataset_dir in all_dataset_dirs:
        # Extract potential dataset names from path
        path_parts = dataset_dir.split('/')
        potential_names = []
        
        # Look for exact matches
        for part in path_parts:
            if part in config:
                potential_names.append(part)
        
        # Look for partial matches
        for dataset_name in config.keys():
            for part in path_parts:
                if dataset_name in part or part in dataset_name:
                    potential_names.append(dataset_name)
                # Handle underscore variations
                if '_' in dataset_name:
                    name_parts = dataset_name.split('_')
                    if all(any(name_part in path_part for path_part in path_parts) for name_part in name_parts):
                        potential_names.append(dataset_name)
        
        if potential_names:
            config_matches.append((dataset_dir, potential_names[0]))
        else:
            config_mismatches.append(dataset_dir)
    
    print(f"Datasets with config matches: {len(config_matches)}")
    print(f"Datasets without config matches: {len(config_mismatches)}")
    
    if config_mismatches:
        print("\nDatasets without config matches:")
        for dataset_dir in config_mismatches:
            print(f"  {dataset_dir}")
            # Show what columns are available
            try:
                parquet_files = list((benchmark_path_obj / dataset_dir).glob("*.parquet"))
                if parquet_files:
                    import pandas as pd
                    df = pd.read_parquet(parquet_files[0])
                    print(f"    Columns: {list(df.columns)}")
            except Exception as e:
                print(f"    Error reading columns: {e}")
    
    # Show config matches for reference
    if config_matches:
        print("\nDatasets with config matches:")
        for dataset_dir, config_name in config_matches:
            print(f"  {dataset_dir} -> {config_name}")
            # Show the config details
            if config_name in config:
                dataset_config = config[config_name]
                print(f"    Timestamp: {dataset_config.get('timestamp_col', 'N/A')}")
                print(f"    Target: {dataset_config.get('target_cols', 'N/A')}")
                print(f"    Metadata: {dataset_config.get('metadata_cols', 'N/A')}")

def main():
    """Main function for command-line usage."""
    if len(sys.argv) != 2:
        print("Usage: python check_datasets.py <benchmark_path>")
        print("\nExample:")
        print("  python check_datasets.py /path/to/benchmark_data")
        sys.exit(1)
    
    benchmark_path = sys.argv[1]
    
    if not Path(benchmark_path).exists():
        print(f"Error: Benchmark path '{benchmark_path}' does not exist")
        sys.exit(1)
    
    check_datasets(benchmark_path)

if __name__ == "__main__":
    main()
