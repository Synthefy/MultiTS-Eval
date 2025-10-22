"""
CSV export functionality for MultiTS-Eval examples.

This module contains functions for:
- Exporting results to CSV files
- Hierarchical CSV export by category, domain, and dataset
- Cleaning output directories
"""

import os
import glob
import pandas as pd
from pathlib import Path
from typing import Optional
from multieval.data import Benchmark


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
