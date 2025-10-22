"""
Save submission file in the required format for MultiTS-Eval competition.

This module provides functionality to convert results from MultiTS-Eval evaluation
into the submission format required by the competition.
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from multieval.data.benchmark import VERSION
import numpy as np


def save_submission(results: Dict[str, Any], output_dir: str, model_name: Optional[str] = None) -> None:
    """
    Save results in the submission format required by MultiTS-Eval competition.
    Creates separate JSON files for each model.
    
    Args:
        results: Results dictionary from MultiTS-Eval evaluation containing metrics
        output_dir: Directory where to save the submission JSON files
        model_name: Optional specific model name to save (if None, saves all models)
        
    The results dictionary should have the structure:
    {
        'model_name': {
            'dataset_results': [
                {
                    'dataset_name': 'dataset_name',
                    'metrics': {'MAE': 15.0, 'RMSE': 15.0, 'Multi-MAPE': 15.0, 'Uni-MAPE': 18.0, 'NMAE': 9.0},
                    'window_count': 100
                },
                ...
            ]
        }
    }
    """
    
    # Load the data hierarchy to get domain and category mappings
    hierarchy_path = Path(__file__).parent.parent / "multieval" / "data" / "data_hierarchy.json"
    with open(hierarchy_path, 'r') as f:
        data_hierarchy = json.load(f)
    
    # Create domain and category mappings
    dataset_domain_map = {}
    dataset_category_map = {}
    
    for domain_info in data_hierarchy.get("domains", []):
        domain = domain_info["domain"]
        category = domain_info["category"]
        for dataset_name in domain_info["dataset_name"]:
            dataset_domain_map[dataset_name] = domain
            dataset_category_map[dataset_name] = category
    
    # Determine which models to process (skip univariate models)
    if model_name and model_name in results:
        models_to_process = [model_name] if not model_name.endswith('_univariate') else []
    else:
        models_to_process = [name for name in results.keys() if not name.endswith('_univariate')]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each model
    for current_model_name in models_to_process:
        model_results = results[current_model_name]
        
        # Handle case where model_results might be a list instead of dict
        if isinstance(model_results, list):
            print(f"Warning: model_results for {current_model_name} is a list, not a dict.")
            print(f"This suggests the results structure was modified incorrectly.")
            print(f"Skipping this model. Please check your notebook execution.")
            continue
        
        # Convert to submission format
        submission_data = []
        
        for dataset_result in model_results['dataset_results']:
            dataset_name = dataset_result['dataset_name']
            
            # Extract domain and category from mappings
            domain = dataset_domain_map.get(dataset_name, "Unknown")
            category = dataset_category_map.get(dataset_name, "unknown")
            
            # Extract metrics with defaults
            metrics = dataset_result.get('metrics', {})
            
            # Get Uni-MAPE from corresponding univariate model if available
            univariate_model_name = f"{current_model_name}_univariate"
            uni_mape = metrics.get('Uni-MAPE', 0.0)
            
            if univariate_model_name in results:
                # Find the same dataset in univariate results
                for univariate_dataset_result in results[univariate_model_name]['dataset_results']:
                    if univariate_dataset_result['dataset_name'] == dataset_name:
                        univariate_metrics = univariate_dataset_result.get('metrics', {})
                        uni_mape = univariate_metrics.get('MAPE', np.nan)  # Uni-MAPE is MAPE from univariate model
                        break
            else:
                uni_mape = metrics.get('MAPE', 0.0)
            
            submission_metrics = {
                'MAE': metrics.get('MAE', 0.0),
                'RMSE': metrics.get('RMSE', 0.0),
                'Multi-MAPE': metrics.get('MAPE', 0.0),
                'Uni-MAPE': uni_mape,
                'NMAE': metrics.get('NMAE', 0.0)
            }

            # replace infinity with 1000
            submission_metrics = {metric: (1000 if np.isinf(value) else value) for metric, value in submission_metrics.items()}
            
            # Create submission entry
            submission_entry = {
                "domain": domain,
                "category": category,
                "dataset": dataset_name,
                "dataset_version": f"v{VERSION}",
                "metrics": submission_metrics
            }
            
            submission_data.append(submission_entry)
        
        # Sort by category, then domain, then dataset name for consistent output
        submission_data.sort(key=lambda x: (x["category"], x["domain"], x["dataset"]))
        
        # Save to file
        output_path = os.path.join(output_dir, f"{current_model_name}_submission.json")
        with open(output_path, 'w') as f:
            json.dump(submission_data, f, indent=2)
        
        print(f"Submission file saved to: {output_path}")
        print(f"Model: {current_model_name}")
        print(f"Total datasets: {len(submission_data)}")
        print(f"Domains: {len(set(entry['domain'] for entry in submission_data))}")
        print(f"Categories: {len(set(entry['category'] for entry in submission_data))}")
        print()


def create_sample_submission(output_path: str) -> None:
    """
    Create a sample submission file with placeholder values for all datasets.
    
    Args:
        output_path: Path where to save the sample submission JSON file
    """
    
    # Load the data hierarchy to get all datasets
    hierarchy_path = Path(__file__).parent.parent / "multieval" / "data" / "data_hierarchy.json"
    with open(hierarchy_path, 'r') as f:
        data_hierarchy = json.load(f)
    
    # Create sample submission data
    submission_data = []
    
    for domain_info in data_hierarchy.get("domains", []):
        domain = domain_info["domain"]
        category = domain_info["category"]
        
        for dataset_name in domain_info["dataset_name"]:
            # Sample metrics (replace with actual values)
            sample_metrics = {
                "MAE": 15.0,
                "RMSE": 15.0,
                "Multi-MAPE": 15.0,
                "Uni-MAPE": 18.0,
                "NMAE": 9.0
            }
            
            submission_entry = {
                "domain": domain,
                "category": category,
                "dataset": dataset_name,
                "dataset_version": f"v{VERSION}",
                "metrics": sample_metrics
            }
            
            submission_data.append(submission_entry)
    
    # Sort by category, then domain, then dataset name for consistent output
    submission_data.sort(key=lambda x: (x["category"], x["domain"], x["dataset"]))
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(submission_data, f, indent=2)
    
    print(f"Sample submission file saved to: {output_path}")
    print(f"Total datasets: {len(submission_data)}")
    print(f"Domains: {len(set(entry['domain'] for entry in submission_data))}")
    print(f"Categories: {len(set(entry['category'] for entry in submission_data))}")


def main():
    """Example usage of the save_submission functions."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Save MultiTS-Eval submission file")
    parser.add_argument("--results", type=str, help="Path to results JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output directory for submission files")
    parser.add_argument("--model", type=str, help="Model name to use from results")
    parser.add_argument("--sample", action="store_true", help="Create sample submission file")
    
    args = parser.parse_args()
    
    if args.sample:
        create_sample_submission(args.output)
    else:
        if not args.results:
            print("Error: --results is required when not using --sample")
            return
        
        # Load results
        with open(args.results, 'r') as f:
            results = json.load(f)
        
        # Save submission
        save_submission(results, args.output, args.model)


if __name__ == "__main__":
    main()
