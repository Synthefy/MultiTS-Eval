"""
Benchmark class as top-level container for multiple categories.
"""

import json
from typing import Iterator, Dict, Any, List, Optional, Union
from pathlib import Path
from .category import Category
from .dataset import Dataset
from collections import OrderedDict


class Benchmark:
    """
    Top-level container for multiple categories.
    Simple orchestrator for evaluations and exports.
    """
    
    def __init__(self, benchmark_path: str, history_length: int = 30, forecast_horizon: int = 1, stride: int = 1):
        """
        Initialize benchmark from directory containing multiple category directories.
        
        Args:
            benchmark_path: Path to directory containing category subdirectories
            history_length: Number of historical points to use for forecasting
            forecast_horizon: Number of future points to forecast
            stride: Step size between windows
        """
        self.benchmark_path = Path(benchmark_path)
        self.history_length = history_length
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self.categories: List[Category] = []
        self._data_hierarchy = self._load_data_hierarchy()
        self.category_names = OrderedDict()
        self.dataset_domain_map = OrderedDict()
        self.dataset_category_map = OrderedDict()
        self._load_categories()
    
    def _load_data_hierarchy(self) -> Dict[str, Any]:
        """Load the data hierarchy from data_hierarchy.json."""
        hierarchy_path = Path(__file__).parent / "data_hierarchy.json"
        try:
            with open(hierarchy_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: data_hierarchy.json not found at {hierarchy_path}")
            return {"datasets": []}
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse data_hierarchy.json: {e}")
            return {"datasets": []}
    
    def _load_categories(self) -> None:
        """Load all categories from subdirectories."""
        # Find all categories that fit the hierarchy     
        self.category_names = OrderedDict()
        self.dataset_domain_map = OrderedDict()
        self.dataset_category_map = OrderedDict()
        self.domain_category_map = OrderedDict()
        for item in self._data_hierarchy.get("domains", []):
            if item.get("category", "") not in self.category_names:
                self.category_names[item["category"]] = OrderedDict()
                self.category_names[item["category"]]["ALL_DATASETS"] = []
            domain = item["domain"]
            domain_datasets = item["dataset_name"]
            self.category_names[item["category"]][domain] = domain_datasets
            self.category_names[item["category"]]["ALL_DATASETS"].extend(domain_datasets)
            for dataset in domain_datasets:
                self.dataset_domain_map[dataset] = domain
                self.dataset_category_map[dataset] = item["category"]
                if domain in self.domain_category_map:
                    assert self.domain_category_map[domain] == item["category"], f"Domain {domain} already has a different category {self.domain_category_map[domain]}"
                self.domain_category_map[domain] = item["category"]
        # load all available category_names
        category_candidates = list()
        self.category_names = {k.lower(): v for k, v in self.category_names.items()}
        for item in self.benchmark_path.iterdir():
            if item.name.lower() in self.category_names:
                category_candidates.append(item)
        
        for category in self.category_names:
            if category not in [c.name for c in category_candidates]:
                print(f"Category {category} not found in filesystem")
        
        self.categories = OrderedDict()
        for category_path in category_candidates:
            category = Category(str(category_path), self.dataset_domain_map, self.dataset_category_map, self.category_names, self.domain_category_map, self.history_length, self.forecast_horizon, self.stride)
            self.categories[category.category] = category
        
        self.dataset_filepaths = sum([cat.dataset_filepaths for cat in self.categories.values()], [])
    
    def __iter__(self) -> Iterator[Category]:
        """Iterate over categories."""
        return iter(self.categories.values())
    
    def __len__(self) -> int:
        """Return number of categories."""
        return len(self.categories)
    
    def evaluate(self) -> Dict[str, float]:
        """
        Aggregate results across categories (cached results only).
        
        Returns:
            Dictionary of aggregated evaluation metrics across all categories
        """
        if not self.categories:
            return {}
        
        # Evaluate each category
        category_results = []
        for category in self.categories:
            results = category.evaluate()
            category_results.append(results)
        
        if not category_results:
            return {}
        
        # Aggregate results across categories
        aggregated = {}
        for metric in category_results[0].keys():
            values = [result[metric] for result in category_results if metric in result]
            if values:
                aggregated[metric] = sum(values) / len(values)  # Simple average
                aggregated[f"{metric}_std"] = (sum((v - aggregated[metric])**2 for v in values) / len(values))**0.5
        
        return aggregated
    
    def get(self, 
            collections: Optional[Union[str, List[str]]] = None,
            domains: Optional[Union[str, List[str]]] = None, 
            datasets: Optional[Union[str, List[str]]] = None) -> Iterator[Dataset]:
        """
        Get datasets filtered by collections (categories), domains, and/or datasets.
        
        Args:
            collections: Collection name(s) to filter by (e.g., "Traditional", "Synthetic")
            domains: Domain name(s) to filter by (e.g., "Energy", "Finance") 
            datasets: Dataset name(s) to filter by (e.g., "al_daily", "bitcoin_price")
            
        Returns:
            Iterator over matching Dataset objects
            
        Examples:
            # Get all datasets from Traditional collection
            for dataset in benchmark.get(collections="Traditional"):
                pass
                
            # Get all Energy domain datasets
            for dataset in benchmark.get(domains="Energy"):
                pass
                
            # Get specific datasets
            for dataset in benchmark.get(datasets=["al_daily", "bitcoin_price"]):
                pass
                
            # Combine filters
            for dataset in benchmark.get(collections="Traditional", domains="Energy"):
                pass
        """
        # Normalize inputs to lists
        if collections is not None:
            collections = [collections] if isinstance(collections, str) else collections
        if domains is not None:
            domains = [domains] if isinstance(domains, str) else domains
        if datasets is not None:
            datasets = [datasets] if isinstance(datasets, str) else datasets
        
        # Find matching datasets
        matching_datasets = set()
        
        # Iterate through collections (or all categories if None)
        collections_to_check = collections if collections is not None else list(self.category_names.keys())
        
        for collection in collections_to_check:
            if collection not in self.category_names:
                continue
                
            category_data = self.category_names[collection]
            
            # Iterate through domains (or all domains if None)
            domains_to_check = domains if domains is not None else [k for k in category_data.keys() if k != "ALL_DATASETS"]
            
        for domain in domains_to_check:
            if domain not in category_data:
                continue
                
            domain_datasets = category_data[domain]
            
            # Iterate through datasets (or all datasets if None)
            datasets_to_check = datasets if datasets is not None else domain_datasets
            
        for dataset in datasets_to_check:
            if dataset in domain_datasets:
                matching_datasets.add(dataset)
        
        # Yield matching datasets from the actual file system
        for category in self.categories:
            for domain in category:
                for dataset in domain:
                    # Extract dataset name from full path hierarchy
                    dataset_name = str(dataset.data_path.relative_to(self.benchmark_path))
                    if dataset_name in matching_datasets:
                        yield dataset
        
    def to_results_csv(self, path: str) -> None:
        """
        Validate forecasts, gather results from all categories, write consolidated CSV.
        
        Args:
            path: Output CSV file path
        """
        import pandas as pd
        
        # Collect results from all categories
        all_results = []
        category_names = []
        
        for i, category in enumerate(self.categories):
            category_name = f"category_{i}"
            category_names.append(category_name)
            
            # Get individual window results from all domains in this category
            for j, domain in enumerate(category):
                domain_name = f"{category_name}_domain_{j}"
                
                for k, dataset in enumerate(domain):
                    dataset_name = f"{domain_name}_dataset_{k}"
                    
                    for window_idx, window in enumerate(dataset):
                        if window.has_forecast:
                            window_result = window.evaluate()
                            window_result['category'] = category_name
                            window_result['domain'] = domain_name
                            window_result['dataset'] = dataset_name
                            window_result['window_id'] = window_idx
                            all_results.append(window_result)
                        
        if not all_results:
            raise ValueError("No valid results found from any category")
        
        # Create consolidated DataFrame
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(path, index=False)
        
        # Also save aggregated results
        aggregated_results = self.evaluate()
        aggregated_df = pd.DataFrame([aggregated_results])
        aggregated_path = path.replace('.csv', '_aggregated.csv')
        aggregated_df.to_csv(aggregated_path, index=False)
        
        print("Results saved to", path)
        print("Aggregated results saved to", aggregated_path)
        
        # Print summary statistics
        print("\nSummary:")
        print(f"Total categories: {len(self.categories)}")
        print(f"Total windows with forecasts: {len(all_results)}")
        print("Average metrics across all categories:")
        for metric, value in aggregated_results.items():
            if not metric.endswith('_std'):
                print(f"  {metric}: {value:.4f}")
