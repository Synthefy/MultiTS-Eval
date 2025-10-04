"""
Category class for grouping domains with similar data properties.
"""

import os
import json
from typing import Iterator, Dict, Any, List
from pathlib import Path
from .domain import Domain
from collections import OrderedDict

class Category:
    """
    A grouping of domains which have similar data properties.
    The four categories are named based on the tar.gz files: traditional, collections, sequential and synthetic.
    Iterable: for domain in category
    Aggregates results across domains.
    """
    
    def __init__(self, category_path: str, dataset_domain_map: Dict[str, str], dataset_category_map: Dict[str, str], category_names: Dict[str, Dict[str, List[str]]], domain_category_map: Dict[str, str], history_length: int = 30, forecast_horizon: int = 1, stride: int = 1):
        """
        Initialize category from directory containing multiple domain directories.
        
        Args:
            category_path: Path to directory containing domain subdirectories
            history_length: Number of historical points to use for forecasting
            forecast_horizon: Number of future points to forecast
            stride: Step size between windows
        """

        self.category_path = Path(category_path)
        self.category = self.category_path.name
        self.history_length = history_length
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self.dataset_domain_map = dataset_domain_map
        self.dataset_category_map = dataset_category_map
        self.domain_category_map = domain_category_map
        self.category_names = category_names
        self.iterated_dataset_dict = json.load(open(Path(__file__).parent / "iterated_datasets.json"))
        self._domains: List[Domain] = []
        self._load_domains()
    
    def _load_domains(self) -> None:
        """Load domains based on data_hierarchy.json, creating virtual domains for datasets."""
        # utilize the dataset_domain_map and dataset_category_map to load the domains
        self.per_domain_paths = OrderedDict()
        for dataset_path in self.category_path.iterdir():
            if dataset_path.name in self.iterated_dataset_dict:
                dataset_subpaths = [dataset_path / subpath for subpath in self.iterated_dataset_dict[dataset_path.name]]
            else:
                dataset_subpaths = [dataset_path]
            for dataset_subpath in dataset_subpaths:
                if dataset_subpath.name in self.dataset_domain_map:
                    domain_name = self.dataset_domain_map[dataset_subpath.name]
                    if domain_name not in self.per_domain_paths:
                        self.per_domain_paths[domain_name] = []
                    self.per_domain_paths[domain_name].append(dataset_subpath)
                else:
                    print(f"Dataset {dataset_subpath.name} not found in data_hierarchy.json")
        
        # load the domains and check if there are any missing domains or datasets
        for domain in self.per_domain_paths:
            domain = Domain(domain, self.per_domain_paths[domain], self.category_names, self.domain_category_map, self.history_length, self.forecast_horizon, self.stride)
            self._domains.append(domain)
        
        for domain in self.category_names[self.category]:
            if domain not in self.per_domain_paths:
                print(f"Domain {domain} not found in file hierarchy")

        self.dataset_filepaths = sum([domain.dataset_paths for domain in self._domains], [])

    
    def __iter__(self) -> Iterator[Domain]:
        """Iterate over domains."""
        return iter(self._domains)
    
    def __len__(self) -> int:
        """Return number of domains."""
        return len(self._domains)
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation across all domains in the category.
        
        Returns:
            Dictionary of aggregated evaluation metrics across all domains
        """
        if not self._domains:
            return {}
        
        # Evaluate each domain
        domain_results = []
        for domain in self._domains:
            results = domain.evaluate()
            domain_results.append(results)
        
        if not domain_results:
            return {}
        
        # Aggregate results across domains
        aggregated = {}
        for metric in domain_results[0].keys():
            values = [result[metric] for result in domain_results if metric in result]
            if values:
                aggregated[metric] = sum(values) / len(values)  # Simple average
                aggregated[f"{metric}_std"] = (sum((v - aggregated[metric])**2 for v in values) / len(values))**0.5
        
        return aggregated
    
    def to_results_csv(self, path: str) -> None:
        """
        Validate forecasts, gather results from all domains, write consolidated CSV.
        Same semantics as dataset: validate → aggregate → save.
        
        Args:
            path: Output CSV file path
        """
        import pandas as pd
        
        # Collect results from all domains
        all_results = []
        domain_names = []
        
        for i, domain in enumerate(self._domains):
            domain_name = f"domain_{i}"
            domain_names.append(domain_name)
            
            # Get individual window results from all datasets in this domain
            for j, dataset in enumerate(domain):
                dataset_name = f"{domain_name}_dataset_{j}"
                    
                for k, window in enumerate(dataset):
                    if window.has_forecast:
                        window_result = window.evaluate()
                        window_result['domain'] = domain_name
                        window_result['dataset'] = dataset_name
                        window_result['window_id'] = k
                        all_results.append(window_result)
                
            window_result['domain'] = domain_name
            window_result['dataset'] = dataset_name
            window_result['window_id'] = k
            all_results.append(window_result)
        
        if not all_results:
            raise ValueError("No valid results found from any domain")
        
        # Create consolidated DataFrame
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(path, index=False)
        
        # Also save aggregated results
        aggregated_results = self.evaluate()
        aggregated_df = pd.DataFrame([aggregated_results])
        aggregated_path = path.replace('.csv', '_aggregated.csv')
        aggregated_df.to_csv(aggregated_path, index=False)
        
        print(f"Results saved to {path}")
        print(f"Aggregated results saved to {aggregated_path}")
