"""
Category class for grouping domains with similar data properties.
"""

import os
import json
from typing import Iterator, Dict, Any, List
from pathlib import Path
from musedfm.data.domain import Domain
from collections import OrderedDict

class Category:
    """
    A grouping of domains which have similar data properties.
    The four categories are named based on the tar.gz files: traditional, collections, sequential and synthetic.
    Iterable: for domain in category
    Aggregates results across domains.
    """
    
    def __init__(self, category_path: str, base_path: str, dataset_domain_map: Dict[str, str], dataset_category_map: Dict[str, str], category_names: Dict[str, Dict[str, List[str]]], domain_category_map: Dict[str, str], history_length: int = 30, forecast_horizon: int = 1, stride: int = 1, load_cached_counts: bool = False):
        """
        Initialize category from directory containing multiple domain directories.
        
        Args:
            category_path: Path to directory containing domain subdirectories
            dataset_domain_map: Dictionary mapping dataset names to domain names
            dataset_category_map: Dictionary mapping dataset names to category names
            category_names: Dictionary of category names and their domains/datasets
            domain_category_map: Dictionary mapping domain names to category names
            history_length: Number of historical points to use for forecasting
            forecast_horizon: Number of future points to forecast
            stride: Step size between windows
            load_cached_counts: If True, load window counts from cached JSON files instead of generating
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
        self.load_cached_counts = load_cached_counts
        self.iterated_dataset_dict = json.load(open(Path(__file__).parent / "iterated_datasets.json"))
        self._domains: List[Domain] = []
        count_filename = f"{self.category}_window_counts_h{self.history_length}_f{self.forecast_horizon}_s{self.stride}.json"
        needs_counting = not self.load_cached_counts or not os.path.exists(os.path.join(base_path, count_filename))
        self._load_domains(needs_counting)
        
        # Load cached window counts if requested
        if load_cached_counts:
            self._load_cached_window_counts(base_path)
    
    def _load_domains(self, needs_counting: bool=True) -> None:
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
            domain = Domain(domain, self.per_domain_paths[domain], self.category_names, self.domain_category_map, self.history_length, self.forecast_horizon, self.stride, needs_counting=needs_counting)
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
    
    def save_window_counts(self, base_path: str = "") -> None:
        """Save window counts to JSON file and copy to compressed tar directory."""        
        os.makedirs(base_path, exist_ok=True)
        
        # Create filename with parameters
        filename = f"{self.category}_window_counts_h{self.history_length}_f{self.forecast_horizon}_s{self.stride}.json"

        window_counts = {dataset.dataset_name: len(dataset) for domain in self._domains for dataset in domain._datasets}
        
        json_path = os.path.join(base_path, filename)
        print(f"saving window counts to {json_path}, window counts: {window_counts}")
        with open(json_path, 'w') as f:
            json.dump(window_counts, f, indent=2)
        return window_counts
    
    def load_window_counts(self, base_path: str = "") -> Dict[str, int]:
        """Load window counts from JSON file if present."""
        filename = f"{self.category}_window_counts_h{self.history_length}_f{self.forecast_horizon}_s{self.stride}.json"
        json_path = os.path.join(base_path, filename)
        print(f"loading window counts from {json_path}")
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
        else:
            self.get_window_counts(use_cached=False, base_path=base_path)
            window_counts = self.save_window_counts(base_path)
            return window_counts

    def get_window_counts(self, use_cached: bool = True, base_path: str = "") -> Dict[str, int]:
        """Get window counts, optionally from cached JSON file."""
        if use_cached:
            cached_counts = self.load_window_counts(base_path)
            if cached_counts:
                return cached_counts
        
        return {dataset.dataset_name: len(dataset) for domain in self._domains for dataset in domain._datasets}
    
    def _load_cached_window_counts(self, base_path: str = "") -> None:
        """Load cached window counts and override dataset window counts."""
        cached_counts = self.load_window_counts(base_path)
        print(f"loaded {len(cached_counts)} cached window counts for category {self.category}")
        print(list(cached_counts.keys()))
        if cached_counts:
            # Override window counts for datasets that have cached values
            for domain in self._domains:
                for dataset in domain._datasets:
                    if dataset.dataset_name in cached_counts:
                        dataset._total_windows = cached_counts[dataset.dataset_name]
        print("successfully counted windows from cached JSON files")