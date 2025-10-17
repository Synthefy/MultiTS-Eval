"""
Domain class for encapsulating multiple datasets.
"""

from typing import Iterator, Dict, List
from pathlib import Path
from musedfm.data.dataset import Dataset


class Domain:
    """
    Encapsulates multiple datasets.
    Iterable: for dataset in domain
    Aggregates results across datasets.
    """
    
    def __init__(self, domain_name: str, dataset_paths: List[Path], 
    category_names: Dict[str, Dict[str, List[str]]], domain_category_map: Dict[str, str], 
    history_length: int = 30, forecast_horizon: int = 1, stride: int = 1, needs_counting: bool = True, batch_size: int = 1):
        """
        Initialize domain from directory containing multiple dataset directories.
        
        Args:
            domain_name: Name of the domain
            dataset_paths: List of paths to dataset directories
            category_names: Dictionary of category names
            domain_category_map: Dictionary of domain to category mapping
            history_length: Number of historical points to use for forecasting
            forecast_horizon: Number of future points to forecast
            stride: Step size between windows
            needs_counting: If True, count windows
            batch_size: Number of windows to collect in each batch (default: 1 for single window mode)
        """
        self.domain_name = domain_name
        self.dataset_paths = dataset_paths
        self.history_length = history_length
        self.forecast_horizon = forecast_horizon
        self.category_names = category_names
        self.domain_category_map = domain_category_map
        self.stride = stride
        self.batch_size = batch_size
        self._datasets: List[Dataset] = []
        self.needs_counting = needs_counting
        self._load_datasets()
        
    def _load_datasets(self) -> None:
        """Load all datasets from subdirectories, including parquet files from subfolders."""
        # assumes that dataset paths are already validated
        for d in self.dataset_paths:
            if d.is_dir():
                # Check if this directory or any of its subdirectories contain parquet files
                parquet_files = list(d.rglob("*.parquet"))
                if parquet_files:
                    dataset = Dataset.from_config(str(d), d.name, self.history_length, self.forecast_horizon, self.stride, needs_counting=self.needs_counting, batch_size=self.batch_size)
                    self._datasets.append(dataset)
                else:
                    print(f"Dataset {d.name} has no parquet files")
            else:
                print(f"Dataset {d.name} not a directory")
        
        str_dataset_paths = [path.name for path in self.dataset_paths]
        for dataset in self.category_names[self.domain_category_map[self.domain_name]][self.domain_name]:
            if str(dataset) not in str_dataset_paths:
                print(f"Dataset {dataset} not found in file hierarchy")

    
    def __iter__(self) -> Iterator[Dataset]:
        """Iterate over datasets."""
        return iter(self._datasets)
    
    def __len__(self) -> int:
        """Return number of datasets."""
        return len(self._datasets)
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation across all datasets (cached results only).
        
        Returns:
            Dictionary of aggregated evaluation metrics across all datasets
        """
        if not self._datasets:
            return {}
        
        # Evaluate each dataset
        dataset_results = []
        for dataset in self._datasets:
            results = dataset.evaluate()
            dataset_results.append(results)
        
        if not dataset_results:
            return {}
        
        # Aggregate results across datasets
        aggregated = {}
        for metric in dataset_results[0].keys():
            values = [result[metric] for result in dataset_results if metric in result]
            if values:
                aggregated[metric] = sum(values) / len(values)  # Simple average
                aggregated[f"{metric}_std"] = (sum((v - aggregated[metric])**2 for v in values) / len(values))**0.5
        
        return aggregated
    
    def to_results_csv(self, path: str) -> None:
        """
        Validate forecasts, gather results from all datasets, write consolidated CSV.
        Same semantics as dataset: validate → aggregate → save.
        
        Args:
            path: Output CSV file path
        """
        import pandas as pd
        
        # Collect results from all datasets
        all_results = []
        dataset_names = []
        
        for i, dataset in enumerate(self._datasets):
            dataset_name = f"dataset_{i}"
            dataset_names.append(dataset_name)
            
            # Get individual window results
            window_results = []
            for j, window in enumerate(dataset):
                if window.has_forecast:
                    window_result = window.evaluate()
                    window_result['dataset'] = dataset_name
                    window_result['window_id'] = j
                    window_results.append(window_result)
            
            all_results.extend(window_results)      
        
        if not all_results:
            raise ValueError("No valid results found from any dataset")
        
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
    
    def reset_iterator(self) -> None:
        """Reset iterator state for all datasets in this domain."""
        for dataset in self._datasets:
            dataset.reset_iterator()