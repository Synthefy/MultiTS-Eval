"""
MUSED-FM Evaluation Script for GIFT-Eval Submission

This script generates forecast results following GIFT-Eval Hugging Face submission specifications.
The script produces forecast CSV files that can be evaluated by the GIFT-Eval framework.

Usage:
    python eval_musedfm.py --model-name YOUR_MODEL_NAME --model-type deep-learning --benchmark-path /path/to/benchmark
    python eval_musedfm.py --model-name chronos-t5-small --model-type pretrained --output-dir /path/to/results
"""

import os
import sys
import time
import warnings
import numpy as np
import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import contextlib

# Set environment variable to suppress warnings at the system level
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.simplefilter("ignore")

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import MUSED-FM components
from musedfm.data import Benchmark
from musedfm.metrics import MAPE, MAE, RMSE, NMAE

# Note: Following GIFT-Eval conventions without GluonTS dependency

# Suppress all warnings
@contextlib.contextmanager
def suppress_all_warnings():
    """Context manager to suppress all warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


class MUSEDFMPredictor:
    """
    MUSED-FM predictor wrapper following GIFT-Eval conventions.
    """
    
    def __init__(self, model, model_name: str, univariate: bool = True):
        """
        Initialize the predictor.
        
        Args:
            model: MUSED-FM model instance
            model_name: Name of the model
            univariate: Whether the model is univariate
        """
        self.model = model
        self.model_name = model_name
        self.univariate = univariate
        self.prediction_length = None


class MUSEDFMDataset:
    """
    MUSED-FM dataset wrapper following GIFT-Eval conventions.
    """
    
    def __init__(self, benchmark: Benchmark, max_windows: Optional[int] = None):
        """
        Initialize the dataset.
        
        Args:
            benchmark: MUSED-FM benchmark instance
            max_windows: Maximum number of windows per dataset
        """
        self.benchmark = benchmark
        self.max_windows = max_windows
        self.items = []
        self._prepare_dataset()
    
    def _prepare_dataset(self):
        """Prepare the dataset for evaluation."""
        print("Preparing dataset for evaluation...")
        
        item_id = 0
        for category in self.benchmark:
            for domain in category:
                for dataset in domain:
                    window_count = 0
                    for window in dataset:
                        if self.max_windows and window_count >= self.max_windows:
                            break
                        
                        # Create evaluation item
                        item = {
                            'item_id': f"{item_id}",
                            'history': window.history(),
                            'target': window.target(),
                            'covariates': window.covariates() if window.covariates() is not None else None,
                            'info': {
                                'category': category.category,
                                'domain': domain.domain_name,
                                'dataset': dataset.data_path.name
                            }
                        }
                        
                        self.items.append(item)
                        item_id += 1
                        window_count += 1
        
        print(f"Prepared {len(self.items)} items for evaluation")
    
    def __iter__(self):
        """Iterate over dataset items."""
        return iter(self.items)
    
    def __len__(self):
        """Return number of items."""
        return len(self.items)


def create_predictor_from_model(model_name: str, model_type: str, **kwargs) -> MUSEDFMPredictor:
    """
    Create a MUSED-FM predictor from model name and type.
    
    Args:
        model_name: Name of the model
        model_type: Type of the model (statistical, deep-learning, etc.)
        **kwargs: Additional model parameters
        
    Returns:
        MUSEDFMPredictor instance
    """
    from musedfm.baselines import (
        MeanForecast, 
        HistoricalInertia, 
        ARIMAForecast, 
        LinearTrend, 
        ExponentialSmoothing
    )
    from musedfm.baselines.linear_regression import LinearRegressionForecast
    
    # Map model names to classes
    model_map = {
        "mean": MeanForecast(),
        "historical_inertia": HistoricalInertia(),
        "linear_trend": LinearTrend(),
        "exponential_smoothing": ExponentialSmoothing(),
        "arima": ARIMAForecast(order=(1, 1, 1)),
        "linear_regression": LinearRegressionForecast(),
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_map.keys())}")
    
    model = model_map[model_name]
    univariate = model_name != "linear_regression"  # Only linear_regression is multivariate
    
    return MUSEDFMPredictor(model, model_name, univariate)


def evaluate_with_gift_conventions(predictor: MUSEDFMPredictor, dataset: MUSEDFMDataset, 
                                  season_length: int = 1) -> Dict[str, float]:
    """
    Evaluate the predictor following GIFT-Eval conventions.
    
    Following conventions:
    - Aggregate results over all dimensions (axis=None equivalent)
    - Do not count NaN values in target towards calculation (mask_invalid_label=True equivalent)
    - Make sure prediction does not have NaN values (allow_nan_forecast=False equivalent)
    
    Args:
        predictor: MUSED-FM predictor
        dataset: MUSED-FM dataset
        season_length: Season length for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"Evaluating with GIFT-Eval conventions...")
    print(f"Dataset size: {len(dataset)}")
    
    all_mape = []
    all_mae = []
    all_rmse = []
    all_nmae = []
    
    for item in dataset:
        # Extract data
        history = item['history']
        target = item['target']
        covariates = item.get('covariates', None)
        
        # Generate forecast
        with suppress_all_warnings():
            if predictor.univariate or covariates is None:
                forecast = predictor.model.forecast(history, None, len(target))
            else:
                forecast = predictor.model.forecast(history, covariates, len(target))
        
        # Handle failed forecasts (allow_nan_forecast=False equivalent)
        if forecast is None:
            forecast = np.zeros(len(target))
        
        # Ensure forecast length matches target length
        if len(forecast) != len(target):
            forecast = np.zeros(len(target))
        
        # Ensure no NaN values in forecast (allow_nan_forecast=False equivalent)
        forecast = np.nan_to_num(forecast, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate metrics with NaN handling (mask_invalid_label=True equivalent)
        # Remove NaN values from target before calculation
        valid_mask = ~np.isnan(target)
        if np.any(valid_mask):
            target_clean = target[valid_mask]
            forecast_clean = forecast[valid_mask]
            
            # Calculate metrics only on valid data
            mape_val = MAPE(target_clean, forecast_clean)
            mae_val = MAE(target_clean, forecast_clean)
            rmse_val = RMSE(target_clean, forecast_clean)
            nmae_val = NMAE(target_clean, forecast_clean)
            
            # Only add if metrics are valid (not NaN)
            if not np.isnan(mape_val):
                all_mape.append(mape_val)
            if not np.isnan(mae_val):
                all_mae.append(mae_val)
            if not np.isnan(rmse_val):
                all_rmse.append(rmse_val)
            if not np.isnan(nmae_val):
                all_nmae.append(nmae_val)
    
    # Aggregate results over all dimensions (axis=None equivalent)
    results = {}
    if all_mape:
        results['MAPE'] = np.mean(all_mape)
    else:
        results['MAPE'] = np.nan
        
    if all_mae:
        results['MAE'] = np.mean(all_mae)
    else:
        results['MAE'] = np.nan
        
    if all_rmse:
        results['RMSE'] = np.mean(all_rmse)
    else:
        results['RMSE'] = np.nan
        
    if all_nmae:
        results['NMAE'] = np.mean(all_nmae)
    else:
        results['NMAE'] = np.nan
    
    return results




def generate_forecast_results(predictor: MUSEDFMPredictor, dataset: MUSEDFMDataset, 
                              output_dir: str, model_name: str) -> str:
    """
    Generate forecast results CSV following GIFT-Eval specifications.
    
    Args:
        predictor: MUSED-FM predictor
        dataset: MUSED-FM dataset
        output_dir: Output directory
        model_name: Name of the model
        
    Returns:
        Path to the generated CSV file
    """
    # Create results directory
    results_dir = Path(output_dir) / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating forecast results for {len(dataset)} items...")
    
    # Prepare forecast data
    forecast_data = []
    
    for item in dataset:
        # Extract data
        history = item['history']
        target = item['target']
        covariates = item.get('covariates', None)
        
        # Generate forecast
        with suppress_all_warnings():
            if predictor.univariate or covariates is None:
                forecast = predictor.model.forecast(history, None, len(target))
            else:
                forecast = predictor.model.forecast(history, covariates, len(target))
        
        # Handle failed forecasts
        if forecast is None:
            forecast = np.zeros(len(target))
        
        # Ensure forecast length matches target length
        if len(forecast) != len(target):
            forecast = np.zeros(len(target))
        
        # Ensure no NaN values in forecast
        forecast = np.nan_to_num(forecast, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create forecast entry
        forecast_entry = {
            'item_id': item['item_id'],
            'category': item['info']['category'],
            'domain': item['info']['domain'],
            'dataset': item['info']['dataset'],
            'model': model_name,
            'forecast_length': len(forecast),
            'forecast_values': ','.join([str(x) for x in forecast]),
            'target_values': ','.join([str(x) for x in target]),
            'history_length': len(history),
            'has_covariates': covariates is not None
        }
        
        forecast_data.append(forecast_entry)
    
    # Create DataFrame and save
    df = pd.DataFrame(forecast_data)
    csv_path = results_dir / "all_results.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Forecast results saved to: {csv_path}")
    print(f"Generated {len(forecast_data)} forecast entries")
    
    return str(csv_path)


def generate_config_json(model_name: str, model_type: str, model_dtype: str, 
                        model_link: str, org: str, testdata_leakage: str, 
                        output_dir: str) -> str:
    """
    Generate config.json file following GIFT-Eval specifications.
    
    Args:
        model_name: Name of the model
        model_type: Type of the model
        model_dtype: Data type of the model
        model_link: Link to the model
        org: Organization name
        testdata_leakage: Whether test data leakage occurred
        output_dir: Output directory
        
    Returns:
        Path to the generated JSON file
    """
    # Create results directory
    results_dir = Path(output_dir) / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare config data
    config = {
        "model": model_name,
        "model_type": model_type,
        "model_dtype": model_dtype,
        "model_link": model_link,
        "org": org,
        "testdata_leakage": testdata_leakage
    }
    
    # Save config
    config_path = results_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Config saved to: {config_path}")
    return str(config_path)


def main():
    """Main function for MUSED-FM evaluation."""
    parser = argparse.ArgumentParser(
        description="MUSED-FM Forecast Generation Script for GIFT-Eval Submission",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval_musedfm.py --model-name mean --model-type statistical --benchmark-path /path/to/benchmark
  python eval_musedfm.py --model-name linear_regression --model-type statistical --output-dir /path/to/results
  python eval_musedfm.py --model-name arima --model-type statistical --windows 100 --org "MyOrg"
        """
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model to evaluate (mean, arima, linear_regression, etc.)"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["statistical", "deep-learning", "agentic", "pretrained", "fine-tuned", "zero-shot"],
        help="Type of the model"
    )
    
    parser.add_argument(
        "--benchmark-path",
        type=str,
        default="/home/caleb/musedfm_data",
        help="Path to the benchmark directory containing categories"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/musedfm_results",
        help="Output directory for results (default: /tmp/musedfm_results)"
    )
    
    parser.add_argument(
        "--windows",
        type=int,
        default=None,
        help="Maximum number of windows per dataset (default: None for all windows)"
    )
    
    parser.add_argument(
        "--model-dtype",
        type=str,
        default="float32",
        help="Data type of the model (default: float32)"
    )
    
    parser.add_argument(
        "--model-link",
        type=str,
        default="",
        help="Link to the model (e.g., https://huggingface.co/amazon/chronos-t5-small)"
    )
    
    parser.add_argument(
        "--org",
        type=str,
        default="MUSED-FM",
        help="Organization name (default: MUSED-FM)"
    )
    
    parser.add_argument(
        "--testdata-leakage",
        type=str,
        default="No",
        choices=["Yes", "No"],
        help="Whether test data leakage occurred (default: No)"
    )
    
    parser.add_argument(
        "--season-length",
        type=int,
        default=1,
        help="Season length for evaluation (default: 1)"
    )
    
    parser.add_argument(
        "--history-length",
        type=int,
        default=512,
        help="History length for windows (default: 512)"
    )
    
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=128,
        help="Forecast horizon for windows (default: 128)"
    )
    
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
        help="Stride between windows (default: 256)"
    )
    
    args = parser.parse_args()
    
    print("MUSED-FM Forecast Generation Script for GIFT-Eval Submission")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Model Type: {args.model_type}")
    print(f"Benchmark Path: {args.benchmark_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Max Windows per Dataset: {args.windows}")
    print(f"Generating forecast results for GIFT-Eval evaluation")
    
    start_time = time.time()
    
    try:
        # Load benchmark
        print("\nLoading benchmark...")
        benchmark = Benchmark(
            args.benchmark_path, 
            history_length=args.history_length, 
            forecast_horizon=args.forecast_horizon, 
            stride=args.stride
        )
        print(f"Loaded benchmark with {len(benchmark)} categories")
        
        # Create predictor
        print(f"\nCreating predictor for model: {args.model_name}")
        predictor = create_predictor_from_model(args.model_name, args.model_type)
        
        # Create dataset
        print("\nPreparing dataset...")
        dataset = MUSEDFMDataset(benchmark, max_windows=args.windows)
        
        # Generate forecasts
        print("\nGenerating forecasts...")
        csv_path = generate_forecast_results(predictor, dataset, args.output_dir, args.model_name)
        
        # Generate config JSON
        print("\nGenerating config file...")
        config_path = generate_config_json(
            args.model_name,
            args.model_type,
            args.model_dtype,
            args.model_link,
            args.org,
            args.testdata_leakage,
            args.output_dir
        )
        
        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds")
        print("Forecast generation completed successfully!")
        print(f"\nOutput files:")
        print(f"  Forecast CSV: {csv_path}")
        print(f"  Config JSON: {config_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
