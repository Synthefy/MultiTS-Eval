# MUSED-FM

A comprehensive multivariate time series forecasting evaluation framework with robust data handling, multiple baseline models, and flexible model integration.

## Overview

The `mused-fm` client provides iteration and evaluation utilities for the **MUSED-FM multivariate timeseries evaluation dataset**. Users download the dataset from Hugging Face as a `.tar.gz`, extract it, and then interact with it via this package.

### Data Structure
- **Benchmark** → container of multiple categories  
- **Category** → contains multiple domains (traditional, collections, sequential, synthetic)
- **Domain** → contains multiple datasets  
- **Dataset** → contains multiple parquet files  
- **Window** → extracted slices from parquet files  

The breakdown of domains, categories and datasets can be found in `src/musedfm/data/data_hierarchy.json`

---

## Directory Layout
```
MUSED-FM/
├── src/musedfm/              # Core MUSED-FM package
│   ├── data/                 # Data handling components
│   │   ├── window.py         # Individual forecasting windows
│   │   ├── dataset.py        # Dataset loading and windowing
│   │   ├── domain.py         # Domain management
│   │   ├── category.py       # Category management
│   │   ├── benchmark.py      # Top-level benchmark container
│   │   ├── dataset_config.json    # Dataset configuration
│   │   ├── data_hierarchy.json    # Data organization hierarchy
│   │   └── special_loaders/  # Custom data loaders
│   │       ├── __init__.py
│   │       └── open_aq_special.py # OpenAQ air quality data loader
│   ├── baselines/            # Baseline forecasting models
│   │   ├── mean_forecast.py
│   │   ├── historical_inertia.py
│   │   ├── linear_trend.py
│   │   ├── exponential_smoothing.py
│   │   └── arima_forecast.py
│   ├── metrics.py            # Evaluation metrics
│   └── plotting/             # Visualization utilities
│       └── plot_forecasts.py
├── src/examples/             # Example scripts and utilities
│   ├── run_musedfm.py        # Main evaluation script
│   ├── eval_musedfm.py       # Evaluation script with forecast saving
│   ├── utils.py              # Utility functions
│   ├── debug.py              # Debug utilities
│   └── export_csvs.py        # CSV export utilities
├── notebooks/                # Jupyter notebooks
│   ├── baseline_models_musedfm_run_eval.ipynb      # Baseline model evaluation
│   ├── chronos_bolt_musedfm_run_eval.ipynb         # Chronos Bolt model evaluation
│   └── chronos_bolt_musedfm_custom_eval.ipynb      # Custom Chronos evaluation
├── README.md                 # This file
├── pyproject.toml            # Project configuration
└── setup.py                  # Package setup
```

---

## Core Components

### 1. Window
- **Represents a single evaluation unit** (history, target, covariates).
- Stores ground truth and submitted forecast with univariate/multivariate classification.

**Methods**
- `history() -> np.ndarray` - Historical time series data
- `target() -> np.ndarray` - Ground truth forecast targets
- `covariates() -> np.ndarray` - Additional features for multivariate models
- `submit_forecast(forecast: np.ndarray, univariate: bool = False) -> None`  
  Stores forecast and triggers evaluation once for this window.
- `evaluate() -> dict`  
  Runs metrics (MAPE, MAE, RMSE, NMAE) and returns cached results with univariate flag.
- `is_univariate -> bool` - Check if forecast is univariate

---

### 2. Dataset
- **Collection of windows** (parsed from parquet files).
- Handles data cleaning, NaN removal, and column validation.
- Iterable: `for window in dataset`

**Methods**
- `__iter__() -> Iterator[Window]`
- `evaluate() -> dict`  
  Runs evaluation for all windows (delegates to their cached results).
- `to_results_csv(path: str) -> None`  
  Validates all windows have forecasts submitted, collects cached results, computes aggregates, saves CSV.

---

### 3. Domain
- **Encapsulates multiple datasets**.
- Iterable: `for dataset in domain`

**Methods**
- `__iter__() -> Iterator[Dataset]`
- `evaluate() -> dict`  
  Runs evaluation across all datasets (cached results only).

---

### 4. Category
- **Groups datasets with similar data properties**.
- Four categories: traditional, collections, sequential, synthetic.
- Iterable: `for domain in category`

**Methods**
- `__iter__() -> Iterator[Domain]`
- `evaluate() -> dict`  
  Runs evaluation across all domains in the category.

---

### 5. Benchmark
- **Top-level container for multiple categories**.
- Simple orchestrator for evaluations and exports.

**Methods**
- `__iter__() -> Iterator[Category]`
- `evaluate() -> dict`  
  Aggregate results across categories (cached results only).
- `to_results_csv(path: str) -> None`  
  Validate forecasts, gather results from all categories, write consolidated CSV.

---

## Baseline Models

The framework includes several baseline forecasting methods:

1. **Mean Forecast** (`MeanForecast`) - Forecasts the mean of historical data
2. **Historical Inertia** (`HistoricalInertia`) - Uses the last observed value
3. **Linear Trend** (`LinearTrend`) - Linear trend extrapolation
4. **Exponential Smoothing** (`ExponentialSmoothing`) - Simple exponential smoothing
5. **ARIMA** (`ARIMAForecast`) - AutoRegressive Integrated Moving Average model
6. **Linear Regression** (`LinearRegressionForecast`) - Linear regression with univariate/multivariate capabilities

Most baseline models are **univariate** (use only target series). The **Linear Regression** model can operate in both univariate and multivariate modes.

---

## Metrics

Defined in `src/musedfm/metrics.py`:
- **MAPE** - Mean Absolute Percentage Error
- **MAE** - Mean Absolute Error  
- **RMSE** - Root Mean Square Error
- **NMAE** - Normalized Mean Absolute Error

All metrics include robust NaN handling and return appropriate warnings when insufficient data is available.

---

## Usage

## Getting and Setting Up the Data:

**Clone the dataset repo:**
```bash
git clone git@hf.co:datasets/Synthefy/MUSED-FM ./mused-fm-data
```

**Follow the instructions in the repo for unzipping the files**

**Create the directory structure and extract to target directory**
```bash
# Create directory structure in shared memory for faster I/O
mkdir -p ~/mused-fm-nested/{collections,traditional,sequential,synthetic}

# Extract archives to shared memory
tar -xzf ./mused-fm-data/collections.tar.gz -C ~/mused-fm-nested &
tar -xzf ./mused-fm-data/traditional.tar.gz -C ~/mused-fm-nested/traditional &
tar -xzf ./mused-fm-data/sequential.tar.gz -C ~/mused-fm-nested/sequential &
tar -xzf ./mused-fm-data/synthetic.tar.gz -C ~/mused-fm-nested/synthetic &
```


### Running the Benchmark

MUSED-FM provides two main evaluation scripts for different use cases:

#### 1. `run_musedfm.py` - Main Evaluation Script

The primary script for running baseline models with visualization and CSV export capabilities:

**Add your model to model_handling**
```python
def get_available_models():
  ...
    return {
      ...
      "your model here": {"model": YourModelClass(), "univariate": False}
    }
```

```bash
# Run all models on all datasets with plots and CSV export and save exports
uv run src/examples/run_musedfm.py --benchmark-path ~/~mused-fm-nested --models all --plots --load-cached-counts --output-dir ~/mused-fm-outputs/results --forecast-save-path ~/mused-fm-outputs/forecasts

# Debug mode
uv run src/examples/run_musedfm.py --benchmark-path /dev/shm/data/mused-fm-nested --models all --debug-mode --load-cached-counts --plots --output-dir /tmp/results --forecast-save-path /tmp/forecasts
```

**Key Features:**
- Generates forecast plots (`--plots`)
- Supports debug mode (`--debug-mode`)
- Uses cached window counts for faster startup (`--load-cached-counts`)
- Flexible filtering by categories, domains, and datasets

#### 2. `eval_musedfm.py` - Evaluation Script with Forecast Saving

Specialized script for just forecast evaluation:

```bash
# Run all models with forecast saving
uv run src/examples/eval_musedfm.py --benchmark-path /dev/shm/data/mused-fm-nested --models all --forecast-save-path /tmp/forecasts

# Run specific models with limited windows
uv run src/examples/eval_musedfm.py --benchmark-path /dev/shm/data/mused-fm-nested --models mean,arima,linear_trend --windows 50 --forecast-save-path /tmp/forecasts

# Run on specific categories with custom parameters
uv run src/examples/eval_musedfm.py --benchmark-path /dev/shm/data/mused-fm-nested --models all --categories traditional --history-length 512 --forecast-horizon 128 --stride 256 --forecast-save-path /tmp/forecasts

# Use cached window counts for faster execution
uv run src/examples/eval_musedfm.py --benchmark-path /dev/shm/data/mused-fm-nested --models all --load-cached-counts --forecast-save-path /tmp/forecasts
```

**Key Features:**
- Saves forecasts to disk for later analysis (`--forecast-save-path`)
- Supports chunked saving for large datasets (`--chunk-size`)
- Optimized for batch processing and evaluation
- Compatible with forecast loading utilities

#### Command Line Arguments

Both scripts support the following common arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--benchmark-path` | Path to benchmark directory | `/home/caleb/musedfm_data` |
| `--models` | Comma-separated list of models or 'all' | `mean,linear_trend` |
| `--categories` | Filter by categories (e.g., 'traditional,synthetic') | All |
| `--domains` | Filter by domains (e.g., 'Energy,Finance') | All |
| `--datasets` | Filter by datasets (e.g., 'al_daily,bitcoin_price') | All |
| `--windows` | Maximum windows per dataset | None (all) |
| `--history-length` | History length for windows | 512 |
| `--forecast-horizon` | Forecast horizon | 128 |
| `--stride` | Stride between windows | 256 |
| `--load-cached-counts` | Use cached window counts | False |

**run_musedfm.py specific arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--plots` | Generate forecast plots | False |
| `--output-dir` | Output directory for plots/CSV | `/tmp` |
| `--debug-mode` | Enable debug mode | False |

**eval_musedfm.py specific arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--forecast-save-path` | Path to save forecasts | `/tmp` |
| `--chunk-size` | Chunk size for saving | 1048576 |

---

## Example Notebooks

MUSED-FM includes several Jupyter notebooks demonstrating different evaluation approaches:

### 1. Baseline Models Evaluation
**File:** [`notebooks/baseline_models_musedfm_run_eval.ipynb`](notebooks/baseline_models_musedfm_run_eval.ipynb)

Comprehensive evaluation of all baseline models using the `run_musedfm.py` functions:
- Demonstrates model comparison across different categories
- Shows plotting and CSV export capabilities
- Includes performance analysis and debugging utilities

### 2. Chronos Bolt Model Evaluation
**File:** [`notebooks/chronos_bolt_musedfm_run_eval.ipynb`](notebooks/chronos_bolt_musedfm_run_eval.ipynb)

Evaluation of the Chronos Bolt forecasting model on MUSED-FM:
- Self-contained ChronosForecast class implementation
- Integration with MUSED-FM evaluation framework
- Comparison with baseline models

### 3. Custom Chronos Evaluation
**File:** [`notebooks/chronos_bolt_musedfm_custom_eval.ipynb`](notebooks/chronos_bolt_musedfm_custom_eval.ipynb)

Custom evaluation approach for Chronos models:
- Advanced model configuration and tuning
- Custom evaluation metrics and analysis
- Specialized forecast handling

**Running Notebooks:**
```bash
# Start Jupyter Lab
uv run jupyter lab

# Or start Jupyter Notebook
uv run jupyter notebook

# Navigate to the notebooks/ directory and open the desired notebook
```

---

## Adding Custom Models

### Using Evaluation Functions Directly

Instead of modifying the `run_musedfm.py` file, you can use the evaluation functions directly with your custom models:

```python
from examples.run_musedfm import run_models_on_benchmark, compare_model_performance
from examples.utils import parse_models
from musedfm.baselines.base_forecaster import BaseForecaster
import numpy as np
from typing import Optional

# 1. Create your custom model class
class MyCustomModel(BaseForecaster):
    def __init__(self, param1=1.0, param2=2.0):
        self.param1 = param1
        self.param2 = param2
    
    def forecast(self, history: np.ndarray, covariates: Optional[np.ndarray] = None, 
                forecast_horizon: Optional[int] = None) -> np.ndarray:
        """
        Your custom forecasting logic here.
        
        Args:
            history: Historical time series data (shape: [time_steps, features])
            covariates: Optional covariate data (shape: [time_steps, covariate_features])
            forecast_horizon: Number of future points to forecast
            
        Returns:
            Forecast values (shape: [forecast_horizon, features])
        """        
        # Handle univariate and multivariate
        if covariates is None:
            # Univariate case
            pass
        else:
            # Multivariate case
            pass
        
        return forecast

# 2. Create model dictionary for evaluation
def get_custom_models():
    return {
        "MY_MODEL": {"model": MyCustomModel(param1=0.1, param2=1.2), "univariate": False} # set univariate to true if your model is only univariate
    }

# 3. Run evaluation using the framework functions
if __name__ == "__main__":
    # Parse models (you can mix custom and built-in models)
    models = get_custom_models()  # or "parse_models() if modifying model_handling.py"
    
    # Run evaluation
    results = run_models_on_benchmark(
        benchmark_path="/dev/shm/data/mused-fm-nested", # modify to your data path
        models=models,
        max_windows=None,  # Limit for testing
        categories=None,  # Filter to specific categories
        domains=None,  # All domains
        datasets=None,  # All datasets
        history_length=512, # stick to default history and forecast lengths
        forecast_horizon=128,
        stride=256,
        load_cached_counts=True,
        collect_plot_data=True
    )
    
    # Compare model performance
    compare_model_performance(results)
    
    # Export results
    from examples.export_csvs import export_hierarchical_results_to_csv
    export_hierarchical_results_to_csv(results, "/tmp/custom_model_results.csv")
    
    print("Custom model evaluation completed!")
```

### Key Benefits of This Approach:

- **No file modification**: Use existing evaluation framework without changing source code
- **Flexible**: Mix custom models with built-in models
- **Reusable**: Easy to create multiple custom models
- **Well-tested**: Leverages the robust evaluation infrastructure
- **Complete**: Includes performance comparison and result export

### Model Classification

- **Univariate models** (`"univariate": True`): Use only the target time series for forecasting
- **Multivariate models** (`"univariate": False`): Use both target series and covariates for forecasting

### Programmatic Usage

```python
from musedfm.data import Benchmark
from musedfm.baselines.base_forecaster import BaseForecaster
import numpy as np
from typing import Optional

# Template for creating custom forecasting models
class MyCustomForecaster(BaseForecaster):
    """
    Template for creating custom forecasting models.
    Inherit from BaseForecaster and implement the forecast method.
    """
    
    def __init__(self, param1=1.0, param2=2.0):
        """
        Initialize your custom model with any parameters.
        
        Args:
            param1: First parameter for your model
            param2: Second parameter for your model
        """
        self.param1 = param1
        self.param2 = param2
    
    def forecast(self, history: np.ndarray, covariates: Optional[np.ndarray] = None, 
                forecast_horizon: Optional[int] = None) -> np.ndarray:
        """
        Generate forecast from historical data.
        
        Args:
            history: Historical time series data (shape: [time_steps, features])
            covariates: Optional covariate data (shape: [time_steps, covariate_features])
            forecast_horizon: Number of future points to forecast
            
        Returns:
            Forecast values (shape: [forecast_horizon, features])
        """
        if forecast_horizon is None:
            forecast_horizon = 1
        
        # Example: Simple moving average with custom parameters
        # Replace this with your actual forecasting logic
        if len(history.shape) == 1:
            # Univariate case
            forecast = np.full(forecast_horizon, np.mean(history) * self.param1)
        else:
            # Multivariate case
            forecast = np.full((forecast_horizon, history.shape[1]), 
                             np.mean(history, axis=0) * self.param1)
        
        return forecast

# Load benchmark data
benchmark = Benchmark("/dev/shm/data/mused-fm-nested")

# Create your custom model
custom_model = MyCustomForecaster(param1=1.5, param2=2.0)

# Evaluate model programmatically
for category in benchmark:
    print(f"Processing category: {category.name}")
    
    for domain in category:
        print(f"  Processing domain: {domain.name}")
        
        for dataset in domain:
            print(f"    Processing dataset: {dataset.name}")
            
            # Process each window in the dataset
            for window in dataset:
                # Generate forecast using your custom model
                forecast = custom_model.forecast(
                    window.history(), 
                    window.covariates(), 
                    len(window.target())
                )
                
                # Submit forecast for evaluation
                # Set univariate=True if your model only uses target series
                # Set univariate=False if your model uses covariates
                window.submit_forecast(forecast, univariate=True)
            
            # Get evaluation results for this dataset
            dataset_results = dataset.evaluate()
            print(f"    Dataset results: {dataset_results}")

# Export consolidated results
benchmark.to_results_csv("custom_model_results.csv")
print("Results exported to custom_model_results.csv")
```

---

## Installation

### Prerequisites
- Python 3.9 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Install Dependencies
```bash
# install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
# create virtual env
uv venv
# Activate virtual environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows
# Install project dependencies
uv sync

```

### Install MUSED-FM
```bash
# Install in development mode
uv pip install -e .
```

---

## Data Handling Features

### Robust Data Processing
- **Automatic NaN handling**: Removes entirely NaN columns and trailing/preceding NaNs
- **Comma-separated number parsing**: Handles numeric data with comma separators
- **Column validation**: Updates target/metadata columns when columns are dropped
- **Special loaders**: Custom data loaders for unique formats (e.g., OpenAQ)

### Dataset Configuration
- **Flexible column specification**: Supports `INDEX#`, `DYNAMIC:`, `CONTAINS:` formats
- **Timestamp handling**: Automatic timestamp generation and datetime column combination
- **Metadata parsing**: Intelligent parsing of metadata column specifications

---

## Output

The framework generates comprehensive evaluation results including:

### Console Output
- **Real-time progress**: Shows processing status for categories, domains, datasets, and windows
- **Model performance summaries**: Detailed metrics for each model after completion
- **Debug information**: Optional debug mode provides detailed analysis of model performance
- **Execution timing**: Total time and per-model timing statistics

### Generated Files

#### Plot Files (when using `--plots` flag)
- **Forecast visualizations**: Actual vs predicted plots for selected windows
- **Performance analysis**: Error distribution and trend analysis
- **Model comparison plots**: Visual comparison of different model outputs

#### Forecast Files (when using `eval_musedfm.py` or passing `--forecast-save-path` to run_musedfm.py)
- **Saved forecasts**: Raw forecast data in parquet format for later analysis
- **Metadata**: Model parameters and evaluation settings
- **Chunked storage**: Efficient storage for large-scale evaluations

### Example Console Output

```
MUSED-FM Example: Multiple Model Forecasting
============================================================
Models: mean,linear_trend,arima
Benchmark path: /dev/shm/data/mused-fm-nested
Categories: All
Domains: All
Datasets: All
Max windows per dataset: 100
Generate plots: True
Export CSV: True
Output directory: /tmp/results

Processing category: traditional
  Processing domain: Energy
    Processing dataset: electricity_demand
      Windows processed: 100/100
    Processing dataset: solar_power
      Windows processed: 100/100
  Processing domain: Finance
    Processing dataset: stock_prices
      Windows processed: 100/100

mean Summary:
  Total windows: 300
  Total time: 2.45s
  Model type: Univariate
  Average MAPE: 15.23%
  Average MAE: 0.0456
  Average RMSE: 0.0678
  Average NMAE: 0.1234

linear_trend Summary:
  Total windows: 300
  Total time: 3.12s
  Model type: Univariate
  Average MAPE: 12.87%
  Average MAE: 0.0389
  Average RMSE: 0.0543
  Average NMAE: 0.0987

arima Summary:
  Total windows: 300
  Total time: 8.45s
  Model type: Univariate
  Average MAPE: 11.23%
  Average MAE: 0.0321
  Average RMSE: 0.0489
  Average NMAE: 0.0876

Best Performance: arima (MAPE: 11.23%)

Total execution time: 14.02 seconds
Results exported to /tmp/results/
Plots saved to /tmp/results/plots/
```

### Output Directory Structure

```
/tmp/results/
├── results.csv                    # Consolidated results
├── hierarchical_results.csv        # Category/domain/dataset breakdown
├── plots/                         # Forecast visualizations
│   ├── mean_forecasts.png
│   ├── linear_trend_forecasts.png
│   └── arima_forecasts.png
└── debug/                         # Debug information (if --debug-mode)
    ├── high_mape_windows.png
    └── model_performance_analysis.png
```

# Make a submission

Submit data as a folder with the following structure:

```
<model_name>/
├── <collection>/
│   └── <domain>/
│       └── <dataset>/
│           ├── {model}_s{stride}_w{history_length}_f{forecast_horizon}_b{batch_idx}.parquet
│           └── ...
└── metadata.json
```

**File naming convention:**
- Parquet files: `{model}_s{stride}_w{history_length}_f{forecast_horizon}_b{batch_idx}.parquet`
- Use `batch_idx` to order files if there are multiple files

**Metadata.json contents:**
```json
{
    "model": "YOUR_MODEL_NAME",
    "model_type": "one of statistical, deep-learning, agentic, pretrained, fine-tuned or zero-shot",
    "model_dtype": "float32, etc.",
    "model_link": "To your HF model link, e.g., https://huggingface.co/amazon/chronos-t5-small",
    "org": "YOUR_ORG_NAME",
    "testdata_leakage": "one of Yes or No",
    "history_length": 512,
    "forecast_horizon": 128,
    "stride": 256
}
```

**Notes:**
- `model`, `stride`, `history_length`, `forecast_horizon` will be used to identify the parquet files
- Example files are provided in `src/examples/eval_musedfm` and `src/examples/run_musedfm`