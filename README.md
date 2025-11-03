# MUSEval: Multivariate Time Series Evaluation Dataset for Foundation Models

[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20HF-MUSEval_Dataset-FFD21E)](https://huggingface.co/datasets/Synthefy/MUSEval)
[![leaderboard](https://img.shields.io/badge/%F0%9F%8F%86%20MUSEval-Leaderboard-0078D4)](https://huggingface.co/spaces/Synthefy/MUSEval)

A multivariate-first foundation model evaluation dataset that focuses on identifying a model's ability to leverage other useful time series for forecasting a single target time series. This codebase provides tooling and APIs for iterating through and evaluating metrics on the dataset, performing both univariate (just the target) and multivariate forecasting, and stratifying results according to each dataset, different domains (ex. energy, web, etc.) and different categories (synthetic, traditional, etc.).

# Usage

## Getting and Setting Up the Data:
**Install git-lfs if not preset**
Follow installation instructions here: [git-lfs](https://github.com/git-lfs/git-lfs?utm_source=gitlfs_site&utm_medium=installation_link&utm_campaign=gitlfs#installing)

**Clone the [dataset](https://huggingface.co/datasets/Synthefy/MUSEval) repo (requires git-lfs, ~100GB):**
```bash
git clone git@hf.co:datasets/Synthefy/MUSEval ./museval-data
```

**Create the directory structure and extract to target directory**
```bash
# Create directory structure in shared memory for faster I/O
mkdir -p <your target folder>/museval-nested/{collections,traditional,sequential,synthetic}

# Extract archives to shared memory
tar -xzf ./museval-data/compressed_data/collections.tar.gz -C <your target folder>/museval-nested/collections &
tar -xzf ./museval-data/compressed_data/traditional.tar.gz -C <your target folder>/museval-nested/traditional &
tar -xzf ./museval-data/compressed_data/sequential.tar.gz -C <your target folder>/museval-nested/sequential &
tar -xzf ./museval-data/compressed_data/synthetic.tar.gz -C <your target folder>/museval-nested/synthetic &
```

---

## Installation

### Prerequisites
- Python 3.11 or higher
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

### Install MUSEval
```bash
# Install in development mode
uv pip install -e .
```

---

### Running the Benchmark

MUSEval provides two main evaluation scripts for different use cases:

#### 1. `run_museval.py` - Main Evaluation Script

The primary script for running baseline models with visualization and CSV export capabilities:

```bash
# Run all models on all datasets with plots and CSV export and save exports
uv run src/examples/run_museval.py --benchmark-path <your target folder>/museval-nested --models all --plots --load-cached-counts --output-dir <your target folder>/museval-outputs/results --forecast-save-path <your target folder>/museval-outputs/forecasts --batch-size 8

# Debug mode (limited windows)
uv run src/examples/run_museval.py --benchmark-path <your target folder>/museval-nested --models all --load-cached-counts --plots --output-dir <your target folder>/results --forecast-save-path <your target folder>/forecasts --windows 200 --debug-mode --batch-size 8
```

**Key Arguments:**
- Generates forecast plots (`--plots`)
- Supports debug mode (`--debug-mode`)
- Uses cached window counts for faster startup (`--load-cached-counts`)
- Flexible filtering by categories, domains, and datasets

#### 2. `eval_museval.py` - Evaluation Script with Forecast Saving

Specialized script for just forecast evaluation:

```bash
# Use cached window counts for faster execution
uv run src/examples/eval_museval.py --benchmark-path <your target folder>museval-nested --models all --load-cached-counts --output-dir <your target folder>/results
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
| `--benchmark-path` | Path to benchmark directory | `../museval_data` |
| `--models` | Comma-separated list of models or 'all' | `mean,linear_trend` |
| `--categories` | Filter by categories (e.g., 'traditional,synthetic') | All |
| `--domains` | Filter by domains (e.g., 'Energy,Finance') | All |
| `--datasets` | Filter by datasets (e.g., 'al_daily,bitcoin_price') | All |
| `--windows` | Maximum windows per dataset | None (all) |
| `--history-length` | History length for windows | 512 |
| `--forecast-horizon` | Forecast horizon | 128 |
| `--stride` | Stride between windows | 256 |
| `--load-cached-counts` | Use cached window counts | False |

**run_museval.py specific arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--plots` | Generate forecast plots | False |
| `--output-dir` | Output directory for plots/CSV | `/tmp` |
| `--debug-mode` | Enable debug mode | False |

**eval_museval.py specific arguments:**
| Argument | Description | Default |
|----------|-------------|---------|
| `--forecast-save-path` | Path to save forecasts | `/tmp` |
| `--chunk-size` | Chunk size for saving | 1048576 |

## Adding Custom Models

The above commands will only run the existing baselines. The dataset uses NaN padding for missing values, which should be handled by the custom model file. To use the tools in this repo to add your own model, you first need to create a model class with the desired features. Then using an instance of that class, there are three ways to add a new model, which we discuss below:
- Add your model to the model_handling.py file
- Utilize the functions in run_museval.py to evaluate your model
- Use the iterator to iterate through the dataset directly

### Creating A model Class
## Adding Custom Models
```python
from examples.run_museval import run_models_on_benchmark, compare_model_performance
from examples.utils import parse_models
from museval.baselines.base_forecaster import BaseForecaster
import numpy as np
from typing import Optional

# 1. Create your custom model class
class MyCustomModel(BaseForecaster):
    def __init__(self, **kwargs):
    
    def forecast(self, history: np.ndarray, covariates: Optional[np.ndarray] = None, 
                forecast_horizon: Optional[int] = None, timestamps: Optional[np.ndarray]) -> np.ndarray:
        """
        Your custom forecasting logic here.
        
        Args:
            history: Historical time series data (shape: [batch_size, time_steps])
            covariates: Optional covariate data (shape: [batch_size, time_steps, covariate_features])
            forecast_horizon: Number of future points to forecast
            timestamps: Timestamps for full window
            
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
```


### Usage pattern 1: Modify model_handling with your custom model

Instead of modifying the `run_museval.py` file, you can use the evaluation functions directly with your custom models by updating the models file in examples/model_handling.py:

**Create model dictionary for evaluation**
```python
def get_custom_models():
    return {
        "MY_MODEL": {"model": MyCustomModel(param1=0.1, param2=1.2), "univariate": False} # set univariate to true if your model is only univariate
    }
```

Then simply run 
```bash
uv run src/examples/run_museval.py --benchmark-path <YOUR PATH>/museval-nested --models <MY_MODEL> --load-cached-counts --output-dir <your output path>/museval-outputs/ --debug-mode --stride 512 --batch-size 32
```

### Usage Pattern 2: Use functions from run_museval.py
Using the function from run_museval can give more fine-grain control, especially if you intend to change how the results are used or visualized.

```python
# Import functions from the examples package
from examples import (
    run_models_on_benchmark, 
    compare_model_performance, 
    export_hierarchical_results_to_csv,
    generate_forecast_plots,
    save_submission
)

# Configuration, modify to values appropriate for your model
BENCHMARK_PATH = "<target path>museval-nested/"  # Adjust this path to your MUSEval data
MAX_WINDOWS = 3  # Limit windows per dataset for faster testing
OUTPUT_DIR = ".<output target path>/<your model name>"
BATCH_SIZE = 1

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

models = {
    "<your model name>": {"model": <Your Model Class>(), "univariate": True/False},
}

results = run_models_on_benchmark(
    benchmark_path=BENCHMARK_PATH,
    models=models,
    max_windows=MAX_WINDOWS,
    history_length=512,
    forecast_horizon=128,
    stride=512,
    load_cached_counts=True,
    collect_plot_data=True  # Enable plot data collection
    batch_size=BATCH_SIZE
)

compare_model_performance(results)

generate_forecast_plots(results, OUTPUT_DIR, limit_windows=10)

export_hierarchical_results_to_csv(results, output_dir=OUTPUT_DIR)

save_submission(results, OUTPUT_DIR, '<your model name>')

```

### Usage 3: Programmatic Usage
Directly iterate on the dataset, which is especially useful when you want fine-grained manipulation of the data. 

```python
from museval.data import Benchmark
from museval.baselines.base_forecaster import BaseForecaster
import numpy as np
from typing import Optional

# Load benchmark data
benchmark = Benchmark("<PATH TO DATA>/museval-nested")

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
                    len(window.target()),
                    window.timestamps()
                )
                
                # Submit forecast for evaluation (automatically runs evaluation code)
                # Set univariate=True if your model only uses target series
                # Set univariate=False if your model uses covariates
                window.submit_forecast(forecast, univariate=True)
            
            # Get evaluation results for this dataset
            dataset_results = dataset.evaluate()
            print(f"    Dataset results: {dataset_results}")

# then convert your results into the submission format (see save_submission)
```

---

## Example Notebooks

MUSEval includes several Jupyter notebooks demonstrating different evaluation approaches:

### 1. Baseline Models Evaluation
**File:** [`notebooks/baseline_models_museval_run_eval.ipynb`](notebooks/baseline_models_museval_run_eval.ipynb)

### 2. Chronos Bolt Model Evaluation (using run_museval)
**File:** [`notebooks/chronos_bolt_museval_run_eval.ipynb`](notebooks/chronos_bolt_museval_run_eval.ipynb)

### 3. Custom Chronos Evaluation (directly iterating on the dataset)
**File:** [`notebooks/chronos_bolt_museval_custom_eval.ipynb`](notebooks/chronos_bolt_museval_custom_eval.ipynb)

---

### run_museval Output Directory Structure

```
/tmp/results/
├── results.csv                    # Consolidated results
├── hierarchical_results.csv        # Category/domain/dataset breakdown
├── plots/                         # Forecast visualizations
│   ├── mean_forecasts.png
│   ├── linear_trend_forecasts.png
│   └── arima_forecasts.png
├── submissions/                   # Competition submission files
│   ├── YOUR_MODEL_submission.json
│   ├── YOUR_SECOND_MODEL_submission.json # (if you have multiple submissions)
│   └── ...
└── debug/                         # Debug information (if --debug-mode)
    ├── high_mape_windows.png
    └── model_performance_analysis.png
```

# Make a submission

Submit data for our [leaderboard](https://huggingface.co/spaces/Synthefy/MUSEval) as a JSON array containing entries for each dataset with the following format:

```json
[
  {
    "domain": "Stock",
    "category": "collections",
    "dataset": "stock_nasdaqtrader",
    "dataset_version": "v1.0",
    "metrics": {
      "MAE": 995.1785190528401,
      "RMSE": 995.3778166659135,
      "Multi-MAPE": 1830.1983981093183,
      "Uni-MAPE": 1830.1983981093183,
      "NMAE": 2.2736757287592733
    }
  },
]
```

**Required fields:**
- `domain`: The domain of the dataset (e.g., "Energy", "Finance", "Stock")
- `category`: The category of the dataset (e.g., "traditional", "collections", "sequential", "synthetic")
- `dataset`: The name of the dataset
- `dataset_version`: The version of the dataset (currently "v1.0")
- `metrics`: Object containing the evaluation metrics:
  - `MAE`: Mean Absolute Error
  - `RMSE`: Root Mean Square Error
  - `Multi-MAPE`: Multivariate Mean Absolute Percentage Error
  - `Uni-MAPE`: Univariate Mean Absolute Percentage Error
  - `NMAE`: Normalized Mean Absolute Error

**Notes:**
- The submission files are automatically generated by the `save_submission()` function (called by run_museval.py)
- Files are saved as `{model_name}_submission.json` in the `submissions/` directory
- Example files are provided in `src/examples/eval_museval` and `src/examples/run_museval`

## Internal Overview

The `museval` iterator provides iteration and evaluation utilities for the **MUSEval multivariate timeseries evaluation dataset**. Users download the dataset from Hugging Face as a `.tar.gz`, extract it, and then interact with it via this package.

### Data Structure
- **Benchmark** → container of multiple categories  
- **Category** → contains multiple domains (traditional, collections, sequential, synthetic)
- **Domain** → contains multiple datasets  
- **Dataset** → contains multiple parquet files  
- **Window** → extracted slices from parquet files  

The breakdown of domains, categories and datasets can be found in `src/museval/data/data_hierarchy.json`

---

## Directory Layout
```
MUSEval/
├── src/museval/              # Core MultiTS-eval package
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
│   ├── run_museval.py        # Main evaluation script
│   ├── eval_museval.py       # Evaluation script with forecast saving
│   ├── utils.py              # Utility functions
│   ├── debug.py              # Debug utilities
│   ├── save_submission.py    # Saves out putputs in submission format
│   └── export_csvs.py        # CSV export utilities
├── notebooks/                # Example usage Jupyter notebooks
│   ├── baseline_models_museval_run_eval.ipynb      # Baseline model evaluation
│   ├── chronos_bolt_museval_run_eval.ipynb         # Chronos Bolt model evaluation
│   └── chronos_bolt_museval_custom_eval.ipynb      # Custom Chronos evaluation
├── README.md                 # This file
├── pyproject.toml            # Project configuration
└── setup.py                  # Package setup
```

---

## Internal Classes

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
- Handles batching for faster iteration
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
- Simple orchestrator for evaluations.

**Methods**
- `__iter__() -> Iterator[Category]`
- `evaluate() -> dict`  
  Aggregate results across categories (cached results only).
- `to_results_csv(path: str) -> None`  
  Validate forecasts, gather results from all categories, write consolidated CSV.

---

## Baseline Models

The framework includes several baseline forecasting methods:

1. **Mean Forecast** (`mean`) - Forecasts the mean of historical data
2. **Historical Inertia** (`historical_inertia`) - Uses the last observed value
3. **Linear Trend** (`linear_trend`) - Linear trend extrapolation
4. **Exponential Smoothing** (`exponential_smoothing`) - Simple exponential smoothing
5. **ARIMA** (`arima`) - AutoRegressive Integrated Moving Average model
6. **Linear Regression** (`linear_regression`) - Linear regression with univariate/multivariate capabilities
7. **Chronos Bolt** (`chronos`) - [Chronos bolt](https://github.com/amazon-science/chronos-forecasting) by Amazon, a TSFM.

Most baseline models are **univariate** (use only target series). The **Linear Regression** model can operate in both univariate and multivariate modes.

Additional models ([toto](https://github.com/DataDog/toto), [timesfm](https://github.com/google-research/timesfm/), [moirai](https://github.com/redoules/moirai)) are supported but require additional install.

---

## Metrics

Defined in `src/museval/metrics.py`:
- **MAPE** - Mean Absolute Percentage Error
- **MAE** - Mean Absolute Error  
- **RMSE** - Root Mean Square Error
- **NMAE** - Normalized Mean Absolute Error

All metrics include robust NaN handling and return appropriate warnings when insufficient data is available.

---

