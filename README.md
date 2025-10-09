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
src/musedfm/
    data/
        window.py              # Individual forecasting windows
        dataset.py             # Dataset loading and windowing
        domain.py              # Domain management
        category.py            # Category management
        benchmark.py           # Top-level benchmark container
        dataset_config.json    # Dataset configuration
        data_hierarchy.json    # Data organization hierarchy
        special_loaders/       # Custom data loaders
            __init__.py
            open_aq_special.py # OpenAQ air quality data loader
    baselines/                 # Baseline forecasting models
        mean_forecast.py
        historical_inertia.py
        linear_trend.py
        exponential_smoothing.py
        arima_forecast.py
    metrics.py                 # Evaluation metrics
    plotting/                  # Visualization utilities
        plot_forecasts.py
examples/
    run_musedfm.py            # Main evaluation script
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

### Running the Benchmark

Use the main evaluation script to run all baseline models:

```bash
# Run all models on all datasets
uv run examples/run_musedfm.py --models all --plots --output-dir /path/to/results --benchmark-path /path/to/data

# Run specific models
uv run examples/run_musedfm.py --models mean,arima,linear_regression --output-dir /path/to/results --benchmark-path /path/to/data

# Limit windows per dataset for faster testing
uv run examples/run_musedfm.py --models all --windows 100 --output-dir /path/to/results --benchmark-path /path/to/data

# Run only the linear regression model (generates both univariate and multivariate forecasts)
uv run examples/run_musedfm.py --models linear_regression --output-dir /path/to/results --benchmark-path /path/to/data
```

### Utilizing examples/run_musedfm.py as a template

To add a custom forecasting model:

1. **Create your model class**:
```python
class MyCustomModel:
    def __init__(self, param1=1.0, param2=2.0):
        self.param1 = param1
        self.param2 = param2
    
    def forecast(self, history, covariates, forecast_length):
        # Your custom forecasting logic here
        # history: numpy array of historical values
        # covariates: numpy array of covariate values (can be None)
        # forecast_length: number of steps to forecast
        return np.array([np.mean(history) * self.param1] * forecast_length)
```

2. **Add to the model dictionary** in `examples/run_musedfm.py`:
```python
def get_available_models():
    return {
        "mean": {"model": MeanForecast(), "univariate": True},
        "historical_inertia": {"model": HistoricalInertia(), "univariate": True},
        "linear_trend": {"model": LinearTrend(), "univariate": True},
        "exponential_smoothing": {"model": ExponentialSmoothing(), "univariate": True},
        "arima": {"model": ARIMAForecast(order=(1, 1, 1)), "univariate": True},
        # Add your custom model:
        "my_custom": {"model": MyCustomModel(param1=1.5, param2=3.0), "univariate": False}
    }
```

3. **Run your model**:
```bash
uv run examples/run_musedfm.py --models my_custom --output-dir /path/to/results --benchmark-path /path/to/data
```

### Model Classification

- **Univariate models** (`"univariate": True`): Use only the target time series for forecasting
- **Multivariate models** (`"univariate": False`): Use both target series and covariates for forecasting

### Programmatic Usage

```python
from musedfm.data import Benchmark
from musedfm.baselines.linear_regression import LinearRegressionForecast

benchmark = Benchmark("/path/to/extracted/data")

# Create a model that can do both univariate and multivariate forecasting
model = LinearRegressionForecast()

for category in benchmark:
    for domain in category:
        for dataset in domain:
            for window in dataset:
                # Generate multivariate forecast (uses covariates)
                multivariate_forecast = model.forecast(window.history(), window.covariates(), len(window.target()))
                
                # Generate univariate forecast (ignores covariates)
                univariate_model = LinearRegressionForecast(use_covariates=False)
                univariate_forecast = univariate_model.forecast(window.history(), None, len(window.target()))
                
                # Submit both forecasts
                window.submit_forecast(multivariate_forecast, univariate_forecast)
                
                # Get evaluation results for both
                multivariate_results = window.evaluate("multivariate")
                univariate_results = window.evaluate("univariate")
                
                print(f"Multivariate MAPE: {multivariate_results['MAPE']:.2f}%")
                print(f"Univariate MAPE: {univariate_results['MAPE']:.2f}%")

# Export results
benchmark.to_results_csv("results.csv")
```

---

## Installation

### Prerequisites
- Python 3.9 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Install Dependencies
```bash
# Install project dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows
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

The framework generates:
- **Console output**: Real-time progress and model performance summaries
- **CSV files**: Detailed metrics for each dataset and model
- **Plots** (optional): Forecast visualizations for analysis
- **Aggregated results**: Overall performance across all datasets

Example output:
```
mean Summary:
  Total windows: 1000
  Total time: 5.13s
  Model type: Univariate
  Average MAPE: 12.45%
  Average MAE: 0.0234
  Average RMSE: 0.0345
  Average NMAE: 0.1234

linear_regression Summary:
  Total windows: 1000
  Total time: 8.45s
  Model type: Multivariate
  Average MAPE: 10.23%
  Average MAE: 0.0198
  Average RMSE: 0.0289
  Average NMAE: 0.0987

linear_regression_univariate Summary:
  Total windows: 1000
  Total time: 6.12s
  Model type: Univariate
  Average MAPE: 11.87%
  Average MAE: 0.0212
  Average RMSE: 0.0312
  Average NMAE: 0.1123
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