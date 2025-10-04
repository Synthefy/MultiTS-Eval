# MUSED-FM

# mused-fm Client Package Design

## Overview
The `mused-fm` client provides iteration and evaluation utilities for the **MUSED-FM multivariate timeseries evaluation dataset**. Users download the dataset from Hugging Face as a `.tar.gz`, extract it, and then interact with it via this package.

Data is structured as:
- **Benchmark** → container of multiple domains  
- **Category (Hierarchy)** -> contains multiple domains
- **Domain (Hierarchy)** → contains multiple datasets  
- **Dataset** → contains multiple parquet files  
- **Window** → extracted slices from parquet files  

The breakdown of domains, categories and datasets can be found in data/data_hierarchy.json

The client exposes iteration over all levels, and provides methods for forecasting submission and evaluation.

---

## Directory Layout
```
src/mused_fm/
    data/
        window.py
        dataset.py
        domain.py
        benchmark.py
    metrics.py
```

---

## Core Components

### 1. Window
- **Represents a single evaluation unit** (history, target, covariates).
- Stores ground truth and submitted forecast.

**Methods**
- `history() -> np.ndarray`
- `target() -> np.ndarray`
- `covariates() -> np.ndarray`
- `submit_forecast(forecast: np.ndarray) -> None`  
  Stores forecast, triggers evaluation once for this window.
- `evaluate() -> dict`  
  Runs metrics (e.g., `MAPE`) from `metrics.py` and caches results.

---

### 2. Dataset
- **Collection of windows** (parsed from parquet files).
- Iterable: `for window in dataset`
- Aggregates results across windows.

**Methods**
- `__iter__() -> Iterator[Window]`
- `evaluate() -> dict`  
  Runs evaluation for all windows (delegates to their cached results).
- `to_results_csv(path: str) -> None`  
  - Validates all windows have forecasts submitted.  
  - Collects cached results, computes aggregates, saves CSV.  
  - **Does not recompute metrics or save forecasts.**

---

### 3. Domain
- **Encapsulates multiple datasets**.
- Iterable: `for dataset in domain`
- Aggregates results across datasets.

**Methods**
- `__iter__() -> Iterator[Dataset]`
- `evaluate() -> dict`  
  Runs evaluation across all datasets (cached results only).
- `to_results_csv(path: str) -> None`  
  Same semantics as dataset: validate → aggregate → save.

---

### 4. Category
- **A grouping of datasets which have similar data properties The four categories are named based on the tar.gz files: traditional, collections, sequential and synthetic**.
- Iterable: `for domain in category`
- Aggregates results across categories.

**Methods**
- `__iter__() -> Iterator[Domain]`
- `evaluate() -> dict`  
  Runs evaluation across all domains in the category ().
- `to_results_csv(path: str) -> None`  
  Same semantics as dataset: validate → aggregate → save.

---


### 5. Benchmark
- **Top-level container for multiple domains**.
- Simple orchestrator for evaluations and exports.

**Methods**
- `__iter__() -> Iterator[Domain]`
- `evaluate() -> dict`  
  Aggregate results across domains (cached results only).
- `to_results_csv(path: str) -> None`  
  Validate forecasts, gather results from all domains, write consolidated CSV.

---

## Metrics
- Defined in `src/mused_fm/metrics.py`
- Example:
  ```python
  def MAPE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
      ...
  ```
- `Window.evaluate()` calls metrics once and caches results.  
- Higher-level `evaluate()` methods aggregate cached results.

---

## Usage Example
```python
from mused_fm.data import Benchmark

benchmark = Benchmark("/path/to/extracted/data")

for category in benchmark:
    for domain in category:
        for dataset in domain:
            for window in dataset:
                forecast = my_model.forecast(window.history(), window.covariates())
                window.submit_forecast(forecast)

benchmark.evaluate()
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

## Testing

### Run Comprehensive Test Suite
The test suite includes baseline forecasting methods and validates the entire MUSED-FM pipeline:

```bash
# Run the comprehensive test
uv run python test_musedfm.py
```

### Baseline Forecasting Methods
The test includes several baseline methods:

1. **Mean Forecast**: Forecasts the mean of historical data
2. **Historical Inertia**: Uses the last observed value (based on [Historical Inertia](https://arxiv.org/pdf/2103.16349))
3. **ARIMA**: AutoRegressive Integrated Moving Average model
4. **Linear Trend**: Linear trend extrapolation
5. **Exponential Smoothing**: Simple exponential smoothing

### Test Structure
The test suite validates:
- ✅ Single dataset loading and processing
- ✅ Domain-level aggregation
- ✅ Category-level aggregation  
- ✅ Benchmark-level aggregation
- ✅ CSV export functionality
- ✅ All baseline forecasting methods

### Performance Notes
- Tests are designed to run quickly (< 2 minutes)
- Uses limited window counts to avoid long execution times
- Processes only first few windows from large datasets
- Includes timing measurements for performance validation

## Notes
- **One-time evaluation per window**: `submit_forecast()` triggers evaluation, results are cached.  
- **No recomputation in `to_results_csv()`**: it only checks completeness, aggregates cached results, and writes them out.  
- **Forecast arrays are not saved**: only metrics/results are written to CSV to keep disk usage low.  
