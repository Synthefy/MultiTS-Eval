# MultiTS-Eval Submission Results

This directory contains submission results for the MultiTS-Eval multivariate time series forecasting benchmark. Each submission includes model metadata and evaluation results across multiple domains and datasets.

## Submission Structure

Each submission directory contains:
- `metadata.json` - Model information, submitter details, and description
- `{model_name}_submission.json` - Evaluation results in the required competition format

## Metadata Format

The `metadata.json` file contains information about the model and submission. It should follow this structure:

```json
{
    "model": "Model Name",
    "submitter": "Submitter Name/Organization",
    "submission_date": "YYYY-MM-DD",
    "task": "multivariate_forecasting",
    "dataset_version": "v1.0",
    "paper_url": "https://example.com/paper",
    "code_url": "https://github.com/example/repo",
    "description": "Detailed model description"
}
```

### Required Fields

- **`model`**: The name of your forecasting model
- **`submitter`**: Your name or organization name  
- **`submission_date`**: Date of submission in YYYY-MM-DD format
- **`task`**: Always "multivariate_forecasting" for this benchmark
- **`dataset_version`**: Currently "v1.0"
- **`description`**: Comprehensive description of your model, methodology, and key innovations

### Optional Fields

- **`paper_url`**: Link to your research paper or preprint
- **`code_url`**: Link to your code repository
- **`model_url`**: Link to pre-trained model (e.g., Hugging Face, model zoo)

### Field Examples

**Model Names**: "ARIMA", "Chronos Bolt", "Transformer-Forecast", "LSTM-Multivariate"

**Submitters**: "Synthefy", "Amazon", "Your Organization", "Individual Researcher"

**Descriptions**: Should include:
- Model architecture and methodology
- Key innovations or improvements
- Training approach and data handling
- Performance characteristics

## Submission Format

Each submission JSON file contains an array of results with the following structure (note multi- and uni-mape correspond to the model run with and without variates, and should differ if the model has different settings for each):

```json
[
  {
    "domain": "Stock",
    "category": "collections", 
    "dataset": "stock_nasdaqtrader",
    "dataset_version": "v1.0",
    "metrics": {
      "MAE": 9.43,
      "RMSE": 10.12,
      "Multi-MAPE": 86.78,
      "Uni-MAPE": 86.78,
      "NMAE": 1.47
    }
  }
]
```
