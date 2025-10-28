from typing import Dict, Any

def get_available_models(device: str = "cuda:0", use_additional_models: bool = False):
    """Get dictionary of available forecasting models.
    
    To add your own custom model:
    1. Create a class that implements the forecast() method
    2. Add it to this dictionary with a descriptive name
    3. The forecast() method should accept: history, covariates, forecast_length
    4. It should return a numpy array of forecasts
    
    Example custom model:
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
    
    Then add to the dictionary:
    "my_custom": MyCustomModel(param1=1.5, param2=3.0)
    
    Args:
        device: Device to use for models that support it (e.g., "cuda:0", "cpu")
    """
    from museval.baselines import (
        MeanForecast, 
        HistoricalInertia, 
        ARIMAForecast, 
        LinearTrend, 
        ExponentialSmoothing
    )
    from museval.baselines.linear_regression import LinearRegressionForecast
    from museval.baselines.chronos_forecast import ChronosForecast
    
    models = {
        "mean": {"model": MeanForecast(), "univariate": True},
        "historical_inertia": {"model": HistoricalInertia(), "univariate": True},
        "linear_trend": {"model": LinearTrend(), "univariate": True},
        "exponential_smoothing": {"model": ExponentialSmoothing(), "univariate": True},
        "arima": {"model": ARIMAForecast(order=(1, 1, 1)), "univariate": True},
        "linear_regression": {"model": LinearRegressionForecast(), "univariate": False},
        "chronos": {"model": ChronosForecast(device=device), "univariate": True},
        # Add your custom models here:
        # "my_custom": {"model": MyCustomModel(), "univariate": False},
        # "another_model": {"model": AnotherModel(param1=value1, param2=value2), "univariate": True}
    }
    if use_additional_models:
        from museval.baselines.moirai_forecast import MoiraiForecast
        from museval.baselines.toto_forecast import TotoForecast
        models.update({
            "moirai": {"model": MoiraiForecast(device=device), "univariate": True},
            "toto": {"model": TotoForecast(), "univariate": False}
        })
        from museval.baselines.timesfm_forecast import TimesFMForecast
        models.update({
            "timesfm": {"model": TimesFMForecast(device=device), "univariate": False}
        })
    
    return models


def parse_models(model_string: str, device: str = "cuda:0") -> Dict[str, Any]:
    """Parse model string and return list of model instances."""
    if model_string.lower() in ["moirai", "timesfm", "toto", "all_additional"]:
        use_additional_models = True
    else:
        use_additional_models = False
    available_models = get_available_models(device=device, use_additional_models=use_additional_models)
    
    if model_string.lower() == "all" or model_string.lower() == "all_additional":
        return available_models
    
    model_names = [name.strip().lower() for name in model_string.split(",")]
    selected_models = {}
    
    for name in model_names:
        if name in available_models:
            selected_models[name] = available_models[name]
        else:
            print(f"Warning: Unknown model '{name}'. Available models: {list(available_models.keys())}")
    
    return selected_models
