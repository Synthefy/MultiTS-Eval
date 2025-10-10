"""
Chronos forecasting model for MUSED-FM evaluation.
"""

import numpy as np
import torch
from typing import Optional


class ChronosForecast:
    """
    Chronos forecasting model wrapper for MUSED-FM evaluation.
    """
    
    def __init__(self, model_path: str = "amazon/chronos-bolt-base", device: str = "cuda:0", num_samples: int = 20):
        """
        Initialize Chronos forecast model.
        
        Args:
            model_path: Path to Chronos model (HuggingFace model ID or local path)
            device: Device to run the model on
            num_samples: Number of samples for probabilistic forecasting
        """
        self.model_path = model_path
        self.device = device
        self.num_samples = num_samples
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the Chronos model."""
        try:
            from chronos import BaseChronosPipeline, ForecastType
            
            self.pipeline = BaseChronosPipeline.from_pretrained(
                self.model_path,
                device_map=self.device,
            )
            print(f"Loaded Chronos model: {self.model_path}")
        except ImportError:
            raise ImportError("Chronos package not installed. Please install with: pip install chronos-forecasting")
        except Exception as e:
            raise RuntimeError(f"Failed to load Chronos model: {e}")
    
    def forecast(self, history: np.ndarray, covariates: Optional[np.ndarray] = None, forecast_horizon: Optional[int] = None) -> np.ndarray:
        """
        Generate forecast from historical data using Chronos.
        
        Args:
            history: Historical time series data
            covariates: Optional covariate data (ignored for Chronos)
            forecast_horizon: Number of future points to forecast (default: 1)
            
        Returns:
            Forecast values
        """
        if forecast_horizon is None:
            forecast_horizon = 1
        
        # normalize using history prior to forecasting
        history_mean = np.mean(history)
        history_std = np.std(history)
        history = (history - history_mean) / max(history_std, 1e-10)
        
        # Convert history to torch tensor
        if isinstance(history, np.ndarray):
            history_tensor = torch.tensor(history, dtype=torch.float32)
        else:
            history_tensor = torch.tensor(np.array(history), dtype=torch.float32)
        
        # Remove NaN values
        history_clean = history_tensor[~torch.isnan(history_tensor)]
        
        if len(history_clean) == 0:
            # If no valid data, return zeros
            return np.zeros(forecast_horizon)
        
        # Ensure we have enough history for forecasting
        if len(history_clean) < 2:
            # If insufficient data, return the last value repeated
            last_value = float(history_clean[-1]) if len(history_clean) > 0 else 0.0
            return np.full(forecast_horizon, last_value)
        
        # Generate forecast using Chronos
        # Use predict method instead of predict_quantiles
        context = [history_clean]                                                                                       
        
        # Determine prediction kwargs based on forecast type
        predict_kwargs = {}
        if hasattr(self.pipeline, 'forecast_type'):
            from chronos import ForecastType
            if self.pipeline.forecast_type == ForecastType.SAMPLES:
                predict_kwargs = {"num_samples": self.num_samples}
        
        # Generate forecast
        forecast_output = self.pipeline.predict(
            context,
            prediction_length=forecast_horizon,
            **predict_kwargs
        )
        
        # Convert to numpy array
        if isinstance(forecast_output, torch.Tensor):
            forecast_np = forecast_output.detach().cpu().numpy()
        else:
            forecast_np = np.array(forecast_output)
        
        # Handle different output shapes
        if forecast_np.ndim > 1:
            # Chronos Bolt returns (batch_size, num_quantiles, prediction_length)
            # We want the median (0.5 quantile) which is typically at index 4 (0.1, 0.2, ..., 0.9)
            if forecast_np.ndim == 3 and forecast_np.shape[1] == 9:  # Standard Chronos Bolt quantiles
                # Take the median (0.5 quantile) at index 4
                forecast_np = forecast_np[0, 4, :]  # (batch_size=1, quantile=4, prediction_length)
            elif forecast_np.shape[0] > 1:
                # If we have multiple samples, take the mean
                forecast_np = np.mean(forecast_np, axis=0)
            else:
                forecast_np = forecast_np[0]
        
        # Ensure we have the right length
        if len(forecast_np) != forecast_horizon:
            if len(forecast_np) > forecast_horizon:
                forecast_np = forecast_np[:forecast_horizon]
            else:
                # Pad with the last value if needed - use concatenation to avoid broadcasting issues
                if len(forecast_np) > 0:
                    last_val = float(forecast_np[-1])
                    padding_length = forecast_horizon - len(forecast_np)
                    padding = np.full(padding_length, last_val)
                    forecast_np = np.concatenate([forecast_np, padding])
                else:
                    # If no valid forecast, fill with zeros
                    forecast_np = np.zeros(forecast_horizon)
        
        # denormalize using history prior to forecasting
        forecast_np = forecast_np * history_std + history_mean
        
        return forecast_np
