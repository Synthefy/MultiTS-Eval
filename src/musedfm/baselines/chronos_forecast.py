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
            
            self.pipeline: BaseChronosPipeline = BaseChronosPipeline.from_pretrained(
                self.model_path,
                device_map=self.device,
            )
            print(f"Loaded Chronos model: {self.model_path}")
        except ImportError:
            raise ImportError("Chronos package not installed. Please install with: pip install chronos-forecasting")
        except Exception as e:
            raise RuntimeError(f"Failed to load Chronos model: {e}")
    
    def forecast(self, history: np.ndarray, covariates: Optional[np.ndarray] = None, forecast_horizon: Optional[int] = None, timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate forecast from historical data using Chronos.
        
        Args:
            history: Historical time series data (shape: [batch_size, history_length])
            covariates: Optional covariate data (ignored for Chronos)
            forecast_horizon: Number of future points to forecast (default: 1)
            timestamps: Optional timestamp data (ignored for Chronos)
            
        Returns:
            Forecast values (shape: [batch_size, forecast_horizon])
        """
        if forecast_horizon is None:
            forecast_horizon = 1
        
        # Vectorized normalization across the batch
        history_mean = np.nanmean(history, axis=1, keepdims=True)  # [batch_size, 1]
        history_std = np.nanstd(history, axis=1, keepdims=True)    # [batch_size, 1]
        history_normalized = (history - history_mean) / np.maximum(history_std, 1e-10)
        
        # Convert to torch tensor and handle NaNs
        history_tensor = torch.tensor(history_normalized, dtype=torch.float32)
        history_tensor = torch.nan_to_num(history_tensor, nan=0.0)
        
        # Generate forecast (Chronos accepts 2D tensor directly)
        forecast_output = self.pipeline.predict(
            history_tensor,
            prediction_length=forecast_horizon
        )
        
        # Convert to numpy array
        if isinstance(forecast_output, torch.Tensor):
            forecast_np = forecast_output.detach().cpu().numpy()
        else:
            forecast_np = np.array(forecast_output)
        
        # Handle quantile output - take median (index 4 for 9 quantiles)
        if forecast_np.ndim == 3 and forecast_np.shape[1] == 9:
            forecast_np = forecast_np[:, 4, :]  # [batch_size, prediction_length]
        
        # Ensure correct length
        if forecast_np.shape[1] != forecast_horizon:
            if forecast_np.shape[1] > forecast_horizon:
                forecast_np = forecast_np[:, :forecast_horizon]
            else:
                # Pad with last value
                last_values = forecast_np[:, -1:]  # [batch_size, 1]
                padding_length = forecast_horizon - forecast_np.shape[1]
                padding = np.repeat(last_values, padding_length, axis=1)
                forecast_np = np.concatenate([forecast_np, padding], axis=1)
        
        # Denormalize
        forecast_np = forecast_np * history_std + history_mean
        
        return forecast_np
