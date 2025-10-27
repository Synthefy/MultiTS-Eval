"""
Chronos-2 forecasting model for evaluation.
"""

import numpy as np
import torch
from typing import Optional

class Chronos2Forecast:    
    def __init__(self, model_path: str = "s3://autogluon/chronos-2", device: str = "cuda:0", num_samples: int = 20):
        """
        Initialize Chronos-2 forecast model.
        
        Args:
            model_path: Path to Chronos-2 model (S3 path or local path)
            device: Device to run the model on
            num_samples: Number of samples for probabilistic forecasting
        """
        self.model_path = model_path
        self.device = device
        self.num_samples = num_samples
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the Chronos-2 model."""
        try:
            from chronos import Chronos2Pipeline
            
            self.pipeline: Chronos2Pipeline = Chronos2Pipeline.from_pretrained(
                self.model_path,
                device_map=self.device,
            )
            print(f"Loaded Chronos-2 model: {self.model_path}")
        except ImportError:
            raise ImportError("Chronos package not installed. Please install with: pip install chronos-forecasting")
        except Exception as e:
            raise RuntimeError(f"Failed to load Chronos-2 model: {e}")
    
    def forecast(self, history: np.ndarray, covariates: Optional[np.ndarray] = None, forecast_horizon: Optional[int] = None, timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate forecast from historical data using Chronos-2.
        
        Args:
            history: Historical time series data (shape: [batch_size, history_length])
            covariates: Optional covariate data (ignored for Chronos-2)
            forecast_horizon: Number of future points to forecast (default: 1)
            timestamps: Optional timestamp data (ignored for Chronos-2)
            
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
        
        # Chronos-2 expects 3D input: (batch_size, n_variates, history_length)
        # Add variate dimension for univariate time series
        history_tensor = history_tensor.unsqueeze(1)  # [batch_size, 1, history_length]
        
        # Generate forecast
        forecast_output = self.pipeline.predict(
            history_tensor,
            prediction_length=forecast_horizon
        )
        
        # Convert to numpy array
        if isinstance(forecast_output, torch.Tensor):
            forecast_np = forecast_output.detach().cpu().numpy()
        else:
            forecast_np = np.array(forecast_output)
        
        # Chronos2 returns output in shape (batch_size, n_variates, prediction_length) for univariate
        original_shape = forecast_np.shape
        
        # Output shape handling
        if forecast_np.ndim == 2:
            pass
        elif forecast_np.ndim == 3:
            # For univariate with n_variates=1, squeeze out the middle dimension
            if forecast_np.shape[1] == 1:
                forecast_np = forecast_np.squeeze(1)  # (batch_size, prediction_length)
            elif forecast_np.shape[1] == 9:
                # Probabilistic output with 9 quantiles - take median
                forecast_np = forecast_np[:, 4, :]  # (batch_size, prediction_length)
            else:
                # Unexpected shape - take the mean across the middle dimension
                # This handles cases with multiple samples/quantiles
                print(f"Warning: Unexpected 3D output shape: {forecast_np.shape}, taking mean across middle dimension")
                forecast_np = np.mean(forecast_np, axis=1)  # (batch_size, prediction_length)
        elif forecast_np.ndim == 4:
            # Shape: (batch_size, n_variates, n_samples_or_quantiles, prediction_length)
            n_quantiles = forecast_np.shape[2]
            
            if n_quantiles == 9:
                # Standard 9 quantiles - take median (index 4)
                forecast_np = forecast_np[:, :, 4, :]  # (batch_size, n_variates, prediction_length)
            else:
                # Non-standard quantile count - take mean across quantiles
                print(f"Warning: 4D output with {n_quantiles} quantiles/samples, taking mean")
                forecast_np = np.mean(forecast_np, axis=2)  # (batch_size, n_variates, prediction_length)
            
            # Squeeze variate dimension for univariate (should be 1)
            if forecast_np.shape[1] == 1:
                forecast_np = forecast_np.squeeze(1)  # (batch_size, prediction_length)
        
        # Ensure we have 2D output
        if forecast_np.ndim != 2:
            raise ValueError(f"Expected 2D output, got shape: {forecast_np.shape} (original: {original_shape})")
        
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
