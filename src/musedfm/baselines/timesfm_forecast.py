"""
TimesFM 2.5 forecasting model for MUSED-FM evaluation.
"""

import numpy as np
import torch
from typing import Optional
from musedfm.baselines.utils import standard_normalize


class TimesFMForecast:
    """
    TimesFM 2.5 forecasting model wrapper for MUSED-FM evaluation.
    """
    
    def __init__(self, model_path: str = "google/timesfm-2.5-200m-pytorch", device: str = "cpu", num_samples: int = 20):
        """
        Initialize TimesFM 2.5 forecast model.
        
        Args:
            model_path: Path to TimesFM model (HuggingFace model ID or local path)
            device: Device to run the model on ('cpu' or 'gpu')
            num_samples: Number of samples for probabilistic forecasting
        """
        self.model_path = model_path
        self.device = device
        self.num_samples = num_samples
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the TimesFM 2.5 model using the new API."""
        try:
            import timesfm
            
            # Set PyTorch precision for better performance
            torch.set_float32_matmul_precision("high")
            
            # Load TimesFM 2.5 model
            try:
                self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_path)
            except AttributeError:
                # Fallback to older TimesFM version if 2.5 is not available
                print("TimesFM 2.5 not available, skipping TimesFM model loading")
                raise RuntimeError("TimesFM 2.5 model not available")
            
            # Compile model with TimesFM 2.5 configuration
            self.model.compile(
                timesfm.ForecastConfig(
                    max_context=1024,  # Can go up to 16k in TimesFM 2.5
                    max_horizon=256,   # Can go up to 1k with quantile head
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )
            print(f"Loaded TimesFM 2.5 model: {self.model_path}")
            
        except ImportError:
            raise ImportError("TimesFM 2.5 package not installed. Please install from: https://github.com/google-research/timesfm/")
        except Exception as e:
            raise RuntimeError(f"Failed to load TimesFM 2.5 model: {e}")
    
    def forecast(self, history: np.ndarray, covariates: Optional[np.ndarray] = None, forecast_horizon: Optional[int] = None, timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate forecast from historical data using TimesFM 2.5.
        
        Args:
            history: Historical time series data (shape: [batch_size, history_length])
            covariates: Optional covariate data (shape: [batch_size, history_length, covariate_dim])
            forecast_horizon: Number of future points to forecast (default: 1)
            timestamps: Optional timestamp data (ignored for TimesFM 2.5)
            
        Returns:
            Forecast values (shape: [batch_size, forecast_horizon])
        """
        if forecast_horizon is None:
            forecast_horizon = 1
        
        batch_size = history.shape[0]
        
        # Check if we have sufficient data
        if history.shape[1] < 1:
            return np.zeros((batch_size, forecast_horizon))
        
        # Normalize the data using standardized utils functions
        history_normalized, history_mean, history_std = standard_normalize(
            history, 
            axis=1, 
            keepdims=True,
            epsilon=1e-6
        )
        # Handle NaN values - replace with 0.0 for processing
        history_normalized = np.nan_to_num(history_normalized, nan=0.0)
        
        # Prepare inputs for TimesFM 2.5
        inputs = []
        
        if covariates is not None and len(covariates.shape) == 3:
            # Normalize covariates using standardized utils functions
            covariates_normalized, covariates_mean, covariates_std = standard_normalize(
                covariates,
                axis=1,
                keepdims=True,
                epsilon=1e-6
            )
            # Multivariate case: flatten history and covariates into single series
            covariates_processed = np.nan_to_num(covariates_normalized, nan=0.0)
            
            # Flatten covariates from [batch_size, history_length, covariate_dim] to [batch_size, history_length * covariate_dim]
            covariates_flat = covariates_processed.reshape(batch_size, -1)
            
            # Combine history and flattened covariates: [batch_size, history_length + covariate_dim * history_length]
            input_series_array = np.concatenate([
                history_normalized,  # [batch_size, history_length]
                covariates_flat     # [batch_size, history_length * covariate_dim]
            ], axis=1)  # [batch_size, history_length + covariate_dim * history_length]
            
            # Convert to list format for TimesFM 2.5 API
            for i in range(batch_size):
                inputs.append(input_series_array[i].tolist())
        else:
            # Univariate case - each series is a single list of values
            for i in range(batch_size):
                inputs.append(history_normalized[i].tolist())
        
        # Generate forecasts using TimesFM 2.5 API
        point_forecast, quantile_forecast = self.model.forecast(
            horizon=forecast_horizon,
            inputs=inputs
        )
        
        # Extract forecasts - TimesFM 2.5 returns numpy arrays directly
        forecasts = point_forecast[:, :forecast_horizon]
        
        return forecasts

