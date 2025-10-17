"""
TimesFM forecasting model for MUSED-FM evaluation.
"""

import numpy as np
from typing import Optional
from musedfm.baselines.utils import standard_normalize


class TimesFMForecast:
    """
    TimesFM forecasting model wrapper for MUSED-FM evaluation.
    """
    
    def __init__(self, model_path: str = "google/timesfm-2.0-500m-pytorch", device: str = "cpu", num_samples: int = 20):
        """
        Initialize TimesFM forecast model.
        
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
        """Load the TimesFM model using the current API."""
        try:
            import timesfm
            
            # Create hyperparameters for the model
            hparams = timesfm.TimesFmHparams(
                context_len=2048,  # Max context length for TimesFM 2.0
                horizon_len=128,
                backend=self.device,
                per_core_batch_size=32,
                num_layers=50,
                use_positional_embedding=False,
                quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
            )
            
            # Create checkpoint configuration
            checkpoint = timesfm.TimesFmCheckpoint(
                version='torch',  # Use PyTorch backend for PyTorch model
                huggingface_repo_id=self.model_path
            )
            
            # Create the model using the factory function
            self.model = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
            print(f"Loaded TimesFM model: {self.model_path}")
            
        except ImportError:
            raise ImportError("TimesFM package not installed. Please install with: pip install timesfm")
        except Exception as e:
            raise RuntimeError(f"Failed to load TimesFM model: {e}")
    
    def forecast(self, history: np.ndarray, covariates: Optional[np.ndarray] = None, forecast_horizon: Optional[int] = None, timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate forecast from historical data using TimesFM.
        
        Args:
            history: Historical time series data (shape: [batch_size, history_length])
            covariates: Optional covariate data (shape: [batch_size, history_length, covariate_dim])
            forecast_horizon: Number of future points to forecast (default: 1)
            timestamps: Optional timestamp data (ignored for TimesFM)
            
        Returns:
            Forecast values (shape: [batch_size, forecast_horizon])
        """
        if forecast_horizon is None:
            forecast_horizon = 1
        
        batch_size = history.shape[0]
        forecasts = np.zeros((batch_size, forecast_horizon))
        
        
        # Check if we have sufficient data
        if history.shape[1] < 1:
            return forecasts
        
        # Normalize the data using standardized utils functions
        history_normalized, history_mean, history_std = standard_normalize(
            history, 
            axis=1, 
            keepdims=True,
            epsilon=1e-6
        )
        # Handle NaN values - replace with 0.0 for processing
        history_normalized = np.nan_to_num(history_normalized, nan=0.0)
        
        # Prepare batch input for TimesFM
        batch_inputs = []
        
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
            
            # Convert to list format for TimesFM API
            for i in range(batch_size):
                batch_inputs.append(input_series_array[i].tolist())
        else:
            # Univariate case - each series is a single list of values
            for i in range(batch_size):
                batch_inputs.append(history_normalized[i].tolist())
        
        # Generate forecasts for the entire batch at once
        point_forecasts, quantile_forecasts = self.model.forecast(
            inputs=batch_inputs,
            normalize=True
        )
        
        # Extract forecasts - take only the requested horizon length
        forecasts = point_forecasts[:, :forecast_horizon]
        
        return forecasts

