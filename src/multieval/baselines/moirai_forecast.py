"""
Moirai 2.0 forecasting model for MUSED-FM evaluation.
"""

import numpy as np
from typing import Optional
from .utils import standard_normalize, standard_denormalize
from gluonts.dataset.common import ListDataset



class MoiraiForecast:
    """
    Moirai 2.0 forecasting model wrapper for MUSED-FM evaluation.
    """
    
    def __init__(self, model_size: str = "small", device: str = "cuda:0", num_samples: int = 100):
        """
        Initialize Moirai 2.0 forecast model.
        
        Args:
            model_size: Size of Moirai model ("small", "base", "large")
            device: Device to run the model on
            num_samples: Number of samples for probabilistic forecasting
        """
        self.model_size = model_size
        self.device = device
        self.num_samples = num_samples
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Moirai 2.0 model."""
        try:
            from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
            
            # Load Moirai 2.0 model
            model_name = f"Salesforce/moirai-2.0-R-{self.model_size}"
            
            self.model = Moirai2Forecast(
                module=Moirai2Module.from_pretrained(model_name),
                prediction_length=256,  # Max prediction length
                context_length=1680,    # Max context length for Moirai 2.0
                target_dim=1,           # Univariate forecasting
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            print(f"Loaded Moirai 2.0 model: {model_name}")
        except ImportError:
            raise ImportError("Moirai 2.0 package not installed. Please install with: pip install uni2ts")
        except Exception as e:
            raise RuntimeError(f"Failed to load Moirai 2.0 model: {e}")
    
    def forecast(self, history: np.ndarray, covariates: Optional[np.ndarray] = None, forecast_horizon: Optional[int] = None, timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate forecast from historical data using Moirai 2.0.
        
        Args:
            history: Historical time series data (shape: [batch_size, history_length])
            covariates: Optional covariate data (ignored for Moirai)
            forecast_horizon: Number of future points to forecast (default: 1)
            timestamps: Optional timestamp data (ignored for Moirai)
            
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
        
        # Handle NaN values in input data first
        history_clean = np.nan_to_num(history, nan=0.0)
        
        # Check for completely zero or constant data that might cause issues
        for i in range(batch_size):
            if np.all(history_clean[i] == 0) or np.std(history_clean[i]) < 1e-10:
                # Use mean forecast for problematic data
                forecasts[i] = np.mean(history_clean[i]) if np.mean(history_clean[i]) != 0 else 1.0
                continue
        
        # Normalize the data using standardized utils functions
        try:
            history_normalized, history_mean, history_std = standard_normalize(
                history_clean, 
                axis=1, 
                keepdims=True,
                epsilon=1e-6
            )
            
            # Check for NaN values in normalized data
            if np.any(np.isnan(history_normalized)) or np.any(np.isnan(history_mean)) or np.any(np.isnan(history_std)):
                # Fallback to mean forecast if normalization fails
                return np.tile(np.nanmean(history_clean, axis=1, keepdims=True), (1, forecast_horizon))
                
        except Exception as e:
            # Fallback to mean forecast if normalization fails
            print(f"Warning: Moirai normalization failed: {e}")
            return np.tile(np.nanmean(history_clean, axis=1, keepdims=True), (1, forecast_horizon))
        
        # Create predictor for the entire batch
        try:
            predictor = self.model.create_predictor(batch_size=batch_size)
        except Exception as e:
            print(f"Warning: Moirai predictor creation failed: {e}")
            return np.tile(np.nanmean(history_clean, axis=1, keepdims=True), (1, forecast_horizon))
        
        # Prepare data in GluonTS format following the Hugging Face example
        
        # Create dataset entries for all batch elements
        dataset_entries = []
        for i in range(batch_size):
            # Skip if this sample already has a forecast (from problematic data check)
            if not np.all(forecasts[i] == 0):
                continue
                
            # Use actual timestamps if available, otherwise use dummy dates
            if timestamps is not None and timestamps.shape[0] > i:
                # Convert timestamps to proper format for GluonTS
                start_timestamp = timestamps[i, 0] if timestamps.shape[1] > 0 else "2020-01-01"
                if hasattr(start_timestamp, 'strftime'):
                    start_date = start_timestamp.strftime('%Y-%m-%d')
                else:
                    start_date = "2020-01-01"  # Fallback
            else:
                start_date = "2020-01-01"  # Dummy start date
            
            dataset_entry = {
                "start": start_date,
                "target": history_normalized[i],
                "item_id": f"sample_{i}"
            }
            dataset_entries.append(dataset_entry)
        
        # Only proceed if we have valid dataset entries
        if not dataset_entries:
            return forecasts
        
        try:
            # Create dataset
            dataset = ListDataset(dataset_entries, freq="D")
            
            # Generate forecasts for the entire batch
            forecasts_list = list(predictor.predict(dataset))
            
            # Process forecasts in batch - extract all forecast values at once
            if forecasts_list:
                # Extract all forecast values as a batch (Moirai 2.0 returns median by default)
                forecast_values = []
                for forecast_sample in forecasts_list:
                    if forecast_sample is not None and hasattr(forecast_sample, 'median'):
                        median_values = forecast_sample.median
                        # Check for NaN values in the forecast
                        if np.any(np.isnan(median_values)):
                            # Use mean forecast if Moirai returns NaN
                            forecast_values.append(np.full(forecast_horizon, np.mean(history_normalized[len(forecast_values)])))
                        else:
                            forecast_values.append(median_values)
                    else:
                        # Create mean forecast for failed predictions
                        forecast_values.append(np.full(forecast_horizon, np.mean(history_normalized[len(forecast_values)])))
                
                # Convert to numpy array for batch processing
                forecast_values = np.array(forecast_values)
                
                # Ensure correct length for all forecasts at once
                if forecast_values.shape[1] != forecast_horizon:
                    if forecast_values.shape[1] > forecast_horizon:
                        forecast_values = forecast_values[:, :forecast_horizon]
                    else:
                        # Pad with last values
                        last_values = forecast_values[:, -1:] if forecast_values.shape[1] > 0 else np.zeros((len(forecast_values), 1))
                        padding_length = forecast_horizon - forecast_values.shape[1]
                        padding = np.tile(last_values, (1, padding_length))
                        forecast_values = np.concatenate([forecast_values, padding], axis=1)
                
                # Denormalize entire batch at once using utils function
                try:
                    denormalized_forecasts = standard_denormalize(forecast_values, history_mean[:len(forecast_values), 0:1], history_std[:len(forecast_values), 0:1])
                    
                    # Check for NaN values after denormalization
                    if np.any(np.isnan(denormalized_forecasts)):
                        # Fallback to mean forecast if denormalization produces NaN
                        denormalized_forecasts = np.tile(history_mean[:len(forecast_values), 0:1], (1, forecast_horizon))
                    
                    # Update forecasts for the samples that were processed
                    sample_idx = 0
                    for i in range(batch_size):
                        if np.all(forecasts[i] == 0):  # This sample was processed
                            forecasts[i] = denormalized_forecasts[sample_idx]
                            sample_idx += 1
                            
                except Exception as e:
                    print(f"Warning: Moirai denormalization failed: {e}")
                    # Fallback to mean forecast
                    sample_idx = 0
                    for i in range(batch_size):
                        if np.all(forecasts[i] == 0):  # This sample was processed
                            forecasts[i] = np.full(forecast_horizon, np.mean(history_clean[i]))
                            sample_idx += 1
            else:
                # Fallback to mean forecast for entire batch
                for i in range(batch_size):
                    if np.all(forecasts[i] == 0):  # This sample was processed
                        forecasts[i] = np.full(forecast_horizon, np.mean(history_clean[i]))
                        
        except Exception as e:
            print(f"Warning: Moirai prediction failed: {e}")
            # Fallback to mean forecast for all samples
            for i in range(batch_size):
                if np.all(forecasts[i] == 0):  # This sample was processed
                    forecasts[i] = np.full(forecast_horizon, np.mean(history_clean[i]))
        
        # Final check for any remaining NaN values
        forecasts = np.nan_to_num(forecasts, nan=0.0)
        
        return forecasts

