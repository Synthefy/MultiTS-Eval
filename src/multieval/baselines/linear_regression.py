"""
Linear Regression baseline model.

This model can operate in both univariate and multivariate modes:
- Univariate: Uses only historical target values
- Multivariate: Uses both historical target values and covariates
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Optional
import matplotlib.pyplot as plt


class LinearRegressionForecast:
    """
    Linear regression forecasting model with univariate and multivariate capabilities.
    
    This model can adapt based on the availability of covariates:
    - If covariates are provided, uses both target history and covariates
    - If covariates are None, uses only target history (univariate mode)
    """
    
    def __init__(self, lookback_window: int = 5, use_covariates: bool = True):
        """
        Initialize the linear regression model.
        
        Args:
            lookback_window: Number of historical points to use as features
            use_covariates: Whether to use covariates when available (multivariate mode)
        """
        self.lookback_window = lookback_window
        self.use_covariates = use_covariates
        self.model = LinearRegression()
        self.feature_dim = None
        self.history_mean = None
        self.history_std = None
        self.covariate_mean = None
        self.covariate_std = None
    
    def forecast(self, history: np.ndarray, covariates: Optional[np.ndarray] = None, 
                 forecast_horizon: Optional[int] = None, timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate forecast using simple linear regression on the whole history.
        
        Args:
            history: Historical target values (shape: [batch_size, history_length])
            covariates: Historical covariate values (shape: [batch_size, history_length, covariate_dim])
            forecast_horizon: Number of steps to forecast
            timestamps: Optional timestamp data (ignored)
            
        Returns:
            Array of forecasted values (shape: [batch_size, forecast_horizon])
        """
        if forecast_horizon is None:
            forecast_horizon = 1
        
        batch_size = history.shape[0]
        forecasts = []
        
        for i in range(batch_size):
            # Extract covariates for this sample if available
            sample_covariates = None
            if covariates is not None and covariates.ndim == 3:
                sample_covariates = covariates[i]
            elif covariates is not None and covariates.ndim == 2:
                # Same covariates for all samples
                sample_covariates = covariates
            
            history_i = history[i]
            
            if len(history_i) < 2:
                # Fallback to mean if insufficient history
                forecast_i = np.full(forecast_horizon, np.mean(history_i))
            else:
                # Handle NaN values in history
                if np.any(np.isnan(history_i)):
                    # Replace NaN values with forward fill, then backward fill
                    history_i = pd.Series(history_i).ffill().bfill().values
                    # If still NaN, replace with mean (handle empty slice case)
                    if np.any(np.isnan(history_i)):
                        # Check if all values are NaN to avoid empty slice warning
                        if np.all(np.isnan(history_i)):
                            # If all values are NaN, use 0 as fallback
                            history_i = np.zeros_like(history_i)
                        else:
                            # Use nanmean only if there are some non-NaN values
                            history_i = np.nan_to_num(history_i, nan=np.nanmean(history_i))
                
                # Handle NaN values in covariates
                if sample_covariates is not None and np.any(np.isnan(sample_covariates)):
                    # Replace NaN values with mean for each covariate column (handle empty slice case)
                    for col in range(sample_covariates.shape[1]):
                        if np.all(np.isnan(sample_covariates[:, col])):
                            sample_covariates[:, col] = 0.0
                        else:
                            sample_covariates[:, col] = np.nan_to_num(sample_covariates[:, col], nan=np.nanmean(sample_covariates[:, col]))

                # Prepare features for regression
                if sample_covariates is not None and len(sample_covariates) > 0:
                    # Use both time indices and covariates
                    time_features = np.arange(len(history_i)).reshape(-1, 1)
                    # Ensure covariates have the same length as history
                    if len(sample_covariates) >= len(history_i):
                        covariate_features = sample_covariates[:len(history_i)]
                    else:
                        # Pad covariates if shorter than history
                        padding = np.zeros((len(history_i) - len(sample_covariates), sample_covariates.shape[1]))
                        covariate_features = np.vstack([sample_covariates, padding])
                    
                    # Combine time and covariate features
                    X = np.hstack([time_features, covariate_features])
                else:
                    # Use only time indices if no covariates
                    X = np.arange(len(history_i)).reshape(-1, 1)
                
                y = history_i  # Target values
                
                # Train linear regression model
                model = LinearRegression()
                model.fit(X, y)
                
                # Generate forecasts
                forecast_i = []
                for j in range(forecast_horizon):
                    # Predict next time step
                    next_time = len(history_i) + j
                    
                    if sample_covariates is not None and len(sample_covariates) > 0:
                        # Use future covariate values if available
                        covariate_idx = len(history_i) + j
                        if covariate_idx < len(sample_covariates):
                            covariate_features = sample_covariates[covariate_idx]
                        else:
                            # Use last available covariate values
                            covariate_features = sample_covariates[-1]
                        
                        # Combine time and covariate features
                        features = np.hstack([[next_time], covariate_features])
                    else:
                        # Use only time index
                        features = [next_time]
                    
                    # Ensure features match training dimensions
                    if len(features) != model.coef_.shape[0]:
                        if len(features) < model.coef_.shape[0]:
                            # Pad with zeros
                            features = np.pad(features, (0, model.coef_.shape[0] - len(features)), 'constant')
                        else:
                            # Truncate
                            features = features[:model.coef_.shape[0]]
                    
                    pred = model.predict([features])[0]
                    forecast_i.append(pred)
                
                forecast_i = np.array(forecast_i)
                
                # Clip extreme predictions that are more than 10x the historical range
                history_range = np.max(history[i]) - np.min(history[i])
                if history_range > 0:  # Only clip if there's actual variation
                    max_allowed = np.max(history[i]) + 10 * history_range
                    min_allowed = np.min(history[i]) - 10 * history_range
                    
                    # Check if clipping is needed
                    if np.any(forecast_i > max_allowed) or np.any(forecast_i < min_allowed):
                        print("Warning: Linear regression forecast contains extreme values, clipping to historical range ±10x")
                        print(f"  Historical range: [{np.min(history[i]):.2e}, {np.max(history[i]):.2e}]")
                        print(f"  Forecast range before clipping: [{np.min(forecast_i):.2e}, {np.max(forecast_i):.2e}]")
                        
                        forecast_i = np.clip(forecast_i, min_allowed, max_allowed)
                        
                        print(f"  Forecast range after clipping: [{np.min(forecast_i):.2e}, {np.max(forecast_i):.2e}]")
            
            forecasts.append(forecast_i)
        
        return np.stack(forecasts, axis=0)  # [batch_size, forecast_horizon]
    
    def plot_linear_fit(self, history: np.ndarray, covariates: Optional[np.ndarray] = None, 
                       save_path: Optional[str] = None):
        """
        Plot the linear regression fit using the existing trained model.
        
        Args:
            history: Historical target values
            covariates: Historical covariate values (optional)
            save_path: Path to save the plot (optional)
        """
        if not hasattr(self, 'model') or self.model is None:
            print("No trained model available for plotting")
            return
        
        if len(history) < 2:
            print("Insufficient history for plotting")
            return
        
        # Generate predictions using the trained model
        # Prepare features the same way as in training
        if covariates is not None and len(covariates) > 0:
            # Use both time indices and covariates
            time_features = np.arange(len(history)).reshape(-1, 1)
            # Ensure covariates have the same length as history
            if len(covariates) >= len(history):
                covariate_features = covariates[:len(history)]
            else:
                # Pad covariates if shorter than history
                padding = np.zeros((len(history) - len(covariates), covariates.shape[1]))
                covariate_features = np.vstack([covariates, padding])
            
            # Combine time and covariate features
            X = np.hstack([time_features, covariate_features])
        else:
            # Use only time indices if no covariates
            X = np.arange(len(history)).reshape(-1, 1)
        
        predictions = self.model.predict(X)
        
        # Create simple plot
        plt.figure(figsize=(10, 6))
        
        # Plot original history
        time_indices = np.arange(len(history))
        plt.plot(time_indices, history, 'b-', label='Original History', linewidth=2)
        
        # Plot fitted values
        plt.plot(time_indices, predictions, 'r--', label='Linear Regression Fit', linewidth=2)
        
        # Add vertical line to separate training and forecast regions
        plt.axvline(x=len(history)-1, color='green', linestyle=':', linewidth=2, 
                   label='End of History')
        
        # Calculate R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(history, predictions)
        
        plt.title(f'Linear Regression Fit (R²: {r2:.4f})')
        plt.xlabel('Time Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        
        # Print diagnostic information
        print(f"\nLinear Regression Diagnostic:")
        print(f"  History length: {len(history)}")
        print(f"  R² score: {r2:.4f}")
        print(f"  History mean: {self.history_mean:.4f}")
        print(f"  History std: {self.history_std:.4f}")
        print(f"  Features used: {X.shape[1]} (time + {'covariates' if covariates is not None and len(covariates) > 0 else 'no covariates'})")
        
        # Check for potential issues
        if r2 < 0.1:
            print(f"  ⚠ Low R² score suggests poor linear fit")