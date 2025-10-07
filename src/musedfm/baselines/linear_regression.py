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
                 forecast_horizon: int = 1) -> np.ndarray:
        """
        Generate forecast using simple linear regression on the whole history.
        
        Args:
            history: Historical target values
            covariates: Historical covariate values (ignored for simplicity)
            forecast_horizon: Number of steps to forecast
            
        Returns:
            Array of forecasted values
        """
        if len(history) < 2:
            # Fallback to mean if insufficient history
            return np.full(forecast_horizon, np.mean(history))
        
        # Handle NaN values in history
        if np.any(np.isnan(history)):
            # Replace NaN values with forward fill, then backward fill
            history = pd.Series(history).ffill().bfill().values
            # If still NaN, replace with mean (handle empty slice case)
            if np.any(np.isnan(history)):
                # Check if all values are NaN to avoid empty slice warning
                if np.all(np.isnan(history)):
                    # If all values are NaN, use 0 as fallback
                    history = np.zeros_like(history)
                else:
                    # Use nanmean only if there are some non-NaN values
                    history = np.nan_to_num(history, nan=np.nanmean(history))
        
        # Handle NaN values in covariates
        if covariates is not None and np.any(np.isnan(covariates)):
            # Replace NaN values with mean for each covariate column (handle empty slice case)
            # Check if any columns are entirely NaN to avoid empty slice warning
            nan_means = np.nanmean(covariates, axis=0)
            # Replace NaN means (from entirely NaN columns) with 0
            nan_means = np.nan_to_num(nan_means, nan=0.0)
            covariates = np.nan_to_num(covariates, nan=nan_means)

        
        # Prepare features for regression
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
        
        y = history  # Target values
        
        # Train linear regression model only if not already trained with same dimensions
        if not hasattr(self, 'model') or self.model is None or not hasattr(self.model, 'coef_') or self.model.coef_.shape[0] != X.shape[1]:
            print(f"Debug: Training model with {X.shape[1]} features")
            self.model = LinearRegression()
            self.model.fit(X, y)
        else:
            print(f"Debug: Reusing existing model with {self.model.coef_.shape[0]} features")
        
        # Store history info for plotting
        self.history_mean = np.mean(history)
        self.history_std = np.std(history)
        self.history_length = len(history)
        
        # Generate forecasts
        forecasts = []
        for i in range(forecast_horizon):
            # Predict next time step
            next_time = len(history) + i
            
            if covariates is not None and len(covariates) > 0:
                # Use future covariate values if available
                covariate_idx = len(history) + i
                if covariate_idx < len(covariates):
                    covariate_features = covariates[covariate_idx]
                else:
                    # Use last available covariate values
                    covariate_features = covariates[-1]
                
                # Combine time and covariate features
                features = np.hstack([[next_time], covariate_features])
            else:
                # Use only time index
                features = [next_time]
            
            # Ensure features match training dimensions
            if len(features) != self.model.coef_.shape[0]:
                if len(features) < self.model.coef_.shape[0]:
                    # Pad with zeros
                    features = np.pad(features, (0, self.model.coef_.shape[0] - len(features)), 'constant')
                else:
                    # Truncate
                    features = features[:self.model.coef_.shape[0]]
            
            pred = self.model.predict([features])[0]
            forecasts.append(pred)
        
        return np.array(forecasts)
    
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
        
        # Debug: Check model dimensions
        print(f"Debug: Model expects {self.model.coef_.shape[0]} features")
        print(f"Debug: History length: {len(history)}")
        print(f"Debug: Covariates shape: {covariates.shape if covariates is not None else 'None'}")
        
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
        
        print(f"Debug: Prepared X shape: {X.shape}")
        
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
