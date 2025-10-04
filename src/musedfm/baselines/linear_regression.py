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
        Generate forecast using linear regression.
        
        Args:
            history: Historical target values
            covariates: Historical covariate values (can be None for univariate mode)
            forecast_horizon: Number of steps to forecast
            
        Returns:
            Array of forecasted values
        """
        if len(history) < self.lookback_window:
            # Fallback to mean if insufficient history
            return np.full(forecast_horizon, np.mean(history))
        
        # Handle NaN values in history
        if np.any(np.isnan(history)):
            # Replace NaN values with forward fill, then backward fill
            history = pd.Series(history).fillna(method='ffill').fillna(method='bfill').values
            # If still NaN, replace with mean
            if np.any(np.isnan(history)):
                history = np.nan_to_num(history, nan=np.nanmean(history))
        
        # Handle NaN values in covariates
        if covariates is not None and np.any(np.isnan(covariates)):
            covariates = np.nan_to_num(covariates, nan=0.0)
        
        # Normalize history
        self.history_mean = np.mean(history)
        self.history_std = np.std(history)
        if self.history_std == 0:
            self.history_std = 1.0
        history_normalized = (history - self.history_mean) / self.history_std
        
        # Normalize covariates if available
        if covariates is not None:
            self.covariate_mean = np.mean(covariates, axis=0)
            self.covariate_std = np.std(covariates, axis=0)
            # Avoid division by zero
            self.covariate_std[self.covariate_std == 0] = 1.0
            covariates_normalized = (covariates - self.covariate_mean) / self.covariate_std
        else:
            covariates_normalized = None
        
        # Determine if we should use covariates
        use_covs = self.use_covariates and covariates_normalized is not None and len(covariates_normalized) > 0
        
        # Prepare features with consistent dimensions
        X = []
        y = []
        
        for i in range(self.lookback_window, len(history_normalized)):
            # Target features: previous lookback_window values
            target_features = history_normalized[i-self.lookback_window:i]
            
            # Covariate features: corresponding covariate values (if available)
            if use_covs and len(covariates_normalized) > i:
                covariate_features = covariates_normalized[i-self.lookback_window:i].flatten()
                features = np.concatenate([target_features, covariate_features])
            else:
                # Pad with zeros if no covariates
                if covariates_normalized is not None and len(covariates_normalized.shape) > 1:
                    covariate_dim = covariates_normalized.shape[1]
                else:
                    covariate_dim = 1
                covariate_features = np.zeros(self.lookback_window * covariate_dim)
                features = np.concatenate([target_features, covariate_features])
            
            X.append(features)
            y.append(history_normalized[i])
        
        if len(X) == 0:
            return np.full(forecast_horizon, np.mean(history))
        
        X = np.array(X)
        y = np.array(y)
        
        # Handle infinite and extreme values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Train the model
        self.model.fit(X, y)
        self.feature_dim = X.shape[1]
        
        # Generate forecasts
        forecasts = []
        current_history = history_normalized[-self.lookback_window:].copy()
        
        for _ in range(forecast_horizon):
            # Prepare features for next prediction
            if use_covs and len(covariates_normalized) >= len(history_normalized) + len(forecasts):
                # Use future covariate values if available
                covariate_idx = len(history_normalized) + len(forecasts)
                if covariate_idx < len(covariates_normalized):
                    covariate_features = covariates_normalized[covariate_idx-self.lookback_window:covariate_idx].flatten()
                    features = np.concatenate([current_history, covariate_features])
                else:
                    # Use last available covariate values
                    last_covariates = covariates_normalized[-self.lookback_window:].flatten()
                    features = np.concatenate([current_history, last_covariates])
            else:
                # Pad with zeros if no covariates
                if covariates_normalized is not None and len(covariates_normalized.shape) > 1:
                    covariate_dim = covariates_normalized.shape[1]
                else:
                    covariate_dim = 1
                covariate_features = np.zeros(self.lookback_window * covariate_dim)
                features = np.concatenate([current_history, covariate_features])
            
            # Ensure features match training dimensions
            if len(features) != self.feature_dim:
                if len(features) < self.feature_dim:
                    features = np.pad(features, (0, self.feature_dim - len(features)), 'constant')
                else:
                    features = features[:self.feature_dim]
            
            # Handle extreme values in features
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Make prediction (in normalized space)
            pred_normalized = self.model.predict([features])[0]
            
            # Handle extreme predictions
            if np.isnan(pred_normalized) or np.isinf(pred_normalized):
                pred_normalized = 0.0  # Mean in normalized space
            
            # Denormalize prediction
            pred = pred_normalized * self.history_std + self.history_mean
            
            # Clip denormalized prediction to reasonable bounds
            pred = np.clip(pred, -1e10, 1e10)
            
            forecasts.append(pred)
            
            # Update history for next prediction (in normalized space)
            current_history = np.roll(current_history, -1)
            current_history[-1] = pred_normalized
        
        return np.array(forecasts)
