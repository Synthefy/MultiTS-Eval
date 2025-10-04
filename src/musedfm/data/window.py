"""
Window class representing a single evaluation unit.
"""

import numpy as np
from typing import Dict, Any, Optional
from ..metrics import evaluate_metrics


class Window:
    """
    Represents a single evaluation unit (history, target, covariates).
    Stores ground truth and submitted forecast.
    """
    
    def __init__(self, history: np.ndarray, target: np.ndarray, covariates: np.ndarray):
        """
        Initialize a window with history, target, and covariates.
        
        Args:
            history: Historical data for forecasting
            target: Ground truth target values
            covariates: Additional covariate data
        """
        self._history = history.copy()
        self._target = target.copy()
        self._covariates = covariates.copy()
        self._forecast: Optional[np.ndarray] = None
        self._evaluation_results: Optional[Dict[str, float]] = None
    
    def history(self) -> np.ndarray:
        """Return the historical data."""
        return self._history.copy()
    
    def target(self) -> np.ndarray:
        """Return the target values."""
        return self._target.copy()
    
    def covariates(self) -> np.ndarray:
        """Return the covariate data."""
        return self._covariates.copy()
    
    def submit_forecast(self, forecast: np.ndarray, univariate: bool = False) -> None:
        """
        Store forecast and trigger evaluation once for this window.
        
        Args:
            forecast: Predicted values
            univariate: Whether this is a univariate forecast (default: False)
        """
        self._forecast = forecast.copy()
        self._is_univariate = univariate
        # Trigger evaluation immediately
        self._evaluation_results = evaluate_metrics(self._target, self._forecast)
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run metrics and return cached results.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self._evaluation_results is None:
            raise ValueError("No forecast submitted yet. Call submit_forecast() first.")
        
        results = self._evaluation_results.copy()
        results['univariate'] = self._is_univariate
        return results
    
    @property
    def has_forecast(self) -> bool:
        """Check if forecast has been submitted."""
        return self._forecast is not None
    
    @property
    def is_univariate(self) -> bool:
        """Check if the forecast is univariate."""
        return getattr(self, '_is_univariate', False)
