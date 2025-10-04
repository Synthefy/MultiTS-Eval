"""
Evaluation metrics for forecasting models.
"""

import numpy as np
from typing import Dict


def MAPE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        MAPE value as a percentage
    """
    # Remove NaN values from both arrays
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    if len(y_true_clean) == 0:
        print(f"Warning: MAPE calculation failed - no valid data points after NaN removal")
        return np.nan
    
    # Avoid division by zero
    mask = y_true_clean != 0
    if not np.any(mask):
        return 0.0
    
    return np.mean(np.abs((y_true_clean[mask] - y_pred_clean[mask]) / y_true_clean[mask])) * 100


def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    # Remove NaN values from both arrays
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    if len(y_true_clean) == 0:
        print(f"Warning: MAE calculation failed - no valid data points after NaN removal")
        return np.nan
    
    return np.mean(np.abs(y_true_clean - y_pred_clean))


def RMSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Square Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    # Remove NaN values from both arrays
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    if len(y_true_clean) == 0:
        print(f"Warning: RMSE calculation failed - no valid data points after NaN removal")
        return np.nan
    
    return np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2))


def NMAE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalized Mean Absolute Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        NMAE value (MAE normalized by the mean of true values)
    """
    # Remove NaN values from both arrays
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    if len(y_true_clean) == 0:
        print(f"Warning: NMAE calculation failed - no valid data points after NaN removal")
        return np.nan
    
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    mean_true = np.mean(y_true_clean)
    
    # Avoid division by zero
    if mean_true == 0:
        return 0.0
    
    return mae / mean_true


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate all metrics for given predictions.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing all metric values
    """
    return {
        'MAPE': MAPE(y_true, y_pred),
        'MAE': MAE(y_true, y_pred),
        'RMSE': RMSE(y_true, y_pred),
        'NMAE': NMAE(y_true, y_pred)
    }
