"""
Evaluation metrics for forecasting models.
"""

import numpy as np
from typing import Dict


def MAPE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error with robust handling of low variance cases.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        MAPE value as a percentage, capped only for low variance cases
    """
    # Remove NaN values from both arrays
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    if len(y_true_clean) == 0:
        print(f"Warning: MAPE calculation failed - no valid data points after NaN removal")
        return np.nan
    
    # Calculate target statistics for robust handling
    target_mean = np.mean(y_true_clean)
    target_std = np.std(y_true_clean)
    target_range = np.max(y_true_clean) - np.min(y_true_clean)
    
    # Handle different cases based on target characteristics
    if target_std < 1e-4:  # More lenient threshold for MAPE
        # Low variance or constant target - use capped MAPE
        if target_range < 1e-4:
            # Constant target - use MAE relative to target magnitude
            raw_mae = np.mean(np.abs(y_true_clean - y_pred_clean))
            normalization_factor = max(abs(target_mean), 1e-6)
            mape = (raw_mae / normalization_factor) * 100
        else:
            # Low variance - use range-based normalization
            raw_mae = np.mean(np.abs(y_true_clean - y_pred_clean))
            mape = (raw_mae / target_range) * 100
        
        # Cap MAPE at 1000% for low variance cases only
        return min(mape, 1000.0)
    else:
        # Normal case - use standard MAPE calculation without capping
        # Avoid division by zero by filtering out very small values
        min_threshold = max(target_std * 0.01, 1e-6)  # 1% of std or minimum threshold
        mask = np.abs(y_true_clean) > min_threshold
        
        if not np.any(mask):
            # All values are too small - use MAE relative to mean
            raw_mae = np.mean(np.abs(y_true_clean - y_pred_clean))
            mape = (raw_mae / max(abs(target_mean), 1e-6)) * 100
        else:
            # Calculate MAPE for values above threshold
            mape = np.mean(np.abs((y_true_clean[mask] - y_pred_clean[mask]) / y_true_clean[mask])) * 100
        
        # No capping for normal cases - allow natural MAPE values
        return mape


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
        NMAE value (MAE of forecast and target after robust normalization)
    """
    # Remove NaN values from both arrays
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]
    
    if len(y_true_clean) == 0:
        print(f"Warning: NMAE calculation failed - no valid data points after NaN removal")
        return np.nan
    
    # Calculate mean and std of target signal
    target_mean = np.mean(y_true_clean)
    target_std = np.std(y_true_clean)
    
    # Handle low variance targets by capping NMAE at 3 standard deviations
    if target_std < 1e-6:
        # For low variance targets, calculate raw MAE and cap at 3 std deviations
        raw_mae = np.mean(np.abs(y_true_clean - y_pred_clean))
        
        # Use target range if available, otherwise use mean magnitude
        target_range = np.max(y_true_clean) - np.min(y_true_clean)
        if target_range < 1e-6:
            # Constant target - use mean magnitude for normalization
            normalization_factor = max(abs(target_mean), 1e-6)
        else:
            # Low variance but not constant - use range
            normalization_factor = target_range
        
        # Calculate normalized MAE
        normalized_mae = raw_mae / normalization_factor
        
        # Cap at 3 standard deviations (3.0 in normalized space)
        return min(normalized_mae, 3.0)
    else:
        # Normal case - use standard deviation normalization
        normalization_factor = target_std
        
        # Normalize both forecast and target
        y_true_normalized = (y_true_clean - target_mean) / normalization_factor
        y_pred_normalized = (y_pred_clean - target_mean) / normalization_factor
        
        # Calculate MAE of normalized values
        normalized_mae = np.mean(np.abs(y_true_normalized - y_pred_normalized))
        
        # Cap at 3 standard deviations for consistency
        return min(normalized_mae, 3.0)


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
