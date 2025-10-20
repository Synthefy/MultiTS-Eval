"""
Evaluation metrics for forecasting models.
"""

import numpy as np
from typing import Dict


def MAPE(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Mean Absolute Percentage Error with robust handling of low variance cases.
    
    Args:
        y_true: Ground truth values (shape: [batch_size, num_values])
        y_pred: Predicted values (shape: [batch_size, num_values])
        
    Returns:
        MAPE values as percentages, one per batch element (shape: [batch_size])
    """
    # Ensure inputs are float type to avoid integer overflow issues
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # Create mask for valid (non-NaN) values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    
    # Check for batch elements with no valid data
    valid_counts = np.sum(valid_mask, axis=1)
    empty_batches = valid_counts == 0        
    
    # Calculate target statistics for robust handling (per batch element)
    target_mean = np.nanmean(y_true, axis=1, where=valid_mask)
    target_std = np.nanstd(y_true, axis=1, where=valid_mask)
    target_max = np.nanmax(y_true, axis=1, where=valid_mask, initial=-np.inf)
    target_min = np.nanmin(y_true, axis=1, where=valid_mask, initial=np.inf)
    target_range = target_max - target_min
    
    # Initialize result array
    mape_per_batch = np.full(y_true.shape[0], np.nan)

    LOW_VARIANCE_THRESHOLD = 1e-3
    MIN_DENOMINATOR = 1e-4  # Minimum denominator to prevent division by very small numbers
    
    # Handle low variance cases (target_std < 1e-3)
    low_variance_mask = target_std < LOW_VARIANCE_THRESHOLD
    low_variance_mask = low_variance_mask & ~empty_batches
    
    if np.any(low_variance_mask):
        # Calculate MAE for low variance cases
        mae_low_var = np.nanmean(np.abs(y_true - y_pred), axis=1, where=valid_mask)
        
        # Constant target case (target_range < 1e-3)
        constant_mask = low_variance_mask & (target_range < LOW_VARIANCE_THRESHOLD)
        if np.any(constant_mask):
            # Use a robust normalization factor to prevent division by very small numbers
            normalization_factor = np.maximum(np.abs(target_mean[constant_mask]), MIN_DENOMINATOR)
            mape_per_batch[constant_mask] = (mae_low_var[constant_mask] / normalization_factor) * 100
        
        # Low variance but not constant case
        low_var_not_constant_mask = low_variance_mask & (target_range >= LOW_VARIANCE_THRESHOLD)
        if np.any(low_var_not_constant_mask):
            # Use range for normalization, but ensure it's not too small
            normalization_factor = np.maximum(target_range[low_var_not_constant_mask], MIN_DENOMINATOR)
            mape_per_batch[low_var_not_constant_mask] = (mae_low_var[low_var_not_constant_mask] / normalization_factor) * 100
    
    # Handle normal cases (target_std >= 1e-3)
    normal_mask = target_std >= LOW_VARIANCE_THRESHOLD
    normal_mask = normal_mask & ~empty_batches
    
    if np.any(normal_mask):
        # Calculate minimum threshold for each batch element
        min_threshold = np.maximum(target_std[normal_mask] * 0.01, MIN_DENOMINATOR)
        
        # Create mask for values above threshold (per batch element)
        threshold_mask = np.zeros_like(y_true, dtype=bool)
        threshold_mask[normal_mask] = np.abs(y_true[normal_mask]) > min_threshold[:, np.newaxis]
        combined_mask = valid_mask & threshold_mask & normal_mask[:, np.newaxis]
        
        # Calculate MAPE for values above threshold
        mape_normal = np.full(np.sum(normal_mask), np.nan)
        for i, batch_idx in enumerate(np.where(normal_mask)[0]):
            batch_combined_mask = combined_mask[batch_idx]
            if np.any(batch_combined_mask):
                # Calculate MAPE with robust denominator handling
                y_true_batch = y_true[batch_idx]
                y_pred_batch = y_pred[batch_idx]
                
                # Ensure denominator is never too small to prevent overflow
                denominator = np.maximum(np.abs(y_true_batch), min_threshold[i])
                mape_values = np.abs((y_true_batch - y_pred_batch) / denominator) * 100
                mape_normal[i] = np.nanmean(mape_values, where=batch_combined_mask)
        
        # Handle cases where all values are too small
        small_values_mask = normal_mask & ~np.any(threshold_mask & valid_mask, axis=1)
        if np.any(small_values_mask):
            mae_small = np.nanmean(np.abs(y_true - y_pred), axis=1, where=valid_mask)
            # Use a robust normalization factor to prevent division by very small numbers
            normalization_factor = np.maximum(np.abs(target_mean[small_values_mask]), MIN_DENOMINATOR)
            mape_per_batch[small_values_mask] = (mae_small[small_values_mask] / normalization_factor) * 100
        
        # Set normal case MAPE values
        normal_with_data_mask = normal_mask & ~small_values_mask
        normal_indices = np.where(normal_with_data_mask)[0]
        for i, batch_idx in enumerate(normal_indices):
            mape_per_batch[batch_idx] = mape_normal[i]
    
    return mape_per_batch


def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Mean Absolute Error.
    
    Args:
        y_true: Ground truth values (shape: [batch_size, num_values])
        y_pred: Predicted values (shape: [batch_size, num_values])
        
    Returns:
        MAE values, one per batch element (shape: [batch_size])
    """
    # Create mask for valid (non-NaN) values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    
    # Calculate MAE for each batch element using vectorized operations
    # Use nanmean to handle cases where all values in a batch element are NaN
    mae_per_batch = np.nanmean(np.abs(y_true - y_pred), axis=1, where=valid_mask)
    
    # Check for batch elements with no valid data
    valid_counts = np.sum(valid_mask, axis=1)
    empty_batches = valid_counts == 0
    
    if np.any(empty_batches):
        print(f"Warning: MAE calculation failed for {np.sum(empty_batches)} batch elements - no valid data points after NaN removal")
        mae_per_batch[empty_batches] = np.nan
    
    return mae_per_batch


def RMSE(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Root Mean Square Error.
    
    Args:
        y_true: Ground truth values (shape: [batch_size, num_values])
        y_pred: Predicted values (shape: [batch_size, num_values])
        
    Returns:
        RMSE values, one per batch element (shape: [batch_size])
    """
    # Create mask for valid (non-NaN) values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    
    # Calculate RMSE for each batch element using vectorized operations
    # Use nanmean to handle cases where all values in a batch element are NaN
    mse_per_batch = np.nanmean((y_true - y_pred) ** 2, axis=1, where=valid_mask)
    rmse_per_batch = np.sqrt(mse_per_batch)
    
    # Check for batch elements with no valid data
    valid_counts = np.sum(valid_mask, axis=1)
    empty_batches = valid_counts == 0
    
    if np.any(empty_batches):
        print(f"Warning: RMSE calculation failed for {np.sum(empty_batches)} batch elements - no valid data points after NaN removal")
        rmse_per_batch[empty_batches] = np.nan
    
    return rmse_per_batch


def NMAE(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Normalized Mean Absolute Error.
    
    Args:
        y_true: Ground truth values (shape: [batch_size, num_values])
        y_pred: Predicted values (shape: [batch_size, num_values])
        
    Returns:
        NMAE values, one per batch element (shape: [batch_size])
    """
    # Create mask for valid (non-NaN) values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    
    # Check for batch elements with no valid data
    valid_counts = np.sum(valid_mask, axis=1)
    empty_batches = valid_counts == 0
    
    if np.any(empty_batches):
        print(f"Warning: NMAE calculation failed for {np.sum(empty_batches)} batch elements - no valid data points after NaN removal")
    

    LOW_VARIANCE_THRESHOLD = 1e-3
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # Calculate mean and std of target signal (per batch element)
    target_mean = np.nanmean(y_true, axis=1, where=valid_mask)
    target_std = np.nanstd(y_true, axis=1, where=valid_mask)
    target_max = np.nanmax(y_true, axis=1, where=valid_mask, initial=-np.inf)
    target_min = np.nanmin(y_true, axis=1, where=valid_mask, initial=np.inf)
    target_range = target_max - target_min
    
    # Initialize result array
    nmae_per_batch = np.full(y_true.shape[0], np.nan)
    
    # Handle low variance targets (target_std < 1e-6)
    low_variance_mask = target_std < LOW_VARIANCE_THRESHOLD
    low_variance_mask = low_variance_mask & ~empty_batches
    
    if np.any(low_variance_mask):
        # Calculate raw MAE for low variance cases
        raw_mae_low_var = np.nanmean(np.abs(y_true - y_pred), axis=1, where=valid_mask)
        
        # Constant target case (target_range < 1e-6)
        constant_mask = low_variance_mask & (target_range < LOW_VARIANCE_THRESHOLD)
        if np.any(constant_mask):
            normalization_factor = np.maximum(np.abs(target_mean[constant_mask]), 1e-6)
            nmae_per_batch[constant_mask] = raw_mae_low_var[constant_mask] / normalization_factor
        
        # Low variance but not constant case
        low_var_not_constant_mask = low_variance_mask & (target_range >= LOW_VARIANCE_THRESHOLD)
        if np.any(low_var_not_constant_mask):
            nmae_per_batch[low_var_not_constant_mask] = raw_mae_low_var[low_var_not_constant_mask] / target_range[low_var_not_constant_mask]
        
        # Cap at 3 standard deviations (3.0 in normalized space)
        nmae_per_batch[low_variance_mask] = np.minimum(nmae_per_batch[low_variance_mask], 3.0)
    
    # Handle normal cases (target_std >= 1e-6)
    normal_mask = target_std >= LOW_VARIANCE_THRESHOLD
    normal_mask = normal_mask & ~empty_batches
    
    if np.any(normal_mask):
        # Use standard deviation normalization
        normalization_factor = target_std[normal_mask]
        target_mean_normal = target_mean[normal_mask]
        
        # Normalize both forecast and target
        y_true_normalized = (y_true[normal_mask] - target_mean_normal[:, np.newaxis]) / normalization_factor[:, np.newaxis]
        y_pred_normalized = (y_pred[normal_mask] - target_mean_normal[:, np.newaxis]) / normalization_factor[:, np.newaxis]
        
        # Calculate MAE of normalized values
        normalized_mae = np.full(np.sum(normal_mask), np.nan)
        for i, batch_idx in enumerate(np.where(normal_mask)[0]):
            batch_valid_mask = valid_mask[batch_idx]
            if np.any(batch_valid_mask):
                normalized_mae[i] = np.nanmean(np.abs(y_true_normalized[i] - y_pred_normalized[i]), where=batch_valid_mask)
        
        # Cap at 3 standard deviations for consistency
        nmae_per_batch[normal_mask] = np.minimum(normalized_mae, 3.0)
    
    return nmae_per_batch


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Evaluate all metrics for given predictions.
    
    Args:
        y_true: Ground truth values (shape: [batch_size, num_values])
        y_pred: Predicted values (shape: [batch_size, num_values])
        
    Returns:
        Dictionary containing metric vectors (one per batch element)
    """
    # Let individual metric functions handle NaN values per batch
    # They will return NaN for batches with no valid data, which will be
    # automatically skipped during aggregation using np.nanmean()
    return {
        'MAPE': MAPE(y_true, y_pred),
        'MAE': MAE(y_true, y_pred),
        'RMSE': RMSE(y_true, y_pred),
        'NMAE': NMAE(y_true, y_pred)
    }
