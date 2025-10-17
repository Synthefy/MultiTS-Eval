"""
Utility functions for baseline forecasting models.
"""

import numpy as np
from typing import Tuple


def handle_nans(data: np.ndarray, method: str = "zero") -> np.ndarray:
    """
    Handle NaN values in data using specified method.
    
    Args:
        data: Input data array
        method: Method to handle NaNs ("zero", "mean", "forward_fill")
        
    Returns:
        Data with NaNs handled
    """
    if method == "zero":
        return np.nan_to_num(data, nan=0.0)
    elif method == "mean":
        if data.ndim == 1:
            mean_val = np.nanmean(data)
            return np.nan_to_num(data, nan=mean_val)
        else:
            # For multi-dimensional arrays, use mean along the last axis
            mean_vals = np.nanmean(data, axis=-1, keepdims=True)
            return np.nan_to_num(data, nan=mean_vals)
    elif method == "forward_fill":
        # Forward fill NaNs
        result = data.copy()
        for i in range(1, result.shape[-1]):
            mask = np.isnan(result[..., i])
            result[..., i] = np.where(mask, result[..., i-1], result[..., i])
        # Handle any remaining NaNs at the beginning
        result = np.nan_to_num(result, nan=0.0)
        return result
    else:
        raise ValueError(f"Unknown NaN handling method: {method}")


def standard_normalize(data: np.ndarray, axis: int = 1, keepdims: bool = True, epsilon: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard normalization of data along specified axis.
    
    Args:
        data: Input data array
        axis: Axis along which to normalize
        keepdims: Whether to keep dimensions
        epsilon: Small value to avoid division by zero
        
    Returns:
        Tuple of (normalized_data, mean, std)
    """
    mean = np.mean(data, axis=axis, keepdims=keepdims)
    std = np.std(data, axis=axis, keepdims=keepdims)
    
    # Avoid division by zero
    std = np.maximum(std, epsilon)
    
    normalized_data = (data - mean) / std
    
    return normalized_data, mean, std


def standard_denormalize(normalized_data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Denormalize data using mean and std.
    
    Args:
        normalized_data: Normalized data
        mean: Mean used for normalization
        std: Standard deviation used for normalization
        
    Returns:
        Denormalized data
    """
    return normalized_data * std + mean