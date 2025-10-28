"""
Debugging functionality for MUSEval examples.

This module contains functions for:
- NaN value tracking, reporting and statistics
- High MAPE detection and plotting
- Debug output and diagnostics
"""

import numpy as np
from typing import Dict, Any, List, Optional, Set
from museval.plotting import plot_window_forecasts


def _initialize_nan_tracking() -> Dict[str, int]:
    """Initialize NaN tracking statistics for a dataset."""
    return {
        'total_windows': 0,
        'windows_with_nan_history': 0,
        'windows_with_nan_target': 0,
        'windows_with_nan_covariates': 0,
        'windows_with_any_nan': 0,
        'history_nan_count': 0,
        'target_nan_count': 0,
        'covariates_nan_count': 0
    }


def _update_nan_tracking(nan_stats: Dict[str, int], window_nan_stats: Dict[str, Any]) -> None:
    """Update NaN tracking statistics with window-level data."""
    nan_stats['total_windows'] += 1
    
    if window_nan_stats['history_nans'] > 0:
        nan_stats['windows_with_nan_history'] += 1
        nan_stats['history_nan_count'] += window_nan_stats['history_nans']
    
    if window_nan_stats['target_nans'] > 0:
        nan_stats['windows_with_nan_target'] += 1
        nan_stats['target_nan_count'] += window_nan_stats['target_nans']
    
    if window_nan_stats['covariates_nans'] > 0:
        nan_stats['windows_with_nan_covariates'] += 1
        nan_stats['covariates_nan_count'] += window_nan_stats['covariates_nans']
    
    if window_nan_stats['any_nans']:
        nan_stats['windows_with_any_nan'] += 1


def _check_window_nan_values(window) -> Dict[str, Any]:
    """Check for NaN values in a single window and return statistics."""
    # Check history for NaN values
    history = window.history()
    history_nans = np.isnan(history).sum()
    
    # Check target for NaN values
    target = window.target()
    target_nans = np.isnan(target).sum()
    
    # Check covariates for NaN values
    covariates = window.covariates()
    covariates_nans = np.isnan(covariates).sum()
    
    return {
        'history_nans': history_nans,
        'target_nans': target_nans,
        'covariates_nans': covariates_nans,
        'any_nans': history_nans > 0 or target_nans > 0 or covariates_nans > 0
    }


def _report_nan_statistics(nan_stats: Dict[str, int]) -> None:
    """Report NaN statistics for a dataset."""
    if nan_stats['windows_with_any_nan'] > 0:
        print("  ⚠ NaN values detected:")
        print(f"    Windows with NaN: {nan_stats['windows_with_any_nan']}/{nan_stats['total_windows']}")
        if nan_stats['windows_with_nan_history'] > 0:
            print(f"    History NaN: {nan_stats['windows_with_nan_history']} windows, {nan_stats['history_nan_count']} values")
        if nan_stats['windows_with_nan_target'] > 0:
            print(f"    Target NaN: {nan_stats['windows_with_nan_target']} windows, {nan_stats['target_nan_count']} values")
        if nan_stats['windows_with_nan_covariates'] > 0:
            print(f"    Covariates NaN: {nan_stats['windows_with_nan_covariates']} windows, {nan_stats['covariates_nan_count']} values")
    else:
        print(f"  ✓ No NaN values detected in {nan_stats['total_windows']} windows")


def plot_high_mape_windows(model_name: str, dataset_name: str, dataset, model: Dict, 
                          dataset_avg_metrics: Dict[str, float], max_windows: int = 3, save_path: str = "",
                          forecast_type: str = "multivariate") -> None:
    """Plot sample windows for models with high MAPE values.
    
    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        dataset: Dataset object to iterate through
        model: Model dictionary containing model and univariate flag
        dataset_avg_metrics: Dictionary containing average metrics
        max_windows: Maximum number of windows to plot
        save_path: Path to save the plots
        forecast_type: Type of forecast to generate ("multivariate" or "univariate")
    """
    if dataset_avg_metrics.get('MAPE', np.nan) > 1000 or np.isnan(dataset_avg_metrics.get('MAPE', np.nan)):
        print(f"  🔍 High MAPE detected for {model_name} on {dataset_name} - plotting sample windows to /{save_path}/{model_name}_{dataset_name}.png")
        # Plot first few windows to see what's happening
        for plot_idx, window in enumerate(dataset):
            if plot_idx >= max_windows:  # Only plot first few windows
                break
            
            # Get the forecast for this specific window
            target_length = window.target().shape[1]  # Get forecast horizon from batched target
            
            if forecast_type == "univariate":
                sample_forecast = model["model"].forecast(window.history(), None, target_length, window.timestamps())
            else:  # multivariate
                if model["univariate"]:
                    sample_forecast = model["model"].forecast(window.history(), None, target_length, window.timestamps())
                else:
                    sample_forecast = model["model"].forecast(window.history(), window.covariates(), target_length, window.timestamps())
            
            if sample_forecast is not None:
                plot_window_forecasts(
                    window, 
                    {model_name: sample_forecast}, 
                    f"{model_name} - {dataset_name} - Window {plot_idx+1}", 
                    figsize=(12, 6), 
                    save_path=f"{save_path}/{model_name}_{dataset_name}_window_{plot_idx+1}.png"
                )


def debug_model_performance(model_name: str, dataset_avg_metrics: Dict[str, float], 
                          model_dataset_windows: Dict[str, int], model_elapsed_time: float) -> None:
    """Print debug information for model performance."""
    mape_value = dataset_avg_metrics.get('MAPE', 'N/A')
    nmae_value = dataset_avg_metrics.get('NMAE', 'N/A')
    if isinstance(mape_value, (int, float)) and not np.isnan(mape_value):
        mape_str = f"{mape_value:.2f}%"
    else:   
        mape_str = "N/A"
    if isinstance(nmae_value, (int, float)) and not np.isnan(nmae_value):
        nmae_str = f"{nmae_value:.4f}"
    else:
        nmae_str = "N/A"
    
    print(f"    {model_name}: {model_dataset_windows[model_name]} windows in {model_elapsed_time:.2f}s with MAPE {mape_str} and NMAE {nmae_str}")


def debug_univariate_performance(univariate_model_name: str, univariate_avg_metrics: Dict[str, float],
                                model_dataset_windows: Dict[str, int], model_elapsed_time: float, model_name: str) -> None:
    """Print debug information for univariate model performance."""
    mape_value = univariate_avg_metrics.get('MAPE', 'N/A')
    if isinstance(mape_value, (int, float)) and not np.isnan(mape_value):
        mape_str = f"{mape_value:.2f}%"
    else:
        mape_str = "N/A"
    
    print(f"    {univariate_model_name}: {model_dataset_windows[model_name]} windows in {model_elapsed_time:.2f}s with MAPE {mape_str}")

def debug_model_summary(model_name: str, results: Dict[str, Any], models: Dict[str, Any]) -> None:
    """Print debug summary for a model."""
    overall_avg_metrics = results[model_name]['metrics']
    
    # Determine model type for display
    if model_name.endswith('_univariate'):
        model_type = 'Univariate'
    else:
        model_type = 'Univariate' if models[model_name]['univariate'] else 'Multivariate'
    
    print(f"\n{model_name} Summary:")
    print(f"  Total windows: {results[model_name]['windows']}")
    print(f"  Total time: {results[model_name]['time']:.2f}s")
    print(f"  Model type: {model_type}")
    if overall_avg_metrics:
        print(f"  Average MAPE: {overall_avg_metrics['MAPE']:.2f}%")
        print(f"  Average MAE: {overall_avg_metrics['MAE']:.4f}")
        print(f"  Average RMSE: {overall_avg_metrics['RMSE']:.4f}")
        print(f"  Average NMAE: {overall_avg_metrics['NMAE']:.4f}")
    
    # Display category and domain results
    for level_name, level_data in [('Category', results[model_name]['category_results']), 
                                  ('Domain', results[model_name]['domain_results'])]:
        if level_data:
            print(f"  {level_name} Results:")
            for name, data in level_data.items():
                metrics = data['metrics']
                print(f"    {name}: {data['dataset_count']} datasets, {data['window_count']} windows")
                if metrics:
                    print(f"      MAPE: {metrics['MAPE']:.2f}%, MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, NMAE: {metrics['NMAE']:.4f}")


def create_individual_window(window, batch_idx: int):
    """
    Create a Window object for a single batch item from a batched window.
    
    Args:
        window: Original batched Window object
        batch_idx: Index of the batch item to extract
        
    Returns:
        Window object containing only the specified batch item
    """
    from museval.data.window import Window
    
    # Extract individual batch items
    individual_history = window.history()[batch_idx:batch_idx+1]  # Keep batch dimension
    individual_target = window.target()[batch_idx:batch_idx+1]   # Keep batch dimension
    individual_covariates = window.covariates()[batch_idx:batch_idx+1]  # Keep batch dimension
    
    # Extract timestamps if available
    individual_timestamps = None
    if window.timestamps() is not None:
        individual_timestamps = window.timestamps()[batch_idx:batch_idx+1]  # Keep batch dimension
    
    return Window(individual_history, individual_target, individual_covariates, individual_timestamps)


def debug_low_variance_window(window, model_name: str, dataset_name: str, window_idx: int, 
                             variance_threshold: float = 1e-4, save_path: str = "", dataset=None) -> bool:
    """
    Debug function to check if a window has low variance targets and create plots.
    Works on individual windows (not batched).
    
    Args:
        window: Window object containing target data (should be individual window, not batched)
        model_name: Name of the model being evaluated
        dataset_name: Name of the dataset
        window_idx: Index of the window in the dataset
        variance_threshold: Threshold below which targets are considered low variance
        save_path: Directory to save plots
        dataset: Dataset object to get file path information
        
    Returns:
        True if this window is low variance, False otherwise
    """
    target = window.target()
    target_std = np.std(target)
    target_range = np.max(target) - np.min(target)
    target_mean = np.mean(target)
    
    is_low_variance = target_std < variance_threshold
    
    if is_low_variance:
        # Get file path if dataset is provided
        file_info = ""
        if dataset and hasattr(dataset, '_parquet_files') and hasattr(dataset, '_current_parquet_index'):
            if dataset._current_parquet_index < len(dataset._parquet_files):
                current_file = dataset._parquet_files[dataset._current_parquet_index]
                file_info = f" (file: {current_file})"
            else:
                file_info = f" (file: {dataset.data_path})"
        elif dataset and hasattr(dataset, 'data_path'):
            file_info = f" (file: {dataset.data_path})"
        
        print(f"  🔍 LOW VARIANCE WINDOW: {model_name} - {dataset_name} - Window {window_idx} (std: {target_std:.2e}){file_info}")
        
        # Create plot for this low variance window
        if save_path:
            plot_low_variance_window(window, model_name, dataset_name, window_idx, 
                                   target_std, target_range, target_mean, save_path)
        
        return True
    
    return False


def debug_low_variance_summary(low_variance_windows: Dict[str, List], variance_threshold: float = 1e-4) -> None:
    """
    Print summary of low variance windows found during evaluation.
    
    Args:
        low_variance_windows: Dictionary mapping model names to lists of low variance window info
        variance_threshold: Threshold used for low variance detection
    """
    if not low_variance_windows:
        return
        
    print("\n" + "="*80)
    print("LOW VARIANCE TARGETS SUMMARY")
    print("="*80)
    
    total_windows = sum(len(windows) for windows in low_variance_windows.values())
    print(f"Total low variance windows found: {total_windows}")
    print(f"Variance threshold: {variance_threshold}")
    
    print(f"\nDatasets with low variance targets:")
    # Extract dataset information from window data
    dataset_counts = {}
    for model_name, windows in low_variance_windows.items():
        for window_info in windows:
            dataset_name = window_info.get('dataset_name', 'unknown')
            if dataset_name not in dataset_counts:
                dataset_counts[dataset_name] = 0
            dataset_counts[dataset_name] += 1
    
    for dataset_name, count in sorted(dataset_counts.items()):
        print(f"  {dataset_name}: {count} windows")
    
    print(f"\nModels with low variance targets:")
    for model_name, windows in low_variance_windows.items():
        if windows:
            print(f"  {model_name}: {len(windows)} windows")
    
    # Show file paths for low variance windows
    print(f"\nFile paths containing low variance windows:")
    file_paths = set()
    for windows in low_variance_windows.values():
        for window_info in windows:
            if 'file_path' in window_info and window_info['file_path'] != "unknown":
                file_paths.add(window_info['file_path'])
    
    for file_path in sorted(file_paths):
        print(f"  {file_path}")
    
    # Analyze patterns
    constant_count = sum(1 for windows in low_variance_windows.values() 
                        for w in windows if w.get('is_constant', False))
    near_zero_count = sum(1 for windows in low_variance_windows.values() 
                         for w in windows if w.get('is_near_zero', False))
    
    print(f"\nPattern analysis:")
    print(f"  Constant targets (range < 1e-6): {constant_count}")
    print(f"  Near-zero targets (mean < 1e-3): {near_zero_count}")
    
    print("="*80)


def plot_low_variance_window(window, model_name: str, dataset_name: str, window_idx: int,
                           target_std: float, target_range: float, target_mean: float, 
                           save_path: str) -> None:
    """
    Create a detailed plot for a low variance window.
    
    Args:
        window: Window object containing target data
        model_name: Name of the model
        dataset_name: Name of the dataset
        window_idx: Index of the window
        target_std: Standard deviation of target
        target_range: Range of target values
        target_mean: Mean of target values
        save_path: Directory to save the plot
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to prevent hanging
    import matplotlib.pyplot as plt
    import os
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Low Variance Window Analysis\n{dataset_name} - {model_name} - Window {window_idx}', 
                 fontsize=14, fontweight='bold')
    
    # Get window data
    history = window.history()
    target = window.target()
    covariates = window.covariates()
    
    # Handle batched data - take first sample from batch
    if target.ndim == 2:
        target = target[0]  # [batch_size, forecast_horizon] -> [forecast_horizon]
    if history.ndim == 2:
        history = history[0]  # [batch_size, history_length] -> [history_length]
    
    # Plot 1: Target values over time
    axes[0, 0].plot(target, 'b-o', linewidth=2, markersize=6)
    axes[0, 0].axhline(y=target_mean, color='r', linestyle='--', alpha=0.7, label=f'Mean: {target_mean:.6f}')
    axes[0, 0].set_title('Target Values')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Target Value')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Target distribution (histogram)
    axes[0, 1].hist(target, bins=min(len(target), 10), alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(x=target_mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {target_mean:.6f}')
    axes[0, 1].set_title('Target Distribution')
    axes[0, 1].set_xlabel('Target Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: History vs Target comparison
    history_indices = np.arange(len(history))
    target_indices = np.arange(len(history), len(history) + len(target))
    
    axes[1, 0].plot(history_indices, history, 'g-o', linewidth=2, markersize=4, label='History')
    axes[1, 0].plot(target_indices, target, 'b-o', linewidth=2, markersize=6, label='Target')
    axes[1, 0].axvline(x=len(history)-0.5, color='k', linestyle=':', alpha=0.7, label='History/Target Split')
    axes[1, 0].set_title('History vs Target')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Statistics summary
    axes[1, 1].axis('off')
    stats_text = f"""Statistics Summary:
    
Target Statistics:
• Mean: {target_mean:.8f}
• Std: {target_std:.8f}
• Range: {target_range:.8f}
• Min: {np.min(target):.8f}
• Max: {np.max(target):.8f}

Classification:
• Low Variance: {target_std < 1e-4}
• Constant: {target_range < 1e-6}
• Near Zero: {abs(target_mean) < 1e-3}

Window Info:
• History Length: {len(history)}
• Target Length: {len(target)}
• Covariates Shape: {covariates.shape if covariates is not None else 'None'}
"""
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot in organized folder structure
    low_variance_dir = os.path.join(save_path, "low_variance", dataset_name)
    os.makedirs(low_variance_dir, exist_ok=True)
    filename = f"{model_name}_window_{window_idx}_low_variance.png"
    filepath = os.path.join(low_variance_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    📊 Low variance window plot saved: {filepath}")


def plot_low_variance_dataset_summary(dataset_name: str, low_variance_windows: List[Dict], 
                                    save_path: str) -> None:
    """
    Create a summary plot for all low variance windows in a dataset.
    
    Args:
        dataset_name: Name of the dataset
        low_variance_windows: List of low variance window information
        save_path: Directory to save the plot
    """
    if not low_variance_windows:
        return
        
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to prevent hanging
    import matplotlib.pyplot as plt
    import os
    
    # Extract data and group by file path
    file_paths = set()
    for w in low_variance_windows:
        if 'file_path' in w and w['file_path'] != "unknown":
            file_paths.add(w['file_path'])
    
    # Create title with file path information
    if file_paths:
        file_info = f" (Files: {', '.join(sorted(file_paths))})"
    else:
        file_info = " (File paths unknown)"
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Low Variance Windows Summary - {dataset_name}{file_info}', 
                 fontsize=14, fontweight='bold')
    
    # Extract data
    window_indices = [w['window_idx'] for w in low_variance_windows]
    target_stds = [w['target_std'] for w in low_variance_windows]
    target_ranges = [w['target_range'] for w in low_variance_windows]
    target_means = [w['target_mean'] for w in low_variance_windows]
    is_constant = [w['is_constant'] for w in low_variance_windows]
    is_near_zero = [w['is_near_zero'] for w in low_variance_windows]
    
    # Get file paths for coloring
    file_paths_list = [w.get('file_path', 'unknown') for w in low_variance_windows]
    
    # Plot 1: Target std vs window index
    # Use different colors for different file paths
    unique_files = list(set(file_paths_list))
    file_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    color_map = {file_path: file_colors[i % len(file_colors)] for i, file_path in enumerate(unique_files)}
    colors = [color_map[fp] for fp in file_paths_list]
    
    axes[0, 0].scatter(window_indices, target_stds, c=colors, alpha=0.7, s=50)
    axes[0, 0].axhline(y=1e-4, color='orange', linestyle='--', alpha=0.7, label='Threshold (1e-4)')
    axes[0, 0].set_title('Target Standard Deviation by Window')
    axes[0, 0].set_xlabel('Window Index')
    axes[0, 0].set_ylabel('Target Std')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add legend for file paths
    if len(unique_files) > 1:
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[fp], 
                                     markersize=8, label=fp) for fp in unique_files]
        axes[0, 0].legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Plot 2: Target range vs window index
    axes[0, 1].scatter(window_indices, target_ranges, c=colors, alpha=0.7, s=50)
    axes[0, 1].axhline(y=1e-6, color='orange', linestyle='--', alpha=0.7, label='Constant Threshold (1e-6)')
    axes[0, 1].set_title('Target Range by Window')
    axes[0, 1].set_xlabel('Window Index')
    axes[0, 1].set_ylabel('Target Range')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Target mean vs window index
    axes[1, 0].scatter(window_indices, target_means, c=colors, alpha=0.7, s=50)
    axes[1, 0].axhline(y=1e-3, color='orange', linestyle='--', alpha=0.7, label='Near-zero Threshold (1e-3)')
    axes[1, 0].axhline(y=-1e-3, color='orange', linestyle='--', alpha=0.7)
    axes[1, 0].set_title('Target Mean by Window')
    axes[1, 0].set_xlabel('Window Index')
    axes[1, 0].set_ylabel('Target Mean')
    axes[1, 0].set_yscale('symlog')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    axes[1, 1].axis('off')
    
    constant_count = sum(is_constant)
    near_zero_count = sum(is_near_zero)
    total_windows = len(low_variance_windows)
    
    # Count windows by file path
    file_counts = {}
    for fp in file_paths_list:
        file_counts[fp] = file_counts.get(fp, 0) + 1
    
    file_info = "\n".join([f"• {fp}: {count} windows" for fp, count in sorted(file_counts.items())])
    
    summary_text = f"""Dataset Summary: {dataset_name}

Total Low Variance Windows: {total_windows}

File Distribution:
{file_info}

Classification:
• Constant Targets: {constant_count} ({constant_count/total_windows*100:.1f}%)
• Near-zero Targets: {near_zero_count} ({near_zero_count/total_windows*100:.1f}%)

Statistics:
• Min Std: {min(target_stds):.2e}
• Max Std: {max(target_stds):.2e}
• Mean Std: {np.mean(target_stds):.2e}

• Min Range: {min(target_ranges):.2e}
• Max Range: {max(target_ranges):.2e}
• Mean Range: {np.mean(target_ranges):.2e}

• Min Mean: {min(target_means):.2e}
• Max Mean: {max(target_means):.2e}
• Mean of Means: {np.mean(target_means):.2e}
"""
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot in organized folder structure
    low_variance_dir = os.path.join(save_path, "low_variance", dataset_name)
    os.makedirs(low_variance_dir, exist_ok=True)
    filename = f"{dataset_name}_low_variance_summary.png"
    filepath = os.path.join(low_variance_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Dataset summary plot saved: {filepath}")


def plot_parquet_file_target_variable(dataset_name: str, dataset, parquet_file_path: str, 
                                     low_variance_window_idx: int, save_path: str) -> None:
    """
    Create a plot of the entire target variable from a specific parquet file.
    
    Args:
        dataset_name: Name of the dataset
        dataset: Dataset object containing all windows
        parquet_file_path: Full path to the parquet file
        low_variance_window_idx: Index of the low variance window in that file
        save_path: Directory to save the plot
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to prevent hanging
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    
    # Find the parquet file in the dataset
    parquet_file = None
    for pf in dataset._parquet_files:
        if str(pf) == parquet_file_path:
            parquet_file = pf
            break
    
    if parquet_file is None:
        print(f"Warning: Could not find parquet file {parquet_file_path}")
        return
    
    # Load the parquet file
    try:
        print(f"    📊 Loading parquet file: {os.path.basename(parquet_file_path)}")
        df = pd.read_parquet(parquet_file)
        print(f"    📊 Loaded {len(df)} rows from parquet file")
    except Exception as e:
        print(f"Error loading parquet file {parquet_file_path}: {e}")
        return
    
    # Get target columns
    target_cols = dataset.column_config.get('target_cols', [])
    if not target_cols:
        print(f"Warning: No target columns found for dataset {dataset_name}")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(target_cols), 1, figsize=(16, 6 * len(target_cols)))
    if len(target_cols) == 1:
        axes = [axes]
    
    fig.suptitle(f'Target Variable Analysis - {dataset_name}\nFile: {os.path.basename(parquet_file_path)}\nLow Variance Window: {low_variance_window_idx}', 
                 fontsize=14, fontweight='bold')
    
    # Calculate window boundaries for the low variance window
    history_length = dataset.history_length
    forecast_horizon = dataset.forecast_horizon
    stride = dataset.stride
    
    # Estimate window position (this is approximate since we don't have exact window mapping)
    window_start = low_variance_window_idx * stride
    window_end = window_start + history_length + forecast_horizon
    
    for i, target_col in enumerate(target_cols):
        if target_col not in df.columns:
            continue
            
        ax = axes[i]
        
        # Plot the entire target variable
        ax.plot(df.index, df[target_col], 'b-', alpha=0.7, linewidth=1, label='Target Values')
        
        # Highlight the low variance window region
        if window_start < len(df) and window_end <= len(df):
            ax.axvspan(window_start, window_end, alpha=0.3, color='red', label=f'Low Variance Window {low_variance_window_idx}')
            
            # Mark the specific window data points
            window_data = df[target_col].iloc[window_start:window_end]
            ax.plot(window_data.index, window_data.values, 'ro', markersize=4, alpha=0.8)
        
        # Add statistics
        target_mean = df[target_col].mean()
        target_std = df[target_col].std()
        target_min = df[target_col].min()
        target_max = df[target_col].max()
        
        ax.set_title(f'{target_col}\nMean: {target_mean:.4f}, Std: {target_std:.4f}, Range: [{target_min:.4f}, {target_max:.4f}]')
        ax.set_xlabel('Data Point Index')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add horizontal lines for mean and std
        ax.axhline(y=target_mean, color='green', linestyle='--', alpha=0.7, label=f'Mean: {target_mean:.4f}')
        ax.axhline(y=target_mean + target_std, color='orange', linestyle='--', alpha=0.7, label=f'+1σ: {target_mean + target_std:.4f}')
        ax.axhline(y=target_mean - target_std, color='orange', linestyle='--', alpha=0.7, label=f'-1σ: {target_mean - target_std:.4f}')
    
    plt.tight_layout()
    
    # Save the plot
    print(f"    📊 Creating target variable plot...")
    os.makedirs(save_path, exist_ok=True)
    filename = f"{dataset_name}_target_variable_{os.path.basename(parquet_file_path).replace('.parquet', '')}_window_{low_variance_window_idx}.png"
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Target variable plot saved: {filepath}")


def plot_parquet_file_overview(dataset_name: str, dataset, low_variance_windows: List[Dict], 
                             save_path: str) -> None:
    """
    Create an overview plot of the entire parquet file showing where low variance windows occur.
    
    Args:
        dataset_name: Name of the dataset
        dataset: Dataset object containing all windows
        low_variance_windows: List of low variance window information
        save_path: Directory to save the plot
    """
    if not low_variance_windows:
        return
        
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to prevent hanging
    import matplotlib.pyplot as plt
    import os
    
    # Get file path information
    file_path = dataset.data_path.name if hasattr(dataset, 'data_path') else "unknown"
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(f'Parquet File Overview - {dataset_name} (File: {file_path})', 
                 fontsize=14, fontweight='bold')
    
    # Collect data from all windows in the dataset
    all_target_means = []
    all_target_stds = []
    all_target_ranges = []
    window_indices = []
    low_variance_indices = set(w['window_idx'] for w in low_variance_windows)
    
    # Sample windows to avoid memory issues (max 1000 windows)
    max_windows_to_sample = min(1000, len(dataset))
    step = max(1, len(dataset) // max_windows_to_sample)
    
    for i, window in enumerate(dataset):
        if i % step == 0:  # Sample every nth window
            target = window.target()
            all_target_means.append(np.mean(target))
            all_target_stds.append(np.std(target))
            all_target_ranges.append(np.max(target) - np.min(target))
            window_indices.append(i)
    
    # Plot 1: Target standard deviation across all windows
    colors = ['red' if i in low_variance_indices else 'blue' for i in window_indices]
    axes[0].scatter(window_indices, all_target_stds, c=colors, alpha=0.6, s=20)
    axes[0].axhline(y=1e-4, color='orange', linestyle='--', alpha=0.7, label='Low Variance Threshold (1e-4)')
    axes[0].set_title('Target Standard Deviation Across All Windows')
    axes[0].set_xlabel('Window Index')
    axes[0].set_ylabel('Target Std')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add annotations for low variance windows
    for window_info in low_variance_windows:
        idx = window_info['window_idx']
        std = window_info['target_std']
        if idx in window_indices:
            axes[0].annotate(f'W{idx}', (idx, std), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8, color='red')
    
    # Plot 2: Target range across all windows
    colors = ['red' if i in low_variance_indices else 'blue' for i in window_indices]
    axes[1].scatter(window_indices, all_target_ranges, c=colors, alpha=0.6, s=20)
    axes[1].axhline(y=1e-6, color='orange', linestyle='--', alpha=0.7, label='Constant Threshold (1e-6)')
    axes[1].set_title('Target Range Across All Windows')
    axes[1].set_xlabel('Window Index')
    axes[1].set_ylabel('Target Range')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add annotations for low variance windows
    for window_info in low_variance_windows:
        idx = window_info['window_idx']
        range_val = window_info['target_range']
        if idx in window_indices:
            axes[1].annotate(f'W{idx}', (idx, range_val), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8, color='red')
    
    # Add summary text
    total_windows = len(dataset)
    low_variance_count = len(low_variance_windows)
    percentage = (low_variance_count / total_windows) * 100
    
    summary_text = f"""Parquet File Summary:
• Total Windows: {total_windows:,}
• Low Variance Windows: {low_variance_count} ({percentage:.2f}%)
• Sampled Windows: {len(window_indices):,}
• Red dots = Low variance windows
• Blue dots = Normal variance windows"""
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save the plot
    os.makedirs(save_path, exist_ok=True)
    filename = f"{dataset_name}_parquet_overview.png"
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Parquet file overview plot saved: {filepath}")


def print_final_low_variance_statistics(all_dataset_stats: Dict[str, Dict], total_windows_processed: Optional[int] = None, total_files_processed: Optional[int] = None, dataset_seen_files: Dict[str, Set[str]] = None) -> None:
    """
    Print a final comprehensive statistics table for low variance windows across all datasets.
    
    Args:
        all_dataset_stats: Dictionary mapping dataset names to their low variance statistics
        total_windows_processed: Total number of windows processed across all datasets
        total_files_processed: Total number of files processed across all datasets
    """
    if not all_dataset_stats:
        return
    
    print("\n" + "="*100)
    print("FINAL LOW VARIANCE STATISTICS SUMMARY")
    print("="*100)
    
    # Calculate totals
    total_datasets = len(all_dataset_stats)
    total_low_variance_windows = sum(stats['total_low_variance_windows'] for stats in all_dataset_stats.values())
    total_low_variance_files = sum(stats['total_low_variance_files'] for stats in all_dataset_stats.values())
    
    # Print header
    print(f"{'Dataset':<20} {'Windows':<15} {'Files':<12} {'Models':<8} {'Constant':<10} {'Near-Zero':<12} {'Files List'}")
    print("-" * 100)
    
    # Print data for each dataset
    for dataset_name, stats in sorted(all_dataset_stats.items()):
        files_list = ', '.join(stats['low_variance_files'])  # Show all files, not just first 3
        windows_str = f"{stats['total_low_variance_windows']}/{stats['total_windows']}"
        files_str = f"{stats['total_low_variance_files']}/{stats['total_files']}"
        
        print(f"{dataset_name:<20} {windows_str:<15} {files_str:<12} "
              f"{stats['constant_low_variance_windows']:<10} {stats['near_zero_low_variance_windows']:<12} "
              f"{files_list}")
    
    # Print totals
    print("-" * 100)
    total_low_variance_windows_str = str(total_low_variance_windows)
    total_files_str = str(total_low_variance_files)
    
    if total_windows_processed is not None:
        total_low_variance_windows_str = f"{total_low_variance_windows}/{total_windows_processed}"
    if total_files_processed is not None:
        total_files_str = f"{total_low_variance_files}/{total_files_processed}"
    
    print(f"{'TOTAL':<20} {total_low_variance_windows_str:<15} {total_files_str:<12} {'':<8} {'':<10} {'':<12} {'':<20}")
    
    # Print summary statistics
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    print(f"Total datasets processed: {total_datasets}")
    print(f"Total low variance windows found: {total_low_variance_windows}")
    print(f"Total files with low variance windows: {total_low_variance_files}")
    print(f"Average windows per dataset: {total_low_variance_windows/total_datasets:.1f}")
    print(f"Average files per dataset: {total_low_variance_files/total_datasets:.1f}")
    
    # Find datasets with most issues
    if all_dataset_stats:
        max_windows_dataset = max(all_dataset_stats.items(), key=lambda x: x[1]['total_windows'])
        max_files_dataset = max(all_dataset_stats.items(), key=lambda x: x[1]['total_files'])
        
        print(f"\nDataset with most low variance windows: {max_windows_dataset[0]} ({max_windows_dataset[1]['total_windows']} windows)")
        print(f"Dataset with most affected files: {max_files_dataset[0]} ({max_files_dataset[1]['total_files']} files)")
    
    print("="*100)
