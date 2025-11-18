"""
Forecast plotting utilities for MUSED-FM visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional
from pathlib import Path
from ..data.window import Window


def plot_window_forecasts(
    window: Window,
    forecasts: Dict[str, np.ndarray],
    title: Optional[str] = None,
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
    show_plot: bool = False
) -> None:
    """
    Plot a window with its history, target, and multiple forecasts.
    
    Args:
        window: Window object containing history and target
        forecasts: Dictionary mapping baseline names to forecast arrays
        title: Optional title for the plot
        figsize: Figure size tuple (width, height)
        save_path: Optional path to save the plot
    """
    # Get data from window
    history = window.history()
    target = window.target()
    
    # Handle batched data by taking first sample
    if history.ndim == 2:
        history = history[0]  # Take first sample from batch
    if target.ndim == 2:
        target = target[0]  # Take first sample from batch
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot history
    history_indices = np.arange(len(history))
    plt.plot(history_indices, history, 'b-', linewidth=2, label='History', alpha=0.8)
    
    # Plot target
    target_indices = np.arange(len(history), len(history) + len(target))
    plt.plot(target_indices, target, 'g-', linewidth=2, label='Target', alpha=0.8)
    
    # Plot forecasts
    colors = ['red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'black', 'yellow']
    for i, (baseline_name, forecast) in enumerate(forecasts.items()):
        # Skip None forecasts
        if forecast is None:
            print(f"Warning: Skipping {baseline_name} forecast (None)")
            continue
        
        # Handle batched forecast data by taking first sample
        if forecast.ndim == 2:
            forecast = forecast[0]  # Take first sample from batch
            
        color = colors[i % len(colors)]
        forecast_indices = np.arange(len(history), len(history) + len(forecast))
        plt.plot(forecast_indices, forecast, '--', color=color, linewidth=2, 
                label=f'{baseline_name} Forecast', alpha=0.8)
    
    # Add vertical line to separate history from target/forecasts
    plt.axvline(x=len(history) - 0.5, color='black', linestyle=':', alpha=0.5)
    
    # Customize plot
    plt.xlabel('Time Steps', fontsize=18)
    plt.ylabel('Value', fontsize=18)
    plt.title(title or 'Window History, Target, and Forecasts', fontsize=28)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add text annotation for the separation line
    plt.text(len(history) - 0.5, plt.ylim()[1] * 0.95, 'History → Target/Forecast', 
             rotation=90, ha='right', va='top', fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        # Create directory if it doesn't exist
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    if show_plot:
        plt.show()
        return plt
    else:
        plt.close()


def plot_multiple_windows(
    windows: list,
    forecasts_dict: Dict[str, Dict[int, np.ndarray]],
    window_titles: Optional[list] = None,
    figsize: tuple = (15, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Plot multiple windows with their forecasts in subplots.
    
    Args:
        windows: List of Window objects
        forecasts_dict: Dictionary mapping baseline names to dictionaries of window_id -> forecast
        window_titles: Optional list of titles for each subplot
        figsize: Figure size tuple (width, height)
        save_path: Optional path to save the plot
    """
    n_windows = len(windows)
    n_cols = min(3, n_windows)
    n_rows = (n_windows + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_windows == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    colors = ['red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, window in enumerate(windows):
        ax = axes[i]
        
        # Get data from window
        history = window.history()
        target = window.target()
        
        # Handle batched data by taking first sample
        if history.ndim == 2:
            history = history[0]  # Take first sample from batch
        if target.ndim == 2:
            target = target[0]  # Take first sample from batch
        
        # Plot history
        history_indices = np.arange(len(history))
        ax.plot(history_indices, history, 'b-', linewidth=2, label='History', alpha=0.8)
        
        # Plot target
        target_indices = np.arange(len(history), len(history) + len(target))
        ax.plot(target_indices, target, 'g-', linewidth=2, label='Target', alpha=0.8)
        
        # Plot forecasts for this window
        for j, (baseline_name, window_forecasts) in enumerate(forecasts_dict.items()):
            if i in window_forecasts:
                color = colors[j % len(colors)]
                forecast = window_forecasts[i]
                
                # Handle batched forecast data by taking first sample
                if forecast.ndim == 2:
                    forecast = forecast[0]  # Take first sample from batch
                
                forecast_indices = np.arange(len(history), len(history) + len(forecast))
                ax.plot(forecast_indices, forecast, '--', color=color, linewidth=2, 
                       label=f'{baseline_name}', alpha=0.8)
        
        # Add vertical line to separate history from target/forecasts
        ax.axvline(x=len(history) - 0.5, color='black', linestyle=':', alpha=0.5)
        
        # Customize subplot
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.set_title(window_titles[i] if window_titles and i < len(window_titles) else f'Window {i}')
        ax.grid(True, alpha=0.3)
        
        # Add legend only to first subplot
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Hide unused subplots
    for i in range(n_windows, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        # Create directory if it doesn't exist
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-window plot saved to: {save_path}")
    
        plt.close()
    plt.show()
    return plt


def plot_baseline_comparison(
    window: Window,
    forecasts: Dict[str, np.ndarray],
    metrics: Optional[Dict[str, Dict[str, float]]] = None,
    title: Optional[str] = None,
    figsize: tuple = (15, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot a window with forecasts and include metrics comparison.
    
    Args:
        window: Window object containing history and target
        forecasts: Dictionary mapping baseline names to forecast arrays
        metrics: Optional dictionary mapping baseline names to metrics
        title: Optional title for the plot
        figsize: Figure size tuple (width, height)
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Get data from window
    history = window.history()
    target = window.target()
    
    # Handle batched data by taking first sample
    if history.ndim == 2:
        history = history[0]  # Take first sample from batch
    if target.ndim == 2:
        target = target[0]  # Take first sample from batch
    
    # Plot 1: Time series
    history_indices = np.arange(len(history))
    ax1.plot(history_indices, history, 'b-', linewidth=2, label='History', alpha=0.8)
    
    target_indices = np.arange(len(history), len(history) + len(target))
    ax1.plot(target_indices, target, 'g-', linewidth=2, label='Target', alpha=0.8)
    
    colors = ['red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, (baseline_name, forecast) in enumerate(forecasts.items()):
        color = colors[i % len(colors)]
        
        # Handle batched forecast data by taking first sample
        if forecast.ndim == 2:
            forecast = forecast[0]  # Take first sample from batch
        
        forecast_indices = np.arange(len(history), len(history) + len(forecast))
        ax1.plot(forecast_indices, forecast, '--', color=color, linewidth=2, 
                label=f'{baseline_name}', alpha=0.8)
    
    ax1.axvline(x=len(history) - 0.5, color='black', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Value')
    ax1.set_title(title or 'Window History, Target, and Forecasts')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Metrics comparison
    if metrics:
        baseline_names = list(metrics.keys())
        mape_values = [metrics[name]['MAPE'] for name in baseline_names]
        
        bars = ax2.bar(baseline_names, mape_values, color=colors[:len(baseline_names)], alpha=0.7)
        ax2.set_ylabel('MAPE (%)')
        ax2.set_title('Forecast Accuracy Comparison (MAPE)')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, mape_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.2f}%', ha='center', va='bottom')
        
        # Rotate x-axis labels if too long
        if max(len(name) for name in baseline_names) > 10:
            ax2.tick_params(axis='x', rotation=45)
    else:
        ax2.set_visible(False)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        # Create directory if it doesn't exist
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
        plt.close()
    else:
        plt.show()
        return plt


def export_metrics_to_csv(
    results: Dict[str, Dict],
    output_dir: str = "/tmp"
) -> bool:
    """
    Export averaged metrics per dataset to CSV files.
    
    Args:
        results: Dictionary mapping model names to results with dataset_results
        output_dir: Directory to save CSV files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not results or '_plot_data' not in results:
            print("No results data to export")
            return False
        
        # Get model names (exclude _plot_data)
        model_names = [k for k in results.keys() if k != '_plot_data']
        if not model_names:
            print("No model results found")
            return False
        
        # Create per-dataset CSV files
        dataset_metrics = {}
        
        for model_name in model_names:
            if 'dataset_results' not in results[model_name]:
                print(f"Warning: No dataset results found for model {model_name}")
                continue
                
            for dataset_result in results[model_name]['dataset_results']:
                dataset_name = dataset_result['dataset_name']
                metrics = dataset_result['metrics']
                
                if dataset_name not in dataset_metrics:
                    dataset_metrics[dataset_name] = {}
                
                dataset_metrics[dataset_name][model_name] = {
                    'MAPE': metrics.get('MAPE', 0.0),
                    'MAE': metrics.get('MAE', 0.0),
                    'RMSE': metrics.get('RMSE', 0.0),
                    'NMAE': metrics.get('NMAE', 0.0),
                    'window_count': dataset_result.get('window_count', 0)
                }
        
        # Export CSV for each dataset
        import pandas as pd
        exported_files = []
        
        for dataset_name, model_metrics in dataset_metrics.items():
            # Create CSV rows for this dataset
            csv_rows = []
            for model_name in model_names:
                if model_name in model_metrics:
                    row = {
                        'model': model_name,
                        'MAPE': model_metrics[model_name]['MAPE'],
                        'MAE': model_metrics[model_name]['MAE'],
                        'RMSE': model_metrics[model_name]['RMSE'],
                        'NMAE': model_metrics[model_name]['NMAE'],
                        'window_count': model_metrics[model_name]['window_count']
                    }
                    csv_rows.append(row)
            
            if csv_rows:
                # Save to CSV
                df = pd.DataFrame(csv_rows)
                # Clean dataset name for filename
                clean_dataset_name = dataset_name.replace('/', '_').replace('\\', '_')
                csv_path = output_path / f"museval_metrics_{clean_dataset_name}.csv"
                df.to_csv(csv_path, index=False)
                exported_files.append(csv_path)
                
                print(f"Metrics exported for dataset '{dataset_name}' to: {csv_path}")
        
        print(f"Exported metrics for {len(exported_files)} datasets")
        print(f"CSV files saved to {output_dir}/")
        
        # Print summary for first dataset as example
        if exported_files:
            df_example = pd.read_csv(exported_files[0])
            print(f"\nExample metrics from {exported_files[0].name}:")
            print(df_example.to_string(index=False, float_format='%.4f'))
        
        return True
        
    except Exception as e:
        print(f"Error exporting metrics to CSV: {e}")
        return False
