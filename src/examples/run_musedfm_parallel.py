"""
Example demonstrating how to run multiple forecasting models with MUSED-FM using parallel processing.

This example shows how to:
1. Load datasets from the MUSED-FM benchmark
2. Run multiple baseline forecasting models in parallel
3. Compare model performance across different metrics
4. Generate visualizations and export results
5. Use parallel processing for faster execution

Usage:
    python run_musedfm_parallel.py --models mean,arima,linear --data-path /path/to/dataset
    python run_musedfm_parallel.py --models all --windows 50
    python run_musedfm_parallel.py --models all --num-processes 4
    python run_musedfm_parallel.py --help
"""

import time
import os
import warnings
import numpy as np
import argparse
from tqdm import tqdm
import copy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Set multiprocessing start method to 'spawn' for CUDA compatibility
mp.set_start_method('spawn', force=True)

# Set environment variable to suppress warnings at the system level
os.environ['PYTHONWARNINGS'] = 'ignore'

# Set warnings to ignore at the module level
warnings.simplefilter("ignore")

from musedfm.data import Benchmark
from musedfm.plotting import plot_window_forecasts

# Import utility and debug functions
from examples.utils import (
    _aggregate_metrics, _aggregate_results_by_level
)
from examples.model_handling import (
    parse_models
)
from examples.debug import (
    _initialize_nan_tracking, _update_nan_tracking, _check_window_nan_values,
    _report_nan_statistics, plot_high_mape_windows,
    debug_model_performance, debug_univariate_performance,
    debug_model_summary
)
from examples.export_csvs import (
    export_hierarchical_results_to_csv
)
from examples.eval_musedfm import (
    SaveManager
)

# Warning suppression removed - errors will be visible


# Old data extraction functions removed - now using dataset slicing approach


def _process_dataset_slice(args):
    """Process a dataset slice with all models.
    
    Args:
        args: Tuple containing (dataset_slice, models, dataset_name, 
              collect_plot_data, num_plots_to_keep, category_name, domain_name, 
              process_id, total_processes, gpu_id)
    
    Returns:
        List of window results for this slice
    """
    (dataset_slice, models, dataset_name, 
     collect_plot_data, num_plots_to_keep, category_name, domain_name,
     process_id, total_processes, gpu_id) = args
    
    # Set GPU device for this process
    if gpu_id is not None:
        import torch
        if torch.cuda.is_available():
            # Ensure this process uses only its assigned GPU
            torch.cuda.set_device(gpu_id)
            
            # Clear all CUDA cache to start fresh
            torch.cuda.empty_cache()
            
            # Get GPU memory info
            gpu_memory_total = torch.cuda.get_device_properties(gpu_id).total_memory
            gpu_memory_allocated = torch.cuda.memory_allocated(gpu_id)
            gpu_memory_free = gpu_memory_total - gpu_memory_allocated
            
            print(f"Process {process_id}/{total_processes} using GPU {gpu_id} (Memory: {gpu_memory_free/1e9:.1f}GB free / {gpu_memory_total/1e9:.1f}GB total)")
            
            # Verify we're on the correct device
            current_device = torch.cuda.current_device()
            if current_device != gpu_id:
                print(f"WARNING: Process {process_id} expected GPU {gpu_id} but is on GPU {current_device}")
        else:
            print(f"Process {process_id}/{total_processes}: CUDA not available, falling back to CPU")
    else:
        print(f"Process {process_id}/{total_processes}: Using CPU")
    
    slice_results = []
    total_windows = len(dataset_slice)
    process_start_time = time.time()
    
    print(f"Process {process_id}/{total_processes} starting: {dataset_name} with {total_windows} windows")
    
    # Process all windows in this slice with simple progress tracking
    for i, window in enumerate(dataset_slice):
        # Simple progress update every 100 windows (no tqdm in subprocess)
        if i % 100 == 0:
            progress = (i + 1) / total_windows * 100
            elapsed = time.time() - process_start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            
            # Calculate time to completion
            remaining_windows = total_windows - (i + 1)
            eta_seconds = remaining_windows / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60
            eta_hours = eta_minutes / 60
            
            # Format ETA nicely
            if eta_hours >= 1:
                eta_str = f"{eta_hours:.1f}h"
            elif eta_minutes >= 1:
                eta_str = f"{eta_minutes:.1f}m"
            else:
                eta_str = f"{eta_seconds:.0f}s"
            
            print(f"Process {process_id}/{total_processes}: {progress:.1f}% ({i+1}/{total_windows} windows) @ {rate:.1f} windows/s, ETA: {eta_str}")
        
        window_result = {
            'results': {},
            'plot_data': [],
            'window_index': i,
            'dataset_name': dataset_name,
            'category_name': category_name,
            'domain_name': domain_name,
            'process_id': process_id
        }
        
        # Process this window with all models
        for model_name, model in models.items():
            target_length = len(window.target())
            
            # Generate multivariate forecast if model supports it
            multivariate_forecast = None
            if not model["univariate"]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    multivariate_forecast = model["model"].forecast(window.history(), window.covariates(), target_length, window.timestamps())

                # Handle failed forecasts
                if multivariate_forecast is None:
                    multivariate_forecast = np.zeros(target_length)  # Fallback to zeros
            
            # Generate univariate forecast
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                univariate_forecast = model["model"].forecast(window.history(), None, target_length, window.timestamps())

            # Handle failed forecasts
            if univariate_forecast is None:
                univariate_forecast = np.zeros(target_length)  # Fallback to zeros

            # Validate forecast length matches target length
            if len(univariate_forecast) != target_length:
                raise ValueError(f"Forecast length mismatch: model '{model_name}' returned {len(univariate_forecast)} values, but target has {target_length} values")

            # Submit both forecasts
            window.submit_forecast(multivariate_forecast, univariate_forecast)
            
            # Get evaluation results for multivariate forecast if submitted
            if multivariate_forecast is not None:
                multivariate_results = window.evaluate("multivariate")
                window_result['results'][f"{model_name}_multivariate"] = multivariate_results
            
            # Store univariate results
            if univariate_forecast is not None:
                univariate_results = window.evaluate("univariate")
                window_result['results'][f"{model_name}_univariate"] = univariate_results
            
            # Collect data for plotting if requested
            if collect_plot_data and i < num_plots_to_keep:
                window_result['plot_data'].append({
                    'window': window,
                    'forecast': multivariate_forecast,
                    'univariate_forecast': univariate_forecast,
                    'model_name': model_name,
                    'window_index': i,
                    'dataset_name': dataset_name
                })
        
        slice_results.append(window_result)
    
    print(f"Process {process_id}/{total_processes} completed: {dataset_name} with {len(slice_results)} windows")
    
    # Calculate and display total processing time
    total_elapsed = time.time() - process_start_time
    total_minutes = total_elapsed / 60
    total_hours = total_minutes / 60
    
    # Format total time nicely
    if total_hours >= 1:
        time_str = f"{total_hours:.1f}h"
    elif total_minutes >= 1:
        time_str = f"{total_minutes:.1f}m"
    else:
        time_str = f"{total_elapsed:.1f}s"
    
    # Calculate final processing rate
    final_rate = len(slice_results) / total_elapsed if total_elapsed > 0 else 0
    
    print(f"Process {process_id}/{total_processes} finished in {time_str} @ {final_rate:.1f} windows/s average")
    
    # Clear GPU memory if using GPU
    if gpu_id is not None:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return slice_results


def run_models_parallel(benchmark_path: str, models: dict, max_windows: int = 100, 
                       categories: str = None, domains: str = None, datasets: str = None,
                       collect_plot_data: bool = False, history_length: int = 512, 
                       forecast_horizon: int = 128, stride: int = 256, load_cached_counts: bool = False,
                       num_plots_to_keep: int = 1, debug_mode: bool = False, chunk_size: int = 1048576, 
                       forecast_save_path: str = "", output_dir: str = "", num_processes: int = None,
                       gpu_ids: str = None, cpu_only: bool = False):
    """Run multiple forecasting models on a benchmark using parallel processing.
    
    This is a parallel version of run_models_on_benchmark that processes windows
    concurrently across multiple processes.
    
    Args:
        num_processes: Number of parallel processes to use. If None, uses CPU count.
        All other arguments are the same as run_models_on_benchmark.
    """
    print("=" * 60)
    print("Running Multiple Models on Benchmark (Parallel)")
    print("=" * 60)
    
    # Detect available GPUs (simple approach)
    available_gpus = []
    if not cpu_only:
        import torch
        if torch.cuda.is_available():
            if gpu_ids:
                # Use specified GPU IDs
                available_gpus = [int(x.strip()) for x in gpu_ids.split(',')]
                # Validate GPU IDs
                max_gpu_id = torch.cuda.device_count() - 1
                available_gpus = [gpu_id for gpu_id in available_gpus if 0 <= gpu_id <= max_gpu_id]
                print(f"Using specified GPUs: {available_gpus}")
            else:
                # Use all available GPUs
                available_gpus = list(range(torch.cuda.device_count()))
                print(f"Using all available GPUs: {available_gpus}")
        else:
            print("CUDA not available, falling back to CPU")
    else:
        print("CPU-only mode enabled, using CPU for all processes")
        # Force CPU-only mode by disabling CUDA
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Set default number of processes (more conservative to avoid I/O thrashing)
    if num_processes is None:
        if cpu_only:
            num_processes = min(mp.cpu_count() // 2, 4)  # Very conservative for CPU-only
        else:
            num_processes = min(mp.cpu_count() // 4, 2)  # Even more conservative for GPU
        print(f"Using {num_processes} parallel processes (conservative to avoid I/O thrashing)")
    else:
        print(f"Using {num_processes} parallel processes (user specified)")
    
    # Distribute processes across GPUs with conservative assignment
    gpu_assignment = []
    if available_gpus:
        # Conservative approach: 1 process per GPU to avoid memory conflicts
        if num_processes > len(available_gpus):
            print(f"WARNING: Requested {num_processes} processes but only {len(available_gpus)} GPUs available.")
            print(f"Limiting to {len(available_gpus)} processes (1 per GPU) to avoid CUDA memory conflicts.")
            num_processes = len(available_gpus)
        
        # Assign exactly 1 process per GPU
        for i in range(num_processes):
            gpu_id = available_gpus[i]
            gpu_assignment.append(gpu_id)
        
        print(f"GPU assignment: {gpu_assignment}")
        print(f"Using {num_processes} processes (1 per GPU) to avoid CUDA memory conflicts")
            
    else:
        gpu_assignment = [None] * num_processes
        print("Using CPU for all processes")
    
    benchmark = Benchmark(benchmark_path, history_length=history_length, forecast_horizon=forecast_horizon, stride=stride, load_cached_counts=load_cached_counts)
    print(f"Loaded benchmark with {len(benchmark)} categories")
    print(f"Running {len(models)} models: {list(models.keys())}")
    
    # Create model instances for each process to avoid serialization issues
    # We'll create n * num_models separate model instances
    print(f"Creating {num_processes} * {len(models)} = {num_processes * len(models)} model instances for parallel processing")
    
    # Store original models for reference
    original_models = models
    
    results = {}
    plot_data = []  # Store windows and forecasts for plotting
    
    # Initialize results for each model
    results_base_dict = {
        'metrics': {},
        'dataset_results': [],
        'category_results': {},
        'domain_results': {},
        'time': 0.0,
        'windows': 0
    }

    # Save managers are not used in parallel processing to avoid serialization issues
    model_save_managers_multivariate = {}
    model_save_managers_univariate = {}
    for model_name in models.keys():
        results[model_name] = copy.deepcopy(results_base_dict)
        
        # Initialize univariate results only for non-univariate models
        if not models[model_name]["univariate"]:
            univariate_model_name = f"{model_name}_univariate"
            results[univariate_model_name] = copy.deepcopy(results_base_dict)
        
        # Disable save managers for parallel processing
        model_save_managers_univariate[model_name] = None
        model_save_managers_multivariate[model_name] = None
    
    # Count total datasets for progress bar
    total_datasets = 0
    for category in benchmark:
        if categories is not None and category.category_path.name not in categories:
            continue
        for domain in category:
            if domains is not None and domain.domain_path.name not in domains:
                continue
            for dataset in domain:
                if datasets is not None and dataset.data_path.name not in datasets:
                    continue
                total_datasets += 1
    
    # Main processing loop with comprehensive progress tracking
    dataset_progress = tqdm(total=total_datasets, desc="Processing datasets", unit="dataset")
    total_windows_processed = 0
    total_models_processed = 0
    start_time = time.time()
    
    for category in benchmark:
        # Apply filters if specified
        if categories is not None and category.category_path.name not in categories:
            continue
        for domain in category:
            if domains is not None and domain.domain_path.name not in domains:
                continue
            for dataset in domain:
                if datasets is not None and dataset.data_path.name not in datasets:
                    continue
                
                # Get full dataset name from benchmark path
                dataset_name = str(dataset.dataset_name)
                
                # Calculate current processing rate
                elapsed_time = time.time() - start_time
                windows_per_second = total_windows_processed / elapsed_time if elapsed_time > 0 else 0
                
                dataset_progress.set_postfix_str(f"Processing: {dataset_name} | Windows: {total_windows_processed} | Rate: {windows_per_second:.1f} windows/s")
                
                # Process all models for this dataset using parallel processing
                print(f"  Processing {max_windows if max_windows is not None else 'all'} windows with {len(models)} models using {num_processes} processes...")
                
                # Initialize per-model tracking for this dataset
                model_dataset_windows = {model_name: 0 for model_name in models.keys()}
                model_dataset_results = {model_name: [] for model_name in models.keys()}
                # Add univariate results tracking only for non-univariate models
                for model_name, model in models.items():
                    if not model["univariate"]:
                        model_dataset_results[f"{model_name}_univariate"] = []
                model_start_times = {model_name: time.time() for model_name in models.keys()}
                print("initialized models", model_start_times)
                
                # Initialize NaN tracking for this dataset
                nan_stats = _initialize_nan_tracking()
                
                # Determine number of files to process
                num_files = min(dataset.num_files(), max_windows) if max_windows is not None else dataset.num_files()
                
                # Create dataset slices for parallel processing
                slice_size = max(1, num_files // num_processes)
                dataset_slices = []
                
                for i in range(num_processes):
                    start_idx = i * slice_size
                    end_idx = min((i + 1) * slice_size, num_files)
                    
                    if start_idx < num_files:
                        # Create a slice of the dataset
                        dataset_slice = dataset[start_idx:end_idx]
                        dataset_slices.append(dataset_slice)
                
                print(f"Created {len(dataset_slices)} dataset slices for parallel processing")
                
                # Warn about potential I/O thrashing with large datasets
                if num_files > 1000:
                    print(f"WARNING: Large dataset detected ({num_files} files). Consider using fewer processes (--num-processes 2) to avoid I/O thrashing.")
                
                # Warn about Chronos memory usage
                if 'chronos' in models and not cpu_only:
                    print(f"WARNING: Chronos model detected with GPU usage. Each process loads a separate Chronos model.")
                    print(f"Consider using --cpu-only for large datasets to avoid CUDA memory conflicts.")
                
                # Create model instances for each slice
                slice_args = []
                for i, dataset_slice in enumerate(dataset_slices):
                    # Create fresh model instances for this slice
                    slice_models = {}
                    for model_name, model in original_models.items():
                        # Create a new instance of the same model class
                        model_class = model['model'].__class__
                        
                        # Special handling for ChronosForecast to pass device parameter
                        if model_name == 'chronos':
                            from musedfm.baselines.chronos_forecast import ChronosForecast
                            device = "cpu" if cpu_only else "cuda:0"
                            slice_models[model_name] = {
                                'model': ChronosForecast(device=device),
                                'univariate': model['univariate']
                            }
                        else:
                            slice_models[model_name] = {
                                'model': model_class(),  # Create new instance
                                'univariate': model['univariate']
                            }
                    
                    slice_args.append((
                        dataset_slice, slice_models, dataset_name,
                        collect_plot_data, num_plots_to_keep, category.category, domain.domain_name,
                        i, len(dataset_slices), gpu_assignment[i]
                    ))
                
                # Process dataset slices in parallel
                all_window_results = []
                print(f"Starting parallel processing of {len(slice_args)} slices for {dataset_name}")
                
                with ProcessPoolExecutor(max_workers=num_processes) as executor:
                    # Submit all slice tasks
                    future_to_args = {executor.submit(_process_dataset_slice, args): args for args in slice_args}
                    
                    # Collect results as they complete
                    slice_progress = tqdm(total=len(slice_args), desc=f"Slices in {dataset_name}", unit="slice", leave=False)
                    completed_slices = 0
                    total_windows_processed = 0
                    slice_start_time = time.time()
                    
                    for future in as_completed(future_to_args):
                        slice_results = future.result()
                        all_window_results.extend(slice_results)
                        completed_slices += 1
                        total_windows_processed += len(slice_results)
                        
                        # Calculate slice processing rate
                        slice_elapsed = time.time() - slice_start_time
                        slice_rate = total_windows_processed / slice_elapsed if slice_elapsed > 0 else 0
                        
                        # Update progress with detailed information
                        slice_progress.set_postfix_str(f"Completed {completed_slices}/{len(slice_args)} slices, {total_windows_processed} windows processed, {slice_rate:.1f} windows/s")
                        slice_progress.update(1)
                    
                    slice_progress.close()
                
                print(f"Completed parallel processing: {len(all_window_results)} windows processed from {len(slice_args)} slices")
                
                window_results = all_window_results
                
                # Aggregate results from parallel processing
                for window_result in window_results:
                    window_index = window_result['window_index']
                    
                    # Process results for each model
                    for model_name in models.keys():
                        multivariate_key = f"{model_name}_multivariate"
                        univariate_key = f"{model_name}_univariate"
                        
                        # Handle multivariate results
                        if multivariate_key in window_result['results']:
                            model_dataset_results[model_name].append(window_result['results'][multivariate_key])
                            model_dataset_windows[model_name] += 1
                            results[model_name]['windows'] += 1
                        
                        # Handle univariate results
                        if univariate_key in window_result['results']:
                            if models[model_name]["univariate"]:
                                model_dataset_results[model_name].append(window_result['results'][univariate_key])
                            else:
                                model_dataset_results[f"{model_name}_univariate"].append(window_result['results'][univariate_key])
                                results[f"{model_name}_univariate"]['windows'] += 1
                    
                    # Collect plot data
                    if collect_plot_data and window_result['plot_data']:
                        plot_data.extend(window_result['plot_data'])
                
                # Save managers are disabled in parallel processing
                
                # Calculate average metrics and store results for each model
                for model_name in models.keys():
                    # Calculate average metrics for this model on this dataset
                    dataset_avg_metrics = _calculate_dataset_metrics(model_dataset_results, model_name)
                    
                    # Store dataset results for this model
                    results[model_name]['dataset_results'].append({
                        'dataset_name': dataset_name,
                        'metrics': dataset_avg_metrics,
                        'window_count': model_dataset_windows[model_name]
                    })
                    
                    model_elapsed_time = time.time() - model_start_times[model_name]
                    results[model_name]['time'] += model_elapsed_time
                    
                    if debug_mode:
                        debug_model_performance(model_name, dataset_avg_metrics, model_dataset_windows, model_elapsed_time)
                        plot_high_mape_windows(model_name, dataset_name, dataset, models[model_name], dataset_avg_metrics, save_path=output_dir)
                
                # Process univariate results for multivariate models only
                for model_name, model in models.items():
                    univariate_model_name, univariate_avg_metrics = _process_univariate_results(
                        model_name, model, model_dataset_results, model_dataset_windows, 
                        model_elapsed_time, dataset_name, results
                    )
                    
                    if univariate_model_name is not None and debug_mode:
                        plot_high_mape_windows(univariate_model_name, dataset_name, dataset, model, univariate_avg_metrics, forecast_type="univariate", save_path=output_dir)
                
                # Report NaN statistics for this dataset
                if debug_mode:
                    _report_nan_statistics(nan_stats)
                
                # Update cumulative statistics
                total_windows_processed += len(all_window_results)
                total_models_processed += len(all_window_results) * len(models)
                
                # Calculate overall processing rate
                elapsed_time = time.time() - start_time
                overall_rate = total_windows_processed / elapsed_time if elapsed_time > 0 else 0
                
                # Update dataset progress
                dataset_progress.set_postfix_str(f"Completed: {dataset_name} | Windows: {total_windows_processed} | Rate: {overall_rate:.1f} windows/s")
                dataset_progress.update(1)
                print(f"  Completed dataset {dataset_name} with {len(all_window_results)} windows")
        
        # Aggregate domain-level metrics for this domain
        _aggregate_results_by_level(results, models, benchmark, domain.domain_name, 'domain')
    
    # Close dataset progress bar
    dataset_progress.close()
    
    # Calculate final processing statistics
    total_elapsed_time = time.time() - start_time
    final_rate = total_windows_processed / total_elapsed_time if total_elapsed_time > 0 else 0
    
    # Print final summary
    print(f"\n🎉 Processing Complete!")
    print(f"📊 Total datasets processed: {total_datasets}")
    print(f"🪟 Total windows processed: {total_windows_processed}")
    print(f"🤖 Total model runs: {total_models_processed}")
    print(f"⏱️  Total processing time: {total_elapsed_time:.1f} seconds")
    print(f"⚡ Overall processing rate: {final_rate:.1f} windows/second")
    print(f"📈 Average windows per dataset: {total_windows_processed / total_datasets:.1f}")
    print(f"🚀 Average model runs per window: {total_models_processed / total_windows_processed:.1f}")
    
    # Aggregate category-level metrics for each category
    for category in benchmark:
        _aggregate_results_by_level(results, models, benchmark, category.category, 'category')
        
    # Calculate overall average metrics across all datasets for each model
    all_model_names = list(models.keys())
    # Add univariate model names for multivariate models only
    for model_name, model in models.items():
        if not model["univariate"]:
            all_model_names.append(f"{model_name}_univariate")
    
    for model_name in all_model_names:
        if results[model_name]['dataset_results']:
            overall_avg_metrics, _, _ = _aggregate_metrics(results[model_name]['dataset_results'])
        else:
            overall_avg_metrics = {}
        
        results[model_name]['metrics'] = overall_avg_metrics
        if debug_mode:
            debug_model_summary(model_name, results, models)
    
    if collect_plot_data:
        results['_plot_data'] = plot_data
    
    return results


def _calculate_dataset_metrics(model_dataset_results, model_name):
    """Calculate average metrics for a model on a dataset."""
    if not model_dataset_results[model_name]:
        return {}
    
    # Get metric names from the first result
    metric_names = list(model_dataset_results[model_name][0].keys())
    
    # Collect all metric vectors from all batches
    all_metric_values = {metric: [] for metric in metric_names}
    
    for result in model_dataset_results[model_name]:
        # Each result now contains metric vectors instead of scalars
        for metric in metric_names:
            all_metric_values[metric].extend(result[metric])
    
    # Calculate single nanmean across all individual window metrics
    avg_metrics = {}
    for metric in metric_names:
        avg_metrics[metric] = np.nanmean(all_metric_values[metric]) if all_metric_values[metric] else np.nan
    
    return avg_metrics


def _process_univariate_results(model_name, model, model_dataset_results, model_dataset_windows, 
                               model_elapsed_time, dataset_name, results):
    """Process univariate results for multivariate models."""
    univariate_model_name = None
    univariate_avg_metrics = {}
    
    if not model["univariate"]:
        univariate_model_name = f"{model_name}_univariate"
        
        # Calculate average metrics for univariate results (parallel processing style)
        if model_dataset_results[univariate_model_name]:
            # Get metric names from the first result
            metric_names = list(model_dataset_results[univariate_model_name][0].keys())
            
            # Collect all metric vectors from all batches
            all_metric_values = {metric: [] for metric in metric_names}
            
            for result in model_dataset_results[univariate_model_name]:
                # Each result now contains metric vectors instead of scalars
                for metric in metric_names:
                    all_metric_values[metric].extend(result[metric])
            
            # Calculate single nanmean across all individual window metrics
            avg_metrics = {}
            for metric in metric_names:
                avg_metrics[metric] = np.nanmean(all_metric_values[metric]) if all_metric_values[metric] else np.nan
            
            univariate_avg_metrics = avg_metrics
        
        # Store dataset results for univariate model
        results[univariate_model_name]['dataset_results'].append({
            'dataset_name': dataset_name,
            'metrics': univariate_avg_metrics,
            'window_count': model_dataset_windows[model_name]  # Same window count as multivariate
        })
        
        # Update timing (same as multivariate)
        results[univariate_model_name]['time'] += model_elapsed_time
    
    return univariate_model_name, univariate_avg_metrics


def compare_model_performance(results):
    """Compare and display model performance metrics."""
    print("\n" + "=" * 60)
    print("Model Performance Comparison")
    print("=" * 60)
    
    # Get all model names
    model_names = [name for name in results.keys() if not name.startswith('_')]
    
    if not model_names:
        print("No model results found.")
        return
    
    # Display metrics for each model
    for model_name in model_names:
        if not results[model_name]['dataset_results']:
            continue
            
        print(f"\n{model_name}:")
        print(f"  Total windows processed: {results[model_name]['windows']}")
        print(f"  Total time: {results[model_name]['time']:.2f} seconds")
        
        if results[model_name]['metrics']:
            print("  Average metrics:")
            for metric, value in results[model_name]['metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}: {value}")


def main():
    """Main function demonstrating multiple model usage with MUSED-FM using parallel processing."""
    parser = argparse.ArgumentParser(
        description="MUSED-FM Example: Run multiple forecasting models in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_musedfm_parallel.py --models mean,arima --benchmark-path /path/to/benchmark
  python run_musedfm_parallel.py --models all --windows 50 --plots
  python run_musedfm_parallel.py --models linear_trend,exponential_smoothing --windows 20
  python run_musedfm_parallel.py --models all --categories Traditional --domains Energy
  python run_musedfm_parallel.py --models mean --datasets al_daily,bitcoin_price --plots
  python run_musedfm_parallel.py --models all --num-processes 4
  python run_musedfm_parallel.py --models mean,arima --windows 100 --num-processes 2
        """
    )
    
    parser.add_argument(
        "--models", 
        type=str, 
        default="mean,linear_trend",
        help="Comma-separated list of models to run, or 'all' for all available models"
    )
    
    parser.add_argument(
        "--benchmark-path",
        type=str,
        default="/home/caleb/musedfm_data",
        help="Path to the benchmark directory containing categories"
    )
    
    parser.add_argument(
        "--categories",
        type=str,
        help="Comma-separated list of categories to filter by (e.g., 'Traditional,Synthetic')"
    )
    
    parser.add_argument(
        "--domains",
        type=str,
        help="Comma-separated list of domains to filter by (e.g., 'Energy,Finance')"
    )
    
    parser.add_argument(
        "--datasets",
        type=str,
        help="Comma-separated list of datasets to filter by (e.g., 'al_daily,bitcoin_price')"
    )
    
    parser.add_argument(
        "--windows",
        type=int,
        default=None,
        help="Maximum number of windows to process per dataset (default: None for all windows)"
    )
    
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate forecast plots"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp",
        help="Output directory for plots and CSV files (default: /tmp)"
    )
    
    parser.add_argument(
        "--history-length",
        type=int,
        default=512,
        help="History length for windows (default: 512)"
    )
    
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=128,
        help="Forecast horizon for windows (default: 128)"
    )
    
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
        help="Stride between windows (default: 256)"
    )
    
    parser.add_argument(
        "--load-cached-counts",
        action="store_true",
        help="Load window counts from cached JSON files instead of generating"
    )

    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Enable debug mode"
    )

    parser.add_argument(
        "--forecast-save-path",
        type=str,
        default="",
        help="Path to save the forecasts (default: empty, no saving)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=131072,
        help="Chunk size for saving forecasts (default: 1048576)"
    )
    
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, uses all available GPUs. Only used when --cpu-only is not specified."
    )
    
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of parallel processes to use (default: CPU count/2 for CPU-only, CPU count/4 for GPU). Use fewer processes to avoid I/O thrashing with large datasets."
    )
    
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU-only mode, disable GPU usage even if available"
    )
    
    args = parser.parse_args()
    
    print("MUSED-FM Example: Multiple Model Forecasting (Parallel)")
    print("=" * 60)
    print(f"Models: {args.models}")
    print(f"Benchmark path: {args.benchmark_path}")
    print(f"categories: {args.categories or 'All'}")
    print(f"Domains: {args.domains or 'All'}")
    print(f"Datasets: {args.datasets or 'All'}")
    print(f"Max windows per dataset: {args.windows}")
    print(f"Generate plots: {args.plots}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of processes: {args.num_processes or 'Auto (CPU count, capped at 8)'}")
    
    # Parse models
    models = parse_models(args.models, device="cpu" if args.cpu_only else "cuda:0")
    if not models:
        print("Error: No valid models specified")
        return
    
    # Parse filter arguments
    categories = args.categories.split(',') if args.categories else None
    domains = args.domains.split(',') if args.domains else None
    datasets = args.datasets.split(',') if args.datasets else None
    
    start_time = time.time()
    
    # Run models on benchmark using parallel processing
    results = run_models_parallel(args.benchmark_path, models, args.windows, 
                                 categories=categories, domains=domains, datasets=datasets,
                                 collect_plot_data=args.plots, history_length=args.history_length,
                                 forecast_horizon=args.forecast_horizon, stride=args.stride, load_cached_counts=args.load_cached_counts,
                                 debug_mode=args.debug_mode, 
                                 chunk_size=args.chunk_size, forecast_save_path=args.forecast_save_path, output_dir=args.output_dir,
                                 num_processes=args.num_processes, gpu_ids=args.gpu_ids, cpu_only=args.cpu_only)
    
    # Compare performance
    compare_model_performance(results)
    # Export hierarchical CSV results
    export_hierarchical_results_to_csv(results, output_dir=args.output_dir)
    
    # Generate plots if requested
    if args.plots:
        print("\nGenerating forecast plots...")
        plot_window_forecasts(results['_plot_data'], output_dir=args.output_dir)
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()
