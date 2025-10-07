"""
MUSED-FM Examples Package

This package contains example scripts and utilities for running MUSED-FM benchmarks.
"""

from .run_musedfm import (
    run_models_on_benchmark,
    compare_model_performance,
    export_hierarchical_results_to_csv,
    export_results_to_csv,
    get_available_models,
    parse_models,
    generate_forecast_plots,
    _aggregate_metrics,
    _aggregate_results_by_level
)

__all__ = [
    'run_models_on_benchmark',
    'compare_model_performance', 
    'export_hierarchical_results_to_csv',
    'export_results_to_csv',
    'get_available_models',
    'parse_models',
    'generate_forecast_plots',
    '_aggregate_metrics',
    '_aggregate_results_by_level'
]
