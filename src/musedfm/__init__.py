"""
MUSED-FM: Multi-Scale Evaluation Dataset for Forecasting Models
"""

from .data import Window, Dataset, Domain, Category, Benchmark
from . import metrics
from . import baselines
from . import plotting

__version__ = "1.0.0"
__all__ = ['Window', 'Dataset', 'Domain', 'Category', 'Benchmark', 'metrics', 'baselines', 'plotting']