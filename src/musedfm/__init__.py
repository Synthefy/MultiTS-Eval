"""
MUSED-FM: Multi-Scale Evaluation Dataset for Forecasting Models
"""

from musedfm.data import Window, Dataset, Domain, Category, Benchmark
from musedfm import metrics
from musedfm import baselines
from musedfm import plotting

__version__ = "1.0.0"
__all__ = ['Window', 'Dataset', 'Domain', 'Category', 'Benchmark', 'metrics', 'baselines', 'plotting']