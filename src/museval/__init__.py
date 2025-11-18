"""
MUSED-FM: Multi-Scale Evaluation Dataset for Forecasting Models
"""

from museval.data import Window, Dataset, Domain, Category, Benchmark
from museval import metrics
from museval import baselines
from museval import plotting
from museval import examples

__version__ = "1.0.0"
__all__ = ['Window', 'Dataset', 'Domain', 'Category', 'Benchmark', 'metrics', 'baselines', 'plotting', 'examples']