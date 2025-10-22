"""
MUSED-FM: Multi-Scale Evaluation Dataset for Forecasting Models
"""

from multieval.data import Window, Dataset, Domain, Category, Benchmark
from multieval import metrics
from multieval import baselines
from multieval import plotting
# from multits import examples  # Commented out to avoid circular import

__version__ = "1.0.0"
__all__ = ['Window', 'Dataset', 'Domain', 'Category', 'Benchmark', 'metrics', 'baselines', 'plotting', 'examples']