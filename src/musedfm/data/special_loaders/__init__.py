"""
Special loaders for datasets that require custom processing.

These loaders handle datasets with unique data formats or structures that
cannot be processed using the standard parquet file loading approach.
"""

from musedfm.data.special_loaders.open_aq_special import OpenAQSpecialLoader

__all__ = ['OpenAQSpecialLoader']
