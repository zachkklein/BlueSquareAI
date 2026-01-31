# pipeline/aggregate.py
"""
Main entry point for antisemitism detection system.
Uses optimized async version with parallel processing and caching.
"""

from pipeline.aggregate_optimized import classify_text_async, classify_text, classify_texts_batch

__all__ = ['classify_text', 'classify_text_async', 'classify_texts_batch']
