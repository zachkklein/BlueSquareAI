"""
Antisemitism detection pipeline.

Main entry point: classify_text()
"""

from pipeline.aggregate import classify_text, classify_text_async, classify_texts_batch

__all__ = ['classify_text', 'classify_text_async', 'classify_texts_batch']

