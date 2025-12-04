"""
Backward compatibility module - imports from lstm_model.py

This file exists for backward compatibility with existing tests.
All implementation is now in lstm_model.py
"""

from backend.services.lstm_model import LSTMModel, MCDropout, get_device

__all__ = ['LSTMModel', 'MCDropout', 'get_device']
