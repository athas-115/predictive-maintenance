"""
Source package initialization.
"""

__version__ = '1.0.0'
__author__ = 'atharva'

# Import main modules for easier access
from . import preprocess
from . import train
from . import evaluate
from . import monitor
from . import retrain

__all__ = [
    'preprocess',
    'train',
    'evaluate',
    'monitor',
    'retrain',
]
