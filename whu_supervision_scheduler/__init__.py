"""
WHU Supervision Scheduler Package

A constraint programming-based scheduler for assigning exam supervision
duties to chairs at WHU, ensuring fairness and respecting constraints.
"""

from .scheduler import WHUSupervisionScheduler
from .utils import (
    validate_excel_file
)

__version__ = '1.0.0'
__author__ = 'WHU Supervision Scheduling Team'

__all__ = [
    'WHUSupervisionScheduler',
    'validate_excel_file'
]