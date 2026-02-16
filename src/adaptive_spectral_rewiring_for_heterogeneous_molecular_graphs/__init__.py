"""
Adaptive Spectral Rewiring for Heterogeneous Molecular Graphs.

This package implements dynamic graph rewiring based on spectral gap analysis
and molecular motif preservation for improved molecular property prediction.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from .models.model import AdaptiveSpectralGNN
from .training.trainer import Trainer
from .evaluation.metrics import compute_metrics

__all__ = ["AdaptiveSpectralGNN", "Trainer", "compute_metrics"]
