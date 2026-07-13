"""Sanguine — a from-scratch AutoML engine, demonstrated on blood-donation prediction.

Inspired by the "Give Life: Predict Blood Donations" task, Sanguine goes beyond
calling an AutoML library: it implements the ML models *and* the pipeline search
from scratch in pure Python (zero dependencies). It evolves preprocessing +
model + hyperparameter pipelines with cross-validated scoring, produces a
leaderboard, and persists the winning model for serving.

An optional adapter runs real scikit-learn / TPOT pipelines for comparison.
"""

from .version import __version__

__all__ = ["__version__"]
