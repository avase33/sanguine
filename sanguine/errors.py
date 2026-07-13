"""Exception hierarchy for Sanguine."""

from __future__ import annotations


class SanguineError(Exception):
    """Base class for all Sanguine errors."""


class DataError(SanguineError):
    """Problem loading or shaping data."""


class NotFittedError(SanguineError):
    """An estimator was used before it was fitted."""


class SearchError(SanguineError):
    """The AutoML search could not complete."""


class PersistenceError(SanguineError):
    """Saving or loading a model failed."""
