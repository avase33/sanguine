"""A prior/majority baseline — the score every real model must beat."""

from __future__ import annotations

from .base import Estimator


class PriorClassifier(Estimator):
    name = "prior_baseline"

    def __init__(self):
        self.prior: float | None = None

    def fit(self, X, y):
        self.prior = sum(y) / len(y) if y else 0.0
        return self

    def predict_proba(self, X):
        self._check_fitted("prior")
        return [self.prior for _ in X]

    def get_params(self):
        return {}
