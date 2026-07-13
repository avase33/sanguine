"""A Pipeline = a preprocessing transformer followed by an estimator.

This is the unit the AutoML engine searches over: a (scaler, model,
hyperparameters) triple. Fitting fits the scaler then the model; prediction
applies the scaler then the model.
"""

from __future__ import annotations

from typing import Any

from .data.preprocess import Transformer
from .models.base import Estimator


class Pipeline:
    def __init__(self, scaler: Transformer, estimator: Estimator, spec: dict[str, Any] | None = None):
        self.scaler = scaler
        self.estimator = estimator
        self.spec = spec or {}

    def fit(self, X: list[list[float]], y: list[int]) -> "Pipeline":
        Xs = self.scaler.fit_transform(X)
        self.estimator.fit(Xs, y)
        return self

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        return self.estimator.predict_proba(self.scaler.transform(X))

    def predict(self, X: list[list[float]], threshold: float = 0.5) -> list[int]:
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]

    def describe(self) -> str:
        return f"{self.scaler.name} → {self.estimator.describe()}"

    def to_spec(self) -> dict[str, Any]:
        return {
            "scaler": self.scaler.name,
            "model": self.estimator.name,
            "params": self.estimator.get_params(),
        }
