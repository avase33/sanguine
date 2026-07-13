"""Estimator interface shared by every model (scikit-learn-like)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..errors import NotFittedError


class Estimator(ABC):
    """Base class: ``fit`` then ``predict`` / ``predict_proba``."""

    name: str = "estimator"

    @abstractmethod
    def fit(self, X: list[list[float]], y: list[int]) -> "Estimator": ...

    @abstractmethod
    def predict_proba(self, X: list[list[float]]) -> list[float]:
        """Return P(y=1) for each row."""

    def predict(self, X: list[list[float]], threshold: float = 0.5) -> list[int]:
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]

    def get_params(self) -> dict[str, Any]:
        return {}

    def _check_fitted(self, attr: str) -> None:
        if getattr(self, attr, None) is None:
            raise NotFittedError(f"{type(self).__name__} is not fitted yet")

    def describe(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.get_params().items())
        return f"{self.name}({params})"
