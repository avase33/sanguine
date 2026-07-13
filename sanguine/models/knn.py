"""k-Nearest-Neighbours classifier."""

from __future__ import annotations

import math

from .base import Estimator


class KNeighborsClassifier(Estimator):
    name = "knn"

    def __init__(self, k: int = 5):
        self.k = k
        self.X_: list[list[float]] | None = None
        self.y_: list[int] | None = None

    def fit(self, X, y):
        self.X_ = [list(r) for r in X]
        self.y_ = list(y)
        return self

    @staticmethod
    def _dist2(a: list[float], b: list[float]) -> float:
        return sum((a[j] - b[j]) ** 2 for j in range(len(a)))

    def predict_proba(self, X):
        self._check_fitted("X_")
        k = min(self.k, len(self.X_))
        out = []
        for row in X:
            dists = sorted(range(len(self.X_)), key=lambda i: self._dist2(row, self.X_[i]))[:k]
            out.append(sum(self.y_[i] for i in dists) / k)
        return out

    def get_params(self):
        return {"k": self.k}
