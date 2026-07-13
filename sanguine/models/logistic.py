"""Logistic regression trained with batch gradient descent + L2 regularization."""

from __future__ import annotations

import math

from .base import Estimator


def _sigmoid(z: float) -> float:
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


class LogisticRegression(Estimator):
    name = "logistic"

    def __init__(self, lr: float = 0.5, epochs: int = 300, l2: float = 0.0):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.w: list[float] | None = None
        self.b: float = 0.0

    def fit(self, X, y):
        n, d = len(X), len(X[0])
        self.w = [0.0] * d
        self.b = 0.0
        for _ in range(self.epochs):
            gw = [0.0] * d
            gb = 0.0
            for i in range(n):
                z = self.b + sum(self.w[j] * X[i][j] for j in range(d))
                err = _sigmoid(z) - y[i]
                gb += err
                for j in range(d):
                    gw[j] += err * X[i][j]
            for j in range(d):
                self.w[j] -= self.lr * (gw[j] / n + self.l2 * self.w[j])
            self.b -= self.lr * gb / n
        return self

    def predict_proba(self, X):
        self._check_fitted("w")
        return [_sigmoid(self.b + sum(self.w[j] * row[j] for j in range(len(row)))) for row in X]

    def get_params(self):
        return {"lr": self.lr, "epochs": self.epochs, "l2": self.l2}
