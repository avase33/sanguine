"""Gaussian Naive Bayes."""

from __future__ import annotations

import math

from .base import Estimator


class GaussianNB(Estimator):
    name = "gaussian_nb"

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_: list[int] | None = None
        self.priors: dict[int, float] = {}
        self.means: dict[int, list[float]] = {}
        self.vars: dict[int, list[float]] = {}

    def fit(self, X, y):
        d = len(X[0])
        self.classes_ = sorted(set(y))
        for c in self.classes_:
            rows = [X[i] for i in range(len(X)) if y[i] == c]
            self.priors[c] = len(rows) / len(X)
            self.means[c] = [sum(r[j] for r in rows) / len(rows) for j in range(d)]
            self.vars[c] = [
                (sum((r[j] - self.means[c][j]) ** 2 for r in rows) / len(rows)) + self.var_smoothing
                for j in range(d)
            ]
        return self

    def _log_likelihood(self, row: list[float], c: int) -> float:
        ll = math.log(self.priors[c])
        for j in range(len(row)):
            var = self.vars[c][j]
            ll += -0.5 * math.log(2 * math.pi * var) - (row[j] - self.means[c][j]) ** 2 / (2 * var)
        return ll

    def predict_proba(self, X):
        self._check_fitted("classes_")
        out = []
        for row in X:
            logs = {c: self._log_likelihood(row, c) for c in self.classes_}
            m = max(logs.values())
            exps = {c: math.exp(logs[c] - m) for c in logs}
            total = sum(exps.values())
            out.append(exps.get(1, 0.0) / total if total else 0.0)
        return out

    def get_params(self):
        return {"var_smoothing": self.var_smoothing}
