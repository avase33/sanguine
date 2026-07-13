"""Feature preprocessing transformers (fit/transform), all pure Python."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class Transformer(ABC):
    @abstractmethod
    def fit(self, X: list[list[float]]) -> "Transformer": ...

    @abstractmethod
    def transform(self, X: list[list[float]]) -> list[list[float]]: ...

    def fit_transform(self, X: list[list[float]]) -> list[list[float]]:
        return self.fit(X).transform(X)

    def get_state(self) -> dict:
        return {"type": type(self).__name__}

    @property
    def name(self) -> str:
        return type(self).__name__


class Identity(Transformer):
    def fit(self, X):
        return self

    def transform(self, X):
        return [list(row) for row in X]


class StandardScaler(Transformer):
    """Zero-mean, unit-variance per feature."""

    def __init__(self):
        self.mean_: list[float] = []
        self.std_: list[float] = []

    def fit(self, X):
        n, d = len(X), len(X[0])
        self.mean_ = [sum(row[j] for row in X) / n for j in range(d)]
        self.std_ = []
        for j in range(d):
            var = sum((row[j] - self.mean_[j]) ** 2 for row in X) / n
            self.std_.append(math.sqrt(var) or 1.0)
        return self

    def transform(self, X):
        return [[(row[j] - self.mean_[j]) / self.std_[j] for j in range(len(row))] for row in X]

    def get_state(self):
        return {"type": "StandardScaler", "mean": self.mean_, "std": self.std_}


class MinMaxScaler(Transformer):
    """Scale each feature to [0, 1]."""

    def __init__(self):
        self.min_: list[float] = []
        self.range_: list[float] = []

    def fit(self, X):
        d = len(X[0])
        self.min_ = [min(row[j] for row in X) for j in range(d)]
        self.range_ = [(max(row[j] for row in X) - self.min_[j]) or 1.0 for j in range(d)]
        return self

    def transform(self, X):
        return [[(row[j] - self.min_[j]) / self.range_[j] for j in range(len(row))] for row in X]

    def get_state(self):
        return {"type": "MinMaxScaler", "min": self.min_, "range": self.range_}


class LogScaler(Transformer):
    """log1p transform then standardize — good for skewed RFM counts."""

    def __init__(self):
        self._std = StandardScaler()

    @staticmethod
    def _log(X):
        return [[math.log1p(max(v, 0.0)) for v in row] for row in X]

    def fit(self, X):
        self._std.fit(self._log(X))
        return self

    def transform(self, X):
        return self._std.transform(self._log(X))

    def get_state(self):
        return {"type": "LogScaler", "std": self._std.get_state()}


SCALERS: dict[str, type[Transformer]] = {
    "identity": Identity,
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "log": LogScaler,
}


def make_scaler(name: str) -> Transformer:
    if name not in SCALERS:
        raise ValueError(f"unknown scaler '{name}'. Options: {sorted(SCALERS)}")
    return SCALERS[name]()
