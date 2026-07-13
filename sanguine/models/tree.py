"""Decision tree classifier (CART with the Gini impurity criterion)."""

from __future__ import annotations

import random

from .base import Estimator


def _gini(y: list[int]) -> float:
    n = len(y)
    if n == 0:
        return 0.0
    p = sum(y) / n
    return 1.0 - (p * p + (1 - p) * (1 - p))


class DecisionTree(Estimator):
    name = "decision_tree"

    def __init__(self, max_depth: int = 5, min_samples_split: int = 10, min_samples_leaf: int = 3,
                 max_features: int | None = None, max_thresholds: int = 32, seed: int = 0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.seed = seed
        self.root: dict | None = None
        self._rng = random.Random(seed)

    def _candidate_thresholds(self, values: list[float]) -> list[float]:
        """Midpoints between adjacent sorted unique values, capped by quantiles."""
        uniq = sorted(set(values))
        mids = [(a + b) / 2 for a, b in zip(uniq, uniq[1:])]
        if len(mids) <= self.max_thresholds:
            return mids
        step = len(mids) / self.max_thresholds
        return [mids[int(i * step)] for i in range(self.max_thresholds)]

    def fit(self, X, y):
        self._rng = random.Random(self.seed)
        self.root = self._build([list(r) for r in X], list(y), 0)
        return self

    def _best_split(self, X, y):
        d = len(X[0])
        features = list(range(d))
        if self.max_features and self.max_features < d:
            features = self._rng.sample(features, self.max_features)
        parent = _gini(y)
        best = (None, None, 0.0)
        n = len(y)
        for f in features:
            for thr in self._candidate_thresholds([row[f] for row in X]):
                left_y = [y[i] for i in range(n) if X[i][f] <= thr]
                right_y = [y[i] for i in range(n) if X[i][f] > thr]
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                weighted = (len(left_y) * _gini(left_y) + len(right_y) * _gini(right_y)) / n
                gain = parent - weighted
                if gain > best[2]:
                    best = (f, thr, gain)
        return best

    def _build(self, X, y, depth) -> dict:
        value = sum(y) / len(y) if y else 0.0
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(set(y)) == 1:
            return {"leaf": True, "value": value}
        f, thr, gain = self._best_split(X, y)
        if f is None or gain <= 0:
            return {"leaf": True, "value": value}
        left_i = [i for i in range(len(X)) if X[i][f] <= thr]
        right_i = [i for i in range(len(X)) if X[i][f] > thr]
        return {
            "leaf": False, "feature": f, "threshold": thr, "value": value,
            "left": self._build([X[i] for i in left_i], [y[i] for i in left_i], depth + 1),
            "right": self._build([X[i] for i in right_i], [y[i] for i in right_i], depth + 1),
        }

    def _proba_one(self, row: list[float]) -> float:
        node = self.root
        while not node["leaf"]:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        return node["value"]

    def predict_proba(self, X):
        self._check_fitted("root")
        return [self._proba_one(row) for row in X]

    def get_params(self):
        return {"max_depth": self.max_depth, "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf, "max_features": self.max_features}
