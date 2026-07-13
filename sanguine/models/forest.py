"""Random forest: bagged decision trees with feature subsampling."""

from __future__ import annotations

import math
import random

from .base import Estimator
from .tree import DecisionTree


class RandomForest(Estimator):
    name = "random_forest"

    def __init__(self, n_estimators: int = 25, max_depth: int = 6, min_samples_leaf: int = 3,
                 max_features: int | None = None, seed: int = 0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.seed = seed
        self.trees: list[DecisionTree] = []

    def fit(self, X, y):
        rng = random.Random(self.seed)
        n = len(X)
        d = len(X[0])
        mf = self.max_features or max(1, int(math.sqrt(d)))
        self.trees = []
        for t in range(self.n_estimators):
            idx = [rng.randrange(n) for _ in range(n)]  # bootstrap sample
            Xb = [X[i] for i in idx]
            yb = [y[i] for i in idx]
            tree = DecisionTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                max_features=mf, seed=self.seed + t)
            tree.fit(Xb, yb)
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        if not self.trees:
            raise RuntimeError("RandomForest is not fitted")
        per_tree = [tree.predict_proba(X) for tree in self.trees]
        m = len(self.trees)
        return [sum(per_tree[t][i] for t in range(m)) / m for i in range(len(X))]

    def get_params(self):
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf, "max_features": self.max_features}
