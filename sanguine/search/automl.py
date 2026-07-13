"""The AutoML orchestrator.

Searches the pipeline space with cross-validated scoring, keeps a leaderboard of
everything tried, and refits the winning pipeline on all the data. Search is
memoized, so evolution's elitism and any duplicate proposals cost nothing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .. import metrics
from ..data.dataset import stratified_kfold
from ..pipeline import Pipeline
from .evolutionary import evolve
from .random_search import random_search
from .space import build_pipeline, spec_key


@dataclass
class LeaderboardEntry:
    rank: int
    spec: dict
    score: float
    std: float

    def to_dict(self) -> dict:
        return {"rank": self.rank, "scaler": self.spec["scaler"], "model": self.spec["model"],
                "params": self.spec["params"], "score": round(self.score, 4), "std": round(self.std, 4)}


@dataclass
class AutoMLResult:
    metric: str
    best_pipeline: Pipeline
    best_spec: dict
    best_score: float
    leaderboard: list[LeaderboardEntry]
    history: list[dict]
    n_evaluated: int = 0
    field_seed: int = field(default=0)


class AutoML:
    def __init__(self, metric: str = "roc_auc", cv: int = 5, seed: int = 42,
                 strategy: str = "evolution", generations: int = 6, population: int = 12,
                 random_iters: int = 60):
        self.metric = metric
        self.cv = cv
        self.seed = seed
        self.strategy = strategy
        self.generations = generations
        self.population = population
        self.random_iters = random_iters

    def cross_val_score(self, spec: dict, X, y) -> tuple[float, float]:
        folds = stratified_kfold(y, self.cv, self.seed)
        scores = []
        for train_idx, test_idx in folds:
            Xtr = [X[i] for i in train_idx]; ytr = [y[i] for i in train_idx]
            Xte = [X[i] for i in test_idx]; yte = [y[i] for i in test_idx]
            pipe = build_pipeline(spec).fit(Xtr, ytr)
            proba = pipe.predict_proba(Xte)
            scores.append(metrics.score(self.metric, yte, proba))
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        return mean, var ** 0.5

    def fit(self, X, y) -> AutoMLResult:
        rng = random.Random(self.seed)
        cache: dict[str, tuple[dict, float, float]] = {}

        def evaluate(spec: dict) -> float:
            key = spec_key(spec)
            if key not in cache:
                mean, std = self.cross_val_score(spec, X, y)
                cache[key] = (spec, mean, std)
            return cache[key][1]

        if self.strategy == "random":
            history = random_search(evaluate, rng, self.random_iters)
        else:
            history = evolve(evaluate, rng, self.generations, self.population)

        # Always include the majority-class baseline for context.
        evaluate({"scaler": "identity", "model": "prior_baseline", "params": {}})

        ranked = sorted(cache.values(), key=lambda t: -t[1])
        leaderboard = [LeaderboardEntry(i + 1, s, m, sd) for i, (s, m, sd) in enumerate(ranked)]
        best_spec, best_score, _ = ranked[0]
        best_pipeline = build_pipeline(best_spec).fit(X, y)

        return AutoMLResult(
            metric=self.metric, best_pipeline=best_pipeline, best_spec=best_spec,
            best_score=best_score, leaderboard=leaderboard, history=history,
            n_evaluated=len(cache), field_seed=self.seed,
        )
