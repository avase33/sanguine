"""Random search over pipeline specs — a strong, simple AutoML baseline."""

from __future__ import annotations

import random
from typing import Callable

from .space import random_spec


def random_search(evaluate: Callable[[dict], float], rng: random.Random, iters: int = 60) -> list[dict]:
    history: list[dict] = []
    best = float("-inf")
    for i in range(iters):
        best = max(best, evaluate(random_spec(rng)))
        history.append({"iter": i, "best_score": round(best, 4)})
    return history
