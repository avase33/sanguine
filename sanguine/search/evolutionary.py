"""Evolutionary (genetic) search over pipeline specs.

Keeps an elite set each generation and breeds the next population by crossover
and mutation. Because scoring is memoized upstream, re-evaluating survivors is
free, so the search spends its budget exploring new pipelines.
"""

from __future__ import annotations

import random
from typing import Callable

from .space import crossover, mutate, random_spec


def evolve(
    evaluate: Callable[[dict], float],
    rng: random.Random,
    generations: int = 6,
    population: int = 12,
    elite: int = 3,
) -> list[dict]:
    pop = [random_spec(rng) for _ in range(population)]
    history: list[dict] = []

    for gen in range(generations):
        scored = sorted(((evaluate(s), s) for s in pop), key=lambda x: -x[0])
        history.append({"generation": gen, "best_score": round(scored[0][0], 4)})
        elites = [s for _, s in scored[:elite]]

        children = [dict(e) for e in elites]
        while len(children) < population:
            if rng.random() < 0.6 and len(elites) >= 2:
                a, b = rng.sample(elites, 2)
                children.append(crossover(a, b, rng))
            else:
                children.append(mutate(rng.choice(elites), rng))
        pop = children

    for s in pop:  # score the final generation
        evaluate(s)
    return history
