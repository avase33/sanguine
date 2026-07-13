"""The AutoML search space: scalers × models × hyperparameters.

A *spec* (genome) is ``{"scaler": name, "model": name, "params": {...}}``.
This module can sample, mutate and cross over specs, and instantiate a runnable
:class:`~sanguine.pipeline.Pipeline` from one.
"""

from __future__ import annotations

import random
from typing import Any

from ..data.preprocess import make_scaler
from ..models import MODELS
from ..pipeline import Pipeline

SCALER_CHOICES = ["identity", "standard", "minmax", "log"]

MODEL_SPACE: dict[str, dict[str, list]] = {
    "logistic": {"lr": [0.05, 0.1, 0.5, 1.0], "epochs": [100, 200, 400], "l2": [0.0, 0.001, 0.01, 0.1]},
    "gaussian_nb": {"var_smoothing": [1e-9, 1e-6, 1e-3]},
    "knn": {"k": [3, 5, 7, 11, 15, 21]},
    "decision_tree": {"max_depth": [3, 4, 5, 6, 8], "min_samples_split": [5, 10, 20],
                      "min_samples_leaf": [1, 3, 5]},
    "random_forest": {"n_estimators": [15, 25, 40], "max_depth": [4, 6, 8],
                      "min_samples_leaf": [1, 3, 5]},
}


def random_params(model: str, rng: random.Random) -> dict[str, Any]:
    return {k: rng.choice(v) for k, v in MODEL_SPACE[model].items()}


def random_spec(rng: random.Random) -> dict[str, Any]:
    model = rng.choice(list(MODEL_SPACE))
    return {"scaler": rng.choice(SCALER_CHOICES), "model": model, "params": random_params(model, rng)}


def build_pipeline(spec: dict[str, Any]) -> Pipeline:
    scaler = make_scaler(spec["scaler"])
    estimator = MODELS[spec["model"]](**spec.get("params", {}))
    return Pipeline(scaler, estimator, spec=spec)


def mutate(spec: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    s = {"scaler": spec["scaler"], "model": spec["model"], "params": dict(spec["params"])}
    roll = rng.random()
    if roll < 0.3:
        s["scaler"] = rng.choice(SCALER_CHOICES)
    elif roll < 0.5:
        s["model"] = rng.choice(list(MODEL_SPACE))
        s["params"] = random_params(s["model"], rng)
    else:
        # tweak a single hyperparameter
        options = MODEL_SPACE[s["model"]]
        if options:
            key = rng.choice(list(options))
            s["params"][key] = rng.choice(options[key])
    return s


def crossover(a: dict[str, Any], b: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    # Take the scaler from one parent and the model+params from the other.
    if rng.random() < 0.5:
        return {"scaler": a["scaler"], "model": b["model"], "params": dict(b["params"])}
    return {"scaler": b["scaler"], "model": a["model"], "params": dict(a["params"])}


def spec_key(spec: dict[str, Any]) -> str:
    params = ",".join(f"{k}={spec['params'][k]}" for k in sorted(spec["params"]))
    return f"{spec['scaler']}|{spec['model']}|{params}"
