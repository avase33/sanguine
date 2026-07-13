# Sanguine architecture

Everything — the ML models *and* the AutoML search — is implemented from scratch
in pure Python, so the whole engine has zero runtime dependencies and is fully
testable offline.

```
                 CLI  ·  FastAPI service
                        │
                 ┌──────▼───────┐
                 │   AutoML      │  search/automl.py
                 │  orchestrator │  (CV scoring, memoized, leaderboard)
                 └──────┬───────┘
             ┌──────────┴───────────┐
             ▼                      ▼
   Search strategies          Search space
   evolutionary.py            space.py
   random_search.py           (scaler × model × params → genome)
             │                      │
             └──────────┬───────────┘
                        ▼
                    Pipeline               pipeline.py
                 scaler → estimator
             ┌──────────┴───────────┐
             ▼                      ▼
        Preprocessing            Models (from scratch)
        standard/minmax/log      logistic · gaussian_nb · knn
                                 decision_tree · random_forest · baseline
                        │
                        ▼
                     Metrics                metrics.py
             accuracy · f1 · ROC AUC · log-loss

  Data: UCI blood-transfusion loader + deterministic synthetic generator + k-fold
  Persistence: pickle model files · Reporting: Markdown/JSON leaderboard + metrics
```

## How the AutoML search works

1. **Search space** — a *genome* is `{scaler, model, params}`. The space spans 4
   preprocessing choices, 5 model families, and their hyperparameter grids.
2. **Scoring** — each genome is scored by **stratified k-fold cross-validation**
   on the training split, optimizing a configurable metric (default ROC AUC).
   Scores are **memoized** by genome, so elitism and duplicate proposals are free.
3. **Strategy** —
   - *evolution*: keep an elite set each generation; breed the next population by
     **crossover** (swap scaler/model between parents) and **mutation** (swap a
     scaler, a model, or tweak one hyperparameter).
   - *random*: independent random genomes — a strong baseline.
4. **Selection** — the highest-CV genome is **refit on all training data** and
   returned as the winning `Pipeline`; everything tried forms the leaderboard.
5. **Evaluation** — the winner is scored on a **held-out test split** (accuracy,
   precision, recall, F1, ROC AUC, log-loss, confusion matrix).

## Why from scratch (vs. just calling TPOT)

The original "Give Life" project pipes data into TPOT. Sanguine re-implements the
whole thing — the estimators, cross-validation, metrics, and an evolutionary
search — to make the mechanics explicit and dependency-free. An optional
[`sklearn_adapter.py`](../sanguine/sklearn_adapter.py) still lets you benchmark
against real scikit-learn and TPOT for comparison.

## Reproducibility

Every stochastic component (dataset generation, splits, folds, search) is seeded,
so a run is bit-for-bit reproducible. `monetary = frequency × 250` mirrors the
real UCI dataset exactly.
