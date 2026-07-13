<div align="center">

# 🩸 Sanguine

### A from-scratch AutoML engine — for predicting blood donations.

Inspired by *"Give Life: Predict Blood Donations,"* Sanguine doesn't just call an AutoML library — it **implements the ML models *and* the pipeline search itself**, then evolves the best preprocessing + model + hyperparameters for the task.

[![CI](https://github.com/avase33/sanguine/actions/workflows/ci.yml/badge.svg)](https://github.com/avase33/sanguine/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![runtime deps](https://img.shields.io/badge/core%20runtime%20deps-0-brightgreen)](pyproject.toml)

</div>

---

The task: from a donor's **RFM history** — Recency, Frequency, Monetary, Time —
predict whether they'll donate again (the UCI Blood Transfusion dataset). The
original DataCamp project pipes this into [TPOT](https://github.com/EpistasisLab/tpot).
**Sanguine re-implements the whole stack** — the estimators, cross-validation,
metrics, and an **evolutionary pipeline search** — in pure Python, then serves
the winning model over an API.

```bash
pip install -e .
python -m sanguine train --md report.md
```

```
Evaluated 41 pipelines (metric=roc_auc, strategy=evolution)
Best: standard → random_forest(n_estimators=40, max_depth=6, ...)
  CV roc_auc = 0.74xx
  Test  roc_auc=0.72xx accuracy=0.78xx f1=0.4xxx
  Model saved to model.pkl
```

## ✨ What makes it stand out

| | |
|---|---|
| 🧬 **AutoML search, from scratch** | Evolutionary search (crossover + mutation + elitism) *and* random search over a scaler × model × hyperparameter space, with **memoized** CV scoring |
| 🤖 **5 models, hand-built** | Logistic regression (GD+L2), Gaussian Naive Bayes, k-NN, decision tree (Gini), random forest (bagging + feature subsampling) — plus a baseline |
| 📊 **Real evaluation** | Stratified k-fold CV, a **rank-based ROC AUC** (handles ties), precision/recall/F1, log-loss, confusion matrix — all from scratch |
| 🏆 **Leaderboard** | Every pipeline tried, ranked, with mean ± std CV score |
| 💾 **Persistence & serving** | Save/load the winning pipeline; predict via CLI or a **FastAPI** endpoint (Pydantic-validated) |
| 🔁 **Reproducible** | Everything seeded; ships a deterministic, schema-accurate dataset so it runs with **zero setup** |
| 🔬 **Benchmarkable** | Optional adapter runs **real scikit-learn / TPOT** for a head-to-head |

## 🚀 Usage

```bash
# Search for the best pipeline, save the model + a Markdown/JSON report:
python -m sanguine train --strategy evolution --metric roc_auc --md report.md --json report.json

# See the full leaderboard:
python -m sanguine leaderboard

# Predict for a donor (recency, frequency, monetary, time):
python -m sanguine predict --features 2,12,3000,30
#   P(donates) = 0.63  ->  DONATE

# Evaluate a saved model on the held-out split:
python -m sanguine evaluate --model model.pkl

# Use the authentic UCI data instead of the bundled synthetic set:
python -m sanguine data          # downloads it, then re-run train
```

### Serving API

```bash
pip install -e ".[server]"
uvicorn sanguine.service.api:app --reload        # http://127.0.0.1:8000/docs

curl -X POST localhost:8000/api/predict -H 'content-type: application/json' \
     -d '{"recency":2,"frequency":12,"monetary":3000,"time":30}'
# {"probability":0.63,"will_donate":true,"threshold":0.5}
```

### Library

```python
from sanguine.data import generate_synthetic, train_test_split
from sanguine.search import AutoML

ds = generate_synthetic()
Xtr, Xte, ytr, yte = train_test_split(ds.X, ds.y)
result = AutoML(metric="roc_auc", strategy="evolution").fit(Xtr, ytr)

print(result.best_pipeline.describe())          # e.g. "standard → random_forest(...)"
print(result.best_pipeline.predict_proba(Xte[:3]))
```

## 🧠 How the search works

A pipeline *genome* is `{scaler, model, hyperparameters}`. Evolution keeps the
top genomes each generation and breeds new ones by **crossover** (swap
scaler/model between parents) and **mutation** (swap a scaler/model or tweak one
hyperparameter). Each genome is scored by **stratified k-fold cross-validation**;
the winner is refit on all data and evaluated on a held-out test set. Full
walkthrough in [`docs/architecture.md`](docs/architecture.md).

## ⚔️ Compare against real TPOT / scikit-learn

```bash
pip install -e ".[tpot]"
python -c "from sanguine.sklearn_adapter import run_tpot; ..."   # head-to-head
```

## 🔬 Tested

Pure-Python tests (no deps, no network): metric correctness (ROC AUC edge cases),
every model beating chance, the AutoML search beating the baseline and
generalizing to held-out data, leaderboard ordering, reproducibility, and model
persistence round-trips. The API is tested with FastAPI's `TestClient`.

```bash
pip install -e ".[dev,server]"
pytest -q
```

CI runs the suite on Python 3.10–3.12, smoke-tests the CLI (train/predict/leaderboard),
and builds the Docker image.

## 🗺️ Roadmap

- [ ] Gradient-boosted trees + a stacking meta-learner
- [ ] Bayesian / TPE hyperparameter optimization
- [ ] Probability calibration (Platt / isotonic) for the imbalanced target
- [ ] Parallel CV evaluation
- [ ] Feature engineering search (interactions, binning)

## 📄 License

MIT © Akhil Vase — see [LICENSE](LICENSE). Dataset: UCI Blood Transfusion Service
Center (Yeh et al.).
