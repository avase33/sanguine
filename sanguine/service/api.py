"""FastAPI prediction service for Sanguine.

  GET  /api/health
  POST /api/predict      -> P(donates) for a donor's RFM features
  POST /api/train        -> run AutoML, hot-swap the served model
  GET  /api/leaderboard  -> the current leaderboard
  GET  /api/model        -> the served pipeline's spec

Run with:  uvicorn sanguine.service.api:app  (or `sanguine serve`).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .. import report
from ..data import load_dataset, train_test_split
from ..search import AutoML
from .schemas import (
    DonorFeatures,
    LeaderboardRow,
    PredictionResponse,
    TrainRequest,
    TrainResponse,
)

app = FastAPI(title="Sanguine", version="0.1.0",
              description="AutoML blood-donation predictor (models + pipeline search built from scratch).")

# In-memory served model + last training result.
_STATE: dict = {"pipeline": None, "result": None, "test_metrics": None}


def _train(req: TrainRequest) -> TrainResponse:
    ds = load_dataset()
    Xtr, Xte, ytr, yte = train_test_split(ds.X, ds.y, test_size=0.25, seed=42)
    automl = AutoML(metric=req.metric, cv=req.cv, strategy=req.strategy,
                    generations=req.generations, population=req.population)
    result = automl.fit(Xtr, ytr)
    test_metrics = report.evaluate(result.best_pipeline, Xte, yte)
    _STATE.update(pipeline=result.best_pipeline, result=result, test_metrics=test_metrics)
    return TrainResponse(
        metric=result.metric,
        best_pipeline=result.best_pipeline.describe(),
        best_cv_score=round(result.best_score, 4),
        test_metrics=test_metrics,
        n_evaluated=result.n_evaluated,
        leaderboard=[LeaderboardRow(rank=e.rank, scaler=e.spec["scaler"], model=e.spec["model"],
                                    score=round(e.score, 4), std=round(e.std, 4))
                     for e in result.leaderboard[:10]],
    )


@app.on_event("startup")
def _startup() -> None:
    if _STATE["pipeline"] is None:
        _train(TrainRequest(generations=4, population=8))  # quick default model


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model_ready": _STATE["pipeline"] is not None}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(donor: DonorFeatures) -> PredictionResponse:
    if _STATE["pipeline"] is None:
        _train(TrainRequest(generations=4, population=8))
    proba = _STATE["pipeline"].predict_proba([donor.as_row()])[0]
    return PredictionResponse(probability=round(proba, 4), will_donate=proba >= 0.5)


@app.post("/api/train", response_model=TrainResponse)
def train(req: TrainRequest) -> TrainResponse:
    return _train(req)


@app.get("/api/leaderboard")
def leaderboard() -> list[dict]:
    if not _STATE["result"]:
        raise HTTPException(status_code=404, detail="no training run yet")
    return [e.to_dict() for e in _STATE["result"].leaderboard[:15]]


@app.get("/api/model")
def model() -> dict:
    if _STATE["pipeline"] is None:
        raise HTTPException(status_code=404, detail="no model")
    return {"pipeline": _STATE["pipeline"].to_spec(),
            "test_metrics": _STATE["test_metrics"]}
