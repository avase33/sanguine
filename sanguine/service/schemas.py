"""Pydantic schemas for the prediction API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DonorFeatures(BaseModel):
    recency: float = Field(..., ge=0, description="Months since last donation", examples=[2])
    frequency: float = Field(..., ge=0, description="Total number of donations", examples=[10])
    monetary: float = Field(..., ge=0, description="Total blood donated (c.c.)", examples=[2500])
    time: float = Field(..., ge=0, description="Months since first donation", examples=[28])

    def as_row(self) -> list[float]:
        return [self.recency, self.frequency, self.monetary, self.time]


class PredictionResponse(BaseModel):
    probability: float
    will_donate: bool
    threshold: float = 0.5


class TrainRequest(BaseModel):
    metric: Literal["roc_auc", "accuracy", "f1"] = "roc_auc"
    strategy: Literal["evolution", "random"] = "evolution"
    generations: int = Field(default=6, ge=1, le=20)
    population: int = Field(default=12, ge=4, le=40)
    cv: int = Field(default=5, ge=2, le=10)


class LeaderboardRow(BaseModel):
    rank: int
    scaler: str
    model: str
    score: float
    std: float


class TrainResponse(BaseModel):
    metric: str
    best_pipeline: str
    best_cv_score: float
    test_metrics: dict
    n_evaluated: int
    leaderboard: list[LeaderboardRow]
