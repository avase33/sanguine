"""API tests — require server extras (fastapi + httpx + pydantic)."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from sanguine.service.api import app  # noqa: E402


def test_predict_and_train():
    with TestClient(app) as client:  # triggers startup (quick training)
        assert client.get("/api/health").json()["model_ready"] is True

        r = client.post("/api/predict", json={"recency": 2, "frequency": 12, "monetary": 3000, "time": 30})
        body = r.json()
        assert 0.0 <= body["probability"] <= 1.0
        assert isinstance(body["will_donate"], bool)

        t = client.post("/api/train", json={"strategy": "random", "generations": 2, "population": 6})
        tb = t.json()
        assert tb["best_cv_score"] > 0.5
        assert len(tb["leaderboard"]) >= 1

        lb = client.get("/api/leaderboard").json()
        assert lb[0]["rank"] == 1
