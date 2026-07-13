import tempfile
from pathlib import Path

from sanguine import report
from sanguine.data import generate_synthetic, train_test_split
from sanguine.persistence import load_model, save_model
from sanguine.search import AutoML


def _run(strategy="evolution"):
    ds = generate_synthetic()
    Xtr, Xte, ytr, yte = train_test_split(ds.X, ds.y, test_size=0.25)
    automl = AutoML(metric="roc_auc", cv=3, strategy=strategy, generations=3, population=6, random_iters=20)
    result = automl.fit(Xtr, ytr)
    return ds, result, (Xte, yte)


def test_evolution_search_beats_baseline_and_ranks():
    ds, result, (Xte, yte) = _run("evolution")
    # Leaderboard sorted descending by score.
    scores = [e.score for e in result.leaderboard]
    assert scores == sorted(scores, reverse=True)
    # Baseline is present and the winner clearly beats it.
    assert any(e.spec["model"] == "prior_baseline" for e in result.leaderboard)
    assert result.best_score > 0.62
    # The refit winner generalizes on held-out data.
    test = report.evaluate(result.best_pipeline, Xte, yte)
    assert test["roc_auc"] > 0.6


def test_random_search_also_works():
    _, result, _ = _run("random")
    assert result.best_score > 0.6
    assert result.n_evaluated >= 5


def test_history_is_monotonic_nondecreasing():
    _, result, _ = _run("random")
    bests = [h["best_score"] for h in result.history]
    assert all(b2 >= b1 for b1, b2 in zip(bests, bests[1:]))


def test_persistence_roundtrip():
    ds, result, (Xte, yte) = _run("evolution")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.pkl"
        save_model(result.best_pipeline, path)
        loaded, meta = load_model(path)
        assert loaded.predict_proba(Xte[:5]) == result.best_pipeline.predict_proba(Xte[:5])
        assert meta["spec"]["model"] == result.best_spec["model"]


def test_report_renders():
    ds, result, (Xte, yte) = _run("evolution")
    test = report.evaluate(result.best_pipeline, Xte, yte)
    md = report.to_markdown(result, test, ds)
    assert "# Sanguine AutoML report" in md and "Leaderboard" in md
    assert '"best_cv_score"' in report.to_json(result, test, ds)
