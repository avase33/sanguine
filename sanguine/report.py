"""Build evaluation + leaderboard reports (Markdown / JSON)."""

from __future__ import annotations

import json

from . import metrics
from .data.dataset import Dataset
from .pipeline import Pipeline
from .search.automl import AutoMLResult


def evaluate(pipeline: Pipeline, X_test, y_test, threshold: float = 0.5) -> dict:
    proba = pipeline.predict_proba(X_test)
    return metrics.full_report(y_test, proba, threshold)


def to_markdown(result: AutoMLResult, test_metrics: dict, dataset: Dataset, top_n: int = 10) -> str:
    s = dataset.summary()
    lines = [
        "# Sanguine AutoML report",
        "",
        f"**Task:** predict blood donation · **Optimized metric:** `{result.metric}`",
        "",
        "## Dataset",
        f"- Rows: **{s['rows']}** · Features: {', '.join(s['features'])}",
        f"- Positive rate: **{s['positive_rate']:.1%}** (class counts {s['class_counts']})",
        "",
        "## Winning pipeline",
        f"```\n{result.best_pipeline.describe()}\n```",
        f"- Cross-validated {result.metric}: **{result.best_score:.4f}**",
        f"- Pipelines evaluated: **{result.n_evaluated}**",
        "",
        "## Held-out test metrics",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"):
        lines.append(f"| {k} | {test_metrics[k]:.4f} |")
    cm = test_metrics["confusion_matrix"]
    lines += [
        "",
        f"Confusion matrix — TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']}",
        "",
        "## Leaderboard (top pipelines)",
        "",
        f"| rank | scaler | model | CV {result.metric} | ± std |",
        "|---|---|---|---|---|",
    ]
    for e in result.leaderboard[:top_n]:
        lines.append(f"| {e.rank} | {e.spec['scaler']} | {e.spec['model']} | "
                     f"{e.score:.4f} | {e.std:.4f} |")
    lines += ["", "## Search progress", ""]
    for h in result.history:
        gen = h.get("generation", h.get("iter"))
        lines.append(f"- step {gen}: best {result.metric} = {h['best_score']}")
    return "\n".join(lines) + "\n"


def to_json(result: AutoMLResult, test_metrics: dict, dataset: Dataset) -> str:
    return json.dumps({
        "metric": result.metric,
        "dataset": dataset.summary(),
        "best_pipeline": result.best_pipeline.to_spec(),
        "best_cv_score": round(result.best_score, 4),
        "n_evaluated": result.n_evaluated,
        "test_metrics": test_metrics,
        "leaderboard": [e.to_dict() for e in result.leaderboard],
        "history": result.history,
    }, indent=2)
