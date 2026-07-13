"""Optional adapter to run real scikit-learn / TPOT for comparison.

Sanguine's own engine is dependency-free; this module lets you benchmark it
against the industry-standard AutoML tool referenced by the original project.
Install extras first:  ``pip install 'sanguine-automl[sklearn]'``.
"""

from __future__ import annotations

from typing import Any


def run_sklearn_baseline(X_train, y_train, X_test, y_test) -> dict[str, Any]:  # pragma: no cover
    """Fit a couple of standard scikit-learn models and return their ROC AUC."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Install scikit-learn: pip install 'sanguine-automl[sklearn]'") from exc

    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
    out = {}
    for name, model in [("logistic", LogisticRegression(max_iter=1000)),
                        ("random_forest", RandomForestClassifier(random_state=42))]:
        model.fit(Xtr, y_train)
        proba = model.predict_proba(Xte)[:, 1]
        out[name] = round(float(roc_auc_score(y_test, proba)), 4)
    return out


def run_tpot(X_train, y_train, X_test, y_test, generations: int = 5, population_size: int = 20) -> dict[str, Any]:  # pragma: no cover
    """Run TPOT's genetic AutoML search and return its test ROC AUC + best pipeline."""
    try:
        from sklearn.metrics import roc_auc_score
        from tpot import TPOTClassifier
    except ImportError as exc:
        raise RuntimeError("Install TPOT: pip install 'sanguine-automl[tpot]'") from exc

    tpot = TPOTClassifier(generations=generations, population_size=population_size,
                          scoring="roc_auc", cv=5, random_state=42, verbosity=1)
    tpot.fit(X_train, y_train)
    proba = tpot.predict_proba(X_test)[:, 1]
    return {"tpot_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "best_pipeline": str(tpot.fitted_pipeline_)}
