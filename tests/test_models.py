from sanguine.data import StandardScaler, generate_synthetic, train_test_split
from sanguine.metrics import roc_auc
from sanguine.models import (
    DecisionTree,
    GaussianNB,
    KNeighborsClassifier,
    LogisticRegression,
    PriorClassifier,
    RandomForest,
)


def _split():
    ds = generate_synthetic()
    Xtr, Xte, ytr, yte = train_test_split(ds.X, ds.y, test_size=0.25)
    scaler = StandardScaler().fit(Xtr)
    return scaler.transform(Xtr), ytr, scaler.transform(Xte), yte


def test_baseline_scores_half():
    Xtr, ytr, Xte, yte = _split()
    model = PriorClassifier().fit(Xtr, ytr)
    assert abs(roc_auc(yte, model.predict_proba(Xte)) - 0.5) < 1e-9


def test_all_models_beat_chance():
    Xtr, ytr, Xte, yte = _split()
    for model in [
        LogisticRegression(epochs=200),
        GaussianNB(),
        KNeighborsClassifier(k=11),
        DecisionTree(max_depth=5),
        RandomForest(n_estimators=20, max_depth=6),
    ]:
        model.fit(Xtr, ytr)
        auc = roc_auc(yte, model.predict_proba(Xte))
        assert auc > 0.55, f"{model.name} auc={auc:.3f}"


def test_logistic_learns_signal():
    Xtr, ytr, Xte, yte = _split()
    model = LogisticRegression(epochs=400, lr=0.5).fit(Xtr, ytr)
    assert roc_auc(yte, model.predict_proba(Xte)) > 0.62


def test_predict_thresholding():
    model = LogisticRegression(epochs=50)
    model.fit([[0.0], [1.0]], [0, 1])
    preds = model.predict([[0.0], [1.0]])
    assert preds[0] in (0, 1) and preds[1] in (0, 1)
