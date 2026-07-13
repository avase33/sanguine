from sanguine import metrics
from sanguine.data import (
    Dataset,
    generate_synthetic,
    stratified_kfold,
    train_test_split,
)


def test_roc_auc_edge_cases():
    assert metrics.roc_auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert metrics.roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert metrics.roc_auc([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_confusion_and_prf():
    y = [1, 1, 0, 0]
    p = [1, 0, 0, 0]
    cm = metrics.confusion_matrix(y, p)
    assert cm == {"tn": 2, "fp": 0, "fn": 1, "tp": 1}
    assert metrics.precision(y, p) == 1.0
    assert metrics.recall(y, p) == 0.5


def test_synthetic_is_deterministic_and_plausible():
    a = generate_synthetic(seed=42)
    b = generate_synthetic(seed=42)
    assert a.X == b.X and a.y == b.y
    assert a.n == 748
    assert 0.1 < a.positive_rate() < 0.4          # realistic imbalance
    # Monetary is exactly frequency * 250 (matches the real UCI dataset).
    assert all(row[2] == row[1] * 250 for row in a.X)


def test_train_test_split_is_stratified():
    ds = generate_synthetic()
    Xtr, Xte, ytr, yte = train_test_split(ds.X, ds.y, test_size=0.25)
    assert len(Xte) > 0 and len(Xtr) + len(Xte) == ds.n
    # class balance preserved within a few points
    assert abs((sum(yte) / len(yte)) - ds.positive_rate()) < 0.06


def test_kfold_partitions_all_rows():
    y = generate_synthetic().y
    folds = stratified_kfold(y, k=5)
    covered = sorted(i for _, test in folds for i in test)
    assert covered == list(range(len(y)))
