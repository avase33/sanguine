"""Dataset loading, synthetic generation, splitting and cross-validation folds.

The headline task is the UCI **Blood Transfusion Service Center** dataset (the
"Give Life" project): predict whether a donor gives blood in a target window from
their RFM history — Recency, Frequency, Monetary, Time.

To stay dependency-free and reproducible offline, Sanguine ships a deterministic
**synthetic** dataset with the exact schema and realistic structure
(``monetary = frequency * 250``, ~24% positive rate). Call
:func:`download_real_dataset` to fetch the authentic UCI data.
"""

from __future__ import annotations

import csv
import math
import random
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import DataError

FEATURES = ["recency", "frequency", "monetary", "time"]
TARGET = "donated"

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data"
_BUNDLED = Path(__file__).resolve().parent / "transfusion.csv"


@dataclass
class Dataset:
    X: list[list[float]]
    y: list[int]
    feature_names: list[str] = field(default_factory=lambda: list(FEATURES))
    name: str = "dataset"

    @property
    def n(self) -> int:
        return len(self.X)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def positive_rate(self) -> float:
        return sum(self.y) / len(self.y) if self.y else 0.0

    def summary(self) -> dict:
        return {"name": self.name, "rows": self.n, "features": self.feature_names,
                "positive_rate": round(self.positive_rate(), 4),
                "class_counts": {0: self.y.count(0), 1: self.y.count(1)}}


def _sigmoid(z: float) -> float:
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def generate_synthetic(n: int = 748, seed: int = 42) -> Dataset:
    """Deterministic RFM dataset with a learnable donation signal."""
    rng = random.Random(seed)
    X: list[list[float]] = []
    y: list[int] = []
    for _ in range(n):
        frequency = min(1 + int(rng.expovariate(1 / 5.0)), 50)
        time = min(max(frequency + int(rng.expovariate(1 / 12.0)) + rng.randint(0, 12), frequency), 98)
        recency = rng.randint(0, min(time, 40))
        monetary = frequency * 250
        # Recent, frequent donors are more likely to donate again.
        z = -1.2 - 0.09 * recency + 0.18 * frequency - 0.02 * time
        label = 1 if rng.random() < _sigmoid(z) else 0
        X.append([float(recency), float(frequency), float(monetary), float(time)])
        y.append(label)
    return Dataset(X=X, y=y, feature_names=list(FEATURES), name="transfusion-synthetic")


def load_csv(path: str | Path) -> Dataset:
    rows = list(csv.reader(open(path, newline="", encoding="utf-8")))
    if not rows:
        raise DataError(f"empty CSV: {path}")
    header = rows[0]
    start = 0
    feature_names = list(FEATURES)
    try:
        [float(c) for c in header]  # numeric first row -> no header
    except ValueError:
        start = 1
        feature_names = [h.strip() for h in header[:-1]] or list(FEATURES)
    X: list[list[float]] = []
    y: list[int] = []
    for r in rows[start:]:
        if not r:
            continue
        try:
            values = [float(c) for c in r]
        except ValueError as exc:
            raise DataError(f"non-numeric row in {path}: {r}") from exc
        X.append(values[:-1])
        y.append(int(values[-1]))
    return Dataset(X=X, y=y, feature_names=feature_names[: len(X[0])], name=Path(path).stem)


def load_dataset(path: str | Path | None = None) -> Dataset:
    """Load the dataset: explicit path → bundled real CSV → synthetic fallback."""
    if path:
        return load_csv(path)
    if _BUNDLED.exists():
        return load_csv(_BUNDLED)
    return generate_synthetic()


def download_real_dataset(dest: str | Path = _BUNDLED) -> Path:  # pragma: no cover - network
    """Fetch the authentic UCI blood-transfusion dataset and save it as CSV."""
    dest = Path(dest)
    with urllib.request.urlopen(UCI_URL, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(raw, encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------
# Splitting & cross-validation
# ---------------------------------------------------------------------------
def train_test_split(
    X: list[list[float]], y: list[int], test_size: float = 0.25, seed: int = 42, stratify: bool = True
):
    idx = list(range(len(X)))
    rng = random.Random(seed)
    if stratify:
        pos = [i for i in idx if y[i] == 1]
        neg = [i for i in idx if y[i] == 0]
        rng.shuffle(pos)
        rng.shuffle(neg)
        n_pos_test = int(len(pos) * test_size)
        n_neg_test = int(len(neg) * test_size)
        test_idx = set(pos[:n_pos_test] + neg[:n_neg_test])
    else:
        rng.shuffle(idx)
        test_idx = set(idx[: int(len(idx) * test_size)])
    Xtr, ytr, Xte, yte = [], [], [], []
    for i in idx:
        if i in test_idx:
            Xte.append(X[i]); yte.append(y[i])
        else:
            Xtr.append(X[i]); ytr.append(y[i])
    return Xtr, Xte, ytr, yte


def stratified_kfold(y: list[int], k: int = 5, seed: int = 42) -> list[tuple[list[int], list[int]]]:
    """Return k (train_idx, test_idx) folds preserving class balance."""
    rng = random.Random(seed)
    pos = [i for i in range(len(y)) if y[i] == 1]
    neg = [i for i in range(len(y)) if y[i] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    folds: list[list[int]] = [[] for _ in range(k)]
    for j, i in enumerate(pos):
        folds[j % k].append(i)
    for j, i in enumerate(neg):
        folds[j % k].append(i)
    result = []
    for f in range(k):
        test_idx = folds[f]
        train_idx = [i for g in range(k) if g != f for i in folds[g]]
        result.append((train_idx, test_idx))
    return result
