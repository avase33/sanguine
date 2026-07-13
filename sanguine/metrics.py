"""Classification metrics, all pure Python.

Includes a rank-based ROC AUC (Mann–Whitney U) that handles ties, so the AutoML
search can optimize a threshold-independent objective — important for the
imbalanced blood-donation task.
"""

from __future__ import annotations

import math


def confusion_matrix(y_true: list[int], y_pred: list[int]) -> dict[str, int]:
    tn = fp = fn = tp = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 1 and p == 0:
            fn += 1
        elif t == 0 and p == 1:
            fp += 1
        else:
            tn += 1
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def precision(y_true: list[int], y_pred: list[int]) -> float:
    cm = confusion_matrix(y_true, y_pred)
    denom = cm["tp"] + cm["fp"]
    return cm["tp"] / denom if denom else 0.0


def recall(y_true: list[int], y_pred: list[int]) -> float:
    cm = confusion_matrix(y_true, y_pred)
    denom = cm["tp"] + cm["fn"]
    return cm["tp"] / denom if denom else 0.0


def f1(y_true: list[int], y_pred: list[int]) -> float:
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def roc_auc(y_true: list[int], y_score: list[float]) -> float:
    pos = [s for t, s in zip(y_true, y_score) if t == 1]
    neg = [s for t, s in zip(y_true, y_score) if t == 0]
    if not pos or not neg:
        return 0.5
    # Average-rank Mann–Whitney U.
    paired = sorted(zip(y_score, y_true), key=lambda x: x[0])
    ranks = [0.0] * len(paired)
    i = 0
    while i < len(paired):
        j = i
        while j < len(paired) and paired[j][0] == paired[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-based average rank for the tie group
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    rank_sum_pos = sum(ranks[k] for k in range(len(paired)) if paired[k][1] == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def log_loss(y_true: list[int], y_score: list[float], eps: float = 1e-15) -> float:
    total = 0.0
    for t, p in zip(y_true, y_score):
        p = min(max(p, eps), 1 - eps)
        total += -(t * math.log(p) + (1 - t) * math.log(1 - p))
    return total / len(y_true)


def score(name: str, y_true: list[int], y_score: list[float], threshold: float = 0.5) -> float:
    """Score by metric name; probability metrics use ``y_score`` directly."""
    if name == "roc_auc":
        return roc_auc(y_true, y_score)
    if name == "log_loss":
        return -log_loss(y_true, y_score)  # higher is better for search
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}[name](y_true, y_pred)


def full_report(y_true: list[int], y_score: list[float], threshold: float = 0.5) -> dict:
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    return {
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "precision": round(precision(y_true, y_pred), 4),
        "recall": round(recall(y_true, y_pred), 4),
        "f1": round(f1(y_true, y_pred), 4),
        "roc_auc": round(roc_auc(y_true, y_score), 4),
        "log_loss": round(log_loss(y_true, y_score), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
