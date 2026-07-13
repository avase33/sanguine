"""From-scratch ML models with a scikit-learn-like interface."""

from .base import Estimator
from .baseline import PriorClassifier
from .forest import RandomForest
from .knn import KNeighborsClassifier
from .logistic import LogisticRegression
from .naive_bayes import GaussianNB
from .tree import DecisionTree

# Registry used by the AutoML search space and persistence.
MODELS: dict[str, type[Estimator]] = {
    "logistic": LogisticRegression,
    "gaussian_nb": GaussianNB,
    "knn": KNeighborsClassifier,
    "decision_tree": DecisionTree,
    "random_forest": RandomForest,
    "prior_baseline": PriorClassifier,
}

__all__ = [
    "Estimator", "LogisticRegression", "GaussianNB", "KNeighborsClassifier",
    "DecisionTree", "RandomForest", "PriorClassifier", "MODELS",
]
