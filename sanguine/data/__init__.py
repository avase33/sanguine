"""Data layer: dataset loading/generation, splitting, preprocessing."""

from .dataset import (
    FEATURES,
    TARGET,
    Dataset,
    download_real_dataset,
    generate_synthetic,
    load_csv,
    load_dataset,
    stratified_kfold,
    train_test_split,
)
from .preprocess import (
    Identity,
    LogScaler,
    MinMaxScaler,
    StandardScaler,
    Transformer,
    make_scaler,
)

__all__ = [
    "FEATURES", "TARGET", "Dataset", "load_dataset", "load_csv", "generate_synthetic",
    "download_real_dataset", "train_test_split", "stratified_kfold",
    "Transformer", "Identity", "StandardScaler", "MinMaxScaler", "LogScaler", "make_scaler",
]
