"""Save and load a fitted pipeline (portable, stdlib pickle)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from .errors import PersistenceError
from .pipeline import Pipeline
from .version import __version__

_MAGIC = "sanguine-model"


def save_model(pipeline: Pipeline, path: str | Path, meta: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    payload = {"magic": _MAGIC, "version": __version__, "spec": pipeline.to_spec(),
               "pipeline": pipeline, "meta": meta or {}}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(payload, fh)
    except Exception as exc:  # noqa: BLE001
        raise PersistenceError(f"failed to save model: {exc}") from exc
    return path


def load_model(path: str | Path) -> tuple[Pipeline, dict[str, Any]]:
    try:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
    except Exception as exc:  # noqa: BLE001
        raise PersistenceError(f"failed to load model: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("magic") != _MAGIC:
        raise PersistenceError("not a Sanguine model file")
    return payload["pipeline"], {"version": payload.get("version"), "spec": payload.get("spec"),
                                 "meta": payload.get("meta", {})}
