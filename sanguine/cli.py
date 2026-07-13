"""Command-line interface for Sanguine.

    sanguine train                         # run AutoML, save the best model + report
    sanguine leaderboard                   # search and print the pipeline leaderboard
    sanguine predict --features 2,10,2500,28
    sanguine evaluate --model model.pkl
    sanguine data                          # download the real UCI dataset
    sanguine serve                         # run the prediction API
"""

from __future__ import annotations

import argparse
import sys

from . import report
from .data import FEATURES, download_real_dataset, load_dataset, train_test_split
from .persistence import load_model, save_model
from .search import AutoML
from .version import __version__


def _train(args):
    ds = load_dataset(args.data)
    Xtr, Xte, ytr, yte = train_test_split(ds.X, ds.y, test_size=0.25, seed=args.seed)
    automl = AutoML(metric=args.metric, cv=args.cv, seed=args.seed, strategy=args.strategy,
                    generations=args.generations, population=args.population, random_iters=args.iters)
    result = automl.fit(Xtr, ytr)
    test_metrics = report.evaluate(result.best_pipeline, Xte, yte)
    return ds, result, test_metrics, (Xte, yte)


def cmd_train(args) -> int:
    ds, result, test_metrics, _ = _train(args)
    print(f"\nEvaluated {result.n_evaluated} pipelines (metric={result.metric}, strategy={args.strategy})")
    print(f"Best: {result.best_pipeline.describe()}")
    print(f"  CV {result.metric} = {result.best_score:.4f}")
    print(f"  Test  roc_auc={test_metrics['roc_auc']:.4f} accuracy={test_metrics['accuracy']:.4f} "
          f"f1={test_metrics['f1']:.4f}")
    save_model(result.best_pipeline, args.output, meta={"metric": result.metric, "test": test_metrics})
    print(f"  Model saved to {args.output}")
    if args.md:
        open(args.md, "w", encoding="utf-8").write(report.to_markdown(result, test_metrics, ds))
        print(f"  Report written to {args.md}")
    if args.json_out:
        open(args.json_out, "w", encoding="utf-8").write(report.to_json(result, test_metrics, ds))
        print(f"  JSON written to {args.json_out}")
    return 0


def cmd_leaderboard(args) -> int:
    ds, result, test_metrics, _ = _train(args)
    print(f"\nLeaderboard (CV {result.metric}, {result.n_evaluated} pipelines):\n")
    print(f"  {'rank':<5}{'scaler':<10}{'model':<16}{'score':<9}{'std':<8}")
    for e in result.leaderboard[:15]:
        print(f"  {e.rank:<5}{e.spec['scaler']:<10}{e.spec['model']:<16}{e.score:<9.4f}{e.std:<8.4f}")
    return 0


def cmd_predict(args) -> int:
    pipe, meta = load_model(args.model)
    if args.features:
        row = [float(x) for x in args.features.split(",")]
        proba = pipe.predict_proba([row])[0]
        print(f"P(donates) = {proba:.4f}  ->  {'DONATE' if proba >= 0.5 else 'no donation'}")
        return 0
    if args.data:
        ds = load_dataset(args.data)
        for i, p in enumerate(pipe.predict_proba(ds.X)):
            print(f"row {i}: P(donates)={p:.4f} -> {int(p >= 0.5)}")
        return 0
    print(f"Provide --features '{','.join(FEATURES)}' or --data <csv>", file=sys.stderr)
    return 1


def cmd_evaluate(args) -> int:
    pipe, meta = load_model(args.model)
    ds = load_dataset(args.data)
    _, Xte, _, yte = train_test_split(ds.X, ds.y, test_size=0.25, seed=args.seed)
    m = report.evaluate(pipe, Xte, yte)
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"):
        print(f"  {k:<10} {m[k]:.4f}")
    print(f"  confusion  {m['confusion_matrix']}")
    return 0


def cmd_data(args) -> int:
    path = download_real_dataset()
    print(f"Downloaded real UCI dataset to {path}")
    return 0


def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("Install server extras: pip install 'sanguine-automl[server]'", file=sys.stderr)
        return 1
    uvicorn.run("sanguine.service.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def _add_train_args(p):
    p.add_argument("--data", help="path to a CSV (default: bundled/synthetic dataset)")
    p.add_argument("--metric", default="roc_auc",
                   choices=["roc_auc", "accuracy", "f1", "precision", "recall", "log_loss"])
    p.add_argument("--strategy", default="evolution", choices=["evolution", "random"])
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--generations", type=int, default=6)
    p.add_argument("--population", type=int, default=12)
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sanguine", description="From-scratch AutoML for blood-donation prediction.")
    p.add_argument("--version", action="version", version=f"Sanguine {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train", help="run AutoML and save the best model")
    _add_train_args(tr)
    tr.add_argument("-o", "--output", default="model.pkl")
    tr.add_argument("--md", help="write a Markdown report")
    tr.add_argument("--json", dest="json_out", help="write a JSON report")
    tr.set_defaults(func=cmd_train)

    lb = sub.add_parser("leaderboard", help="search and print the leaderboard")
    _add_train_args(lb)
    lb.set_defaults(func=cmd_leaderboard)

    pr = sub.add_parser("predict", help="predict with a saved model")
    pr.add_argument("--model", default="model.pkl")
    pr.add_argument("--features", help="comma-separated: recency,frequency,monetary,time")
    pr.add_argument("--data", help="CSV to predict for every row")
    pr.set_defaults(func=cmd_predict)

    ev = sub.add_parser("evaluate", help="evaluate a saved model on the test split")
    ev.add_argument("--model", default="model.pkl")
    ev.add_argument("--data")
    ev.add_argument("--seed", type=int, default=42)
    ev.set_defaults(func=cmd_evaluate)

    dt = sub.add_parser("data", help="download the real UCI dataset")
    dt.set_defaults(func=cmd_data)

    sv = sub.add_parser("serve", help="run the prediction API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--reload", action="store_true")
    sv.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
