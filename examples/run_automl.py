"""Run the AutoML search offline and print the leaderboard + report.

    python examples/run_automl.py
"""

from sanguine import report
from sanguine.data import generate_synthetic, train_test_split
from sanguine.search import AutoML


def main() -> None:
    ds = generate_synthetic()
    print("Dataset:", ds.summary())

    Xtr, Xte, ytr, yte = train_test_split(ds.X, ds.y, test_size=0.25)
    automl = AutoML(metric="roc_auc", cv=5, strategy="evolution", generations=6, population=12)
    result = automl.fit(Xtr, ytr)

    print(f"\nEvaluated {result.n_evaluated} pipelines. Leaderboard (top 8):\n")
    print(f"  {'rank':<5}{'scaler':<10}{'model':<16}{'CV roc_auc':<12}std")
    for e in result.leaderboard[:8]:
        print(f"  {e.rank:<5}{e.spec['scaler']:<10}{e.spec['model']:<16}{e.score:<12.4f}{e.std:.4f}")

    test = report.evaluate(result.best_pipeline, Xte, yte)
    print(f"\nWinner: {result.best_pipeline.describe()}")
    print(f"Held-out test: roc_auc={test['roc_auc']:.4f}  accuracy={test['accuracy']:.4f}  f1={test['f1']:.4f}")


if __name__ == "__main__":
    main()
