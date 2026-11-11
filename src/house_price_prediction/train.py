"""Train, evaluate, and save the King County house-price model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split

from .dataset import DatasetSource, discover_dataset
from .features import FEATURE_COLUMNS, TARGET_COLUMN, engineer_features, validate_training_data
from .metrics import regression_metrics
from .model import CATEGORICAL_FEATURES, DEFAULT_MODEL_PARAMS, create_model


def _save_shap(model: CatBoostRegressor, X_test: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
    except ImportError as exc:
        raise RuntimeError(
            'SHAP is optional. Install it with: python -m pip install -e ".[explainability]".'
        ) from exc

    sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    shap.summary_plot(shap_values, sample, max_display=15, show=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = plt.gcf()
    figure.savefig(output_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def train(
    data_path: str | Path | None = None,
    output_dir: str | Path = ".",
    iterations: int | None = None,
    include_shap: bool = False,
) -> dict[str, object]:
    """Train on 80%, evaluate once on a 20% holdout, and save the artifacts."""
    source: DatasetSource = discover_dataset(data_path)
    source_data = source.read()
    rows_read = len(source_data)
    raw = validate_training_data(source_data)
    if len(raw) < 10:
        raise ValueError(
            f"At least 10 valid rows are required for training; found {len(raw)}."
        )

    features = engineer_features(raw)
    target = raw[TARGET_COLUMN].astype(float)
    target_log = np.log1p(target)

    X_train, X_test, y_train_log, y_test_log, y_train, y_test = train_test_split(
        features,
        target_log,
        target,
        test_size=0.20,
        random_state=42,
    )

    params: dict[str, object] = dict(DEFAULT_MODEL_PARAMS)
    if iterations is not None:
        if iterations < 1:
            raise ValueError("iterations must be a positive integer.")
        params["iterations"] = iterations

    model = create_model(**params)
    model.fit(X_train, y_train_log, cat_features=CATEGORICAL_FEATURES)

    predicted_log = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
    predicted = np.maximum(np.expm1(predicted_log), 0.0)
    actual = np.asarray(y_test, dtype=float)
    model_metrics = regression_metrics(actual, predicted)

    baseline = DummyRegressor(strategy="median")
    baseline.fit(np.zeros((len(y_train_log), 1)), y_train_log)
    baseline_prediction = np.expm1(
        baseline.predict(np.zeros((len(y_test_log), 1)))
    )
    baseline_metrics = regression_metrics(actual, baseline_prediction)
    baseline_mae = baseline_metrics["mae_usd"]
    mae_reduction = (
        None
        if baseline_mae == 0
        else 100 * (baseline_mae - model_metrics["mae_usd"]) / baseline_mae
    )

    output_dir = Path(output_dir).expanduser().resolve()
    model_dir = output_dir / "models"
    report_dir = output_dir / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "catboost_house_price_model.cbm"
    model.save_model(str(model_path))

    metrics = {
        "dataset": "King County House Sales",
        "rows_read": int(rows_read),
        "rows_after_cleaning": int(len(raw)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "evaluation": "single untouched random holdout",
        "test_size": 0.20,
        "random_state": 42,
        "target_transform": "log1p during training; metrics reported in USD",
        "baseline_median": baseline_metrics,
        "catboost": model_metrics,
        "mae_reduction_vs_baseline_pct": mae_reduction,
    }
    (report_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    importance = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(report_dir / "feature_importance.csv", index=False)

    pd.DataFrame(
        {
            "actual_price_usd": actual,
            "predicted_price_usd": predicted,
            "absolute_error_usd": np.abs(actual - predicted),
        }
    ).to_csv(report_dir / "holdout_predictions.csv", index=False)

    if include_shap:
        _save_shap(model, X_test, report_dir / "figures")

    model_card = {
        "model": "CatBoostRegressor",
        "artifact": "models/catboost_house_price_model.cbm",
        "target": TARGET_COLUMN,
        "target_unit": "USD",
        "target_transform": "log1p",
        "prediction_transform": "expm1",
        "categorical_features": CATEGORICAL_FEATURES,
        "features": FEATURE_COLUMNS,
        "parameters": model.get_all_params(),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    (model_dir / "model_card.json").write_text(
        json.dumps(model_card, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    print(f"Dataset: {source.label}")
    print(f"Model: {model_path}")
    print(f"Reports: {report_dir}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the King County house-price model."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help=(
            "Optional local CSV for controlled experiments. "
            "By default, the pinned Kaggle dataset is downloaded automatically."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Folder for the model and reports (default: current directory).",
    )
    parser.add_argument("--iterations", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Create an optional SHAP summary plot.",
    )
    args = parser.parse_args()
    try:
        metrics = train(
            data_path=args.data,
            output_dir=args.output_dir,
            iterations=args.iterations,
            include_shap=args.shap,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
