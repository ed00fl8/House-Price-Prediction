import json

import numpy as np


def test_training_pipeline_creates_model_and_reports(trained_model):
    root = trained_model["root"]
    metrics = trained_model["metrics"]

    assert metrics["rows_after_cleaning"] == 72
    assert metrics["train_rows"] == 57
    assert metrics["test_rows"] == 15
    assert trained_model["model_path"].is_file()
    assert (root / "models" / "model_card.json").is_file()
    assert (root / "reports" / "metrics.json").is_file()
    assert (root / "reports" / "holdout_predictions.csv").is_file()
    assert (root / "reports" / "feature_importance.csv").is_file()

    saved_metrics = json.loads((root / "reports" / "metrics.json").read_text())
    assert np.isfinite(saved_metrics["catboost"]["mae_usd"])
