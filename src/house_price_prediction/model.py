"""Model construction and artifact loading."""

from __future__ import annotations

from pathlib import Path

from catboost import CatBoostRegressor

from .features import FEATURE_COLUMNS

MODEL_FILENAME = "catboost_house_price_model.cbm"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / MODEL_FILENAME
CATEGORICAL_FEATURES = ["zipcode"]

DEFAULT_MODEL_PARAMS = {
    "iterations": 1583,
    "depth": 7,
    "learning_rate": 0.02976768278,
    "l2_leaf_reg": 2.862750053,
    "random_strength": 0.600264132,
}


def create_model(**overrides: object) -> CatBoostRegressor:
    """Create the CatBoost regressor used by the training pipeline."""
    params = {
        **DEFAULT_MODEL_PARAMS,
        **overrides,
        "loss_function": "RMSE",
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": 1,
    }
    return CatBoostRegressor(**params)


def load_model(path: str | Path | None = None) -> CatBoostRegressor:
    """Load a saved model and verify its input feature schema."""
    if path is not None:
        model_path = Path(path).expanduser()
    else:
        candidates = [
            PROJECT_ROOT / "models" / MODEL_FILENAME,
            Path.cwd() / "models" / MODEL_FILENAME,
        ]
        model_path = next(
            (candidate for candidate in candidates if candidate.is_file()),
            candidates[0],
        )
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Run `python -m "
            "house_price_prediction.train` first, or pass --model with a model path."
        )

    model = CatBoostRegressor()
    model.load_model(str(model_path))
    model_features = list(model.feature_names_)
    if model_features != FEATURE_COLUMNS:
        raise ValueError(
            "The model does not match this project's feature schema. "
            f"Expected {FEATURE_COLUMNS}, received {model_features}."
        )
    return model
