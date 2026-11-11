from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from house_price_prediction.train import train


def synthetic_training_data(n: int = 72) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    living = rng.integers(700, 3600, n)
    grade = rng.integers(5, 11, n)
    zipcode = rng.choice([98001, 98004, 98052, 98118], n)
    built = rng.integers(1940, 2015, n)
    price = np.maximum(
        90_000 + living * 155 + grade * 9_000 + (zipcode == 98004) * 100_000
        + rng.normal(0, 35_000, n),
        65_000,
    )
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "date": pd.date_range("2014-05-01", periods=n, freq="D").strftime(
                "%Y%m%dT000000"
            ),
            "price": price,
            "bedrooms": rng.integers(1, 6, n),
            "bathrooms": rng.uniform(1, 4, n).round(2),
            "sqft_living": living,
            "sqft_lot": rng.integers(1_000, 15_000, n),
            "floors": rng.choice([1, 1.5, 2, 2.5], n),
            "waterfront": rng.integers(0, 2, n),
            "view": rng.integers(0, 5, n),
            "condition": rng.integers(1, 6, n),
            "grade": grade,
            "sqft_above": living - rng.integers(0, 300, n),
            "sqft_basement": rng.integers(0, 300, n),
            "yr_built": built,
            "yr_renovated": rng.choice([0, 0, 0, 2000, 2010], n),
            "zipcode": zipcode,
            "lat": rng.uniform(47.2, 47.8, n),
            "long": rng.uniform(-122.5, -121.8, n),
            "sqft_living15": living + rng.integers(-150, 250, n),
            "sqft_lot15": rng.integers(1_000, 15_000, n),
        }
    )


@pytest.fixture(scope="session")
def synthetic_dataset() -> pd.DataFrame:
    return synthetic_training_data()


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory, synthetic_dataset):
    root = tmp_path_factory.mktemp("trained-project")
    data_path = root / "unusual-name.csv"
    synthetic_dataset.to_csv(data_path, index=False)
    metrics = train(data_path=data_path, output_dir=root, iterations=12)
    return {
        "root": root,
        "model_path": root / "models" / "catboost_house_price_model.cbm",
        "metrics": metrics,
    }
