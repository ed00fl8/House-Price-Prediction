import numpy as np
import pandas as pd

from house_price_prediction.model import load_model
from house_price_prediction.predict import predict_dataframe


def test_model_load_and_batch_inference(trained_model, monkeypatch):
    monkeypatch.chdir(trained_model["root"])
    model = load_model()
    assert len(model.feature_names_) == 24

    records = pd.DataFrame(
        {
            "bedrooms": [3, 4],
            "bathrooms": [2.25, 3.0],
            "sqft_living": [1800, 2800],
            "sqft_lot": [5000, 7200],
            "floors": [1, 2],
            "waterfront": [0, 0],
            "view": [0, 1],
            "condition": [3, 4],
            "grade": [7, 8],
            "sqft_above": [1200, 2300],
            "sqft_basement": [600, 500],
            "yr_built": [1990, 2005],
            "yr_renovated": [0, 2018],
            "zipcode": [98001, 98004],
            "lat": [47.32, 47.62],
            "long": [-122.27, -122.20],
            "sqft_living15": [1750, 2650],
            "sqft_lot15": [5100, 7000],
            "date": ["20141013T000000", "20150220T000000"],
        }
    )
    predictions = predict_dataframe(records)
    assert predictions.shape == (2,)
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()
