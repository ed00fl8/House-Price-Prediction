import pandas as pd
import pytest

from house_price_prediction.features import FEATURE_COLUMNS, engineer_features


def raw_row(**overrides):
    row = {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 1500,
        "sqft_lot": 5000,
        "floors": 1.0,
        "waterfront": 0,
        "view": 0,
        "condition": 3,
        "grade": 7,
        "sqft_above": 1200,
        "sqft_basement": 300,
        "yr_built": 1990,
        "yr_renovated": 0,
        "zipcode": 98001,
        "lat": 47.3,
        "long": -122.2,
        "sqft_living15": 1400,
        "sqft_lot15": 5000,
        "date": "20141013T000000",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_engineered_schema_matches_model_contract():
    result = engineer_features(raw_row())
    assert result.columns.tolist() == FEATURE_COLUMNS
    assert result.loc[0, "zipcode"] == "98001"
    assert result.loc[0, "sale_year"] == 2014
    assert result.loc[0, "sale_month"] == 10
    assert result.loc[0, "house_age"] == 24
    assert result.loc[0, "renovated"] == 0
    assert result.loc[0, "years_since_renovation"] == 24
    assert result.loc[0, "grade_living"] == 10500


def test_renovation_features():
    result = engineer_features(raw_row(yr_built=1980, yr_renovated=2010))
    assert result.loc[0, "renovated"] == 1
    assert result.loc[0, "years_since_renovation"] == 4


def test_missing_required_column_fails_clearly():
    frame = raw_row().drop(columns=["grade"])
    with pytest.raises(ValueError, match="grade"):
        engineer_features(frame)


def test_invalid_date_fails_clearly():
    with pytest.raises(ValueError, match="invalid dates"):
        engineer_features(raw_row(date="not-a-date"))
