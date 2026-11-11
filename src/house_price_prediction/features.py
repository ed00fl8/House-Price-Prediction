"""Input validation and deterministic feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW_FEATURE_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
    "date",
]

NUMERIC_RAW_COLUMNS = [
    column for column in RAW_FEATURE_COLUMNS if column not in {"zipcode", "date"}
]

FEATURE_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
    "sale_year",
    "sale_month",
    "house_age",
    "renovated",
    "years_since_renovation",
    "log_sqft_living",
    "grade_living",
]

TARGET_COLUMN = "price"


def _normalise_zipcode(values: pd.Series) -> pd.Series:
    """Return ZIP codes as stable strings for CatBoost's categorical feature."""
    numeric = pd.to_numeric(values, errors="coerce")
    non_missing = values.notna()
    invalid = non_missing & numeric.isna()
    if invalid.any():
        examples = values.loc[invalid].astype(str).head(5).tolist()
        raise ValueError(f"'zipcode' contains invalid values: {examples}")

    non_integer = numeric.notna() & ~np.isclose(numeric.fillna(0), np.round(numeric.fillna(0)))
    if non_integer.any():
        raise ValueError("'zipcode' must contain integer ZIP codes.")

    result = numeric.round().astype("Int64").astype("string")
    return result.fillna("__MISSING__")


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Convert raw King County records into the exact model input schema.

    The function is shared by training and inference so both paths apply the
    same transformations. Extra columns (for example ``id`` or ``price``) are
    ignored and can never enter the feature matrix.
    """
    missing = sorted(set(RAW_FEATURE_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    frame = data.loc[:, RAW_FEATURE_COLUMNS].copy()

    for column in NUMERIC_RAW_COLUMNS:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Column '{column}' must contain numeric values.") from exc
        values = frame[column].to_numpy(dtype="float64", na_value=np.nan)
        if np.isinf(values).any():
            raise ValueError(f"Column '{column}' cannot contain infinite values.")

    dates = pd.to_datetime(frame["date"], errors="coerce")
    invalid_dates = dates.isna()
    if invalid_dates.any():
        examples = frame.loc[invalid_dates, "date"].astype(str).head(5).tolist()
        raise ValueError(f"Column 'date' contains missing or invalid dates: {examples}")

    frame["zipcode"] = _normalise_zipcode(frame["zipcode"])
    frame["sale_year"] = dates.dt.year.astype("int16")
    frame["sale_month"] = dates.dt.month.astype("int8")

    if (frame["sqft_living"].dropna() < 0).any():
        raise ValueError("'sqft_living' cannot contain negative values.")

    frame["house_age"] = frame["sale_year"] - frame["yr_built"]

    renovation_input = data.get("yr_renovated", pd.Series(0, index=data.index))
    renovation_year = pd.to_numeric(renovation_input, errors="coerce")
    invalid_renovation = renovation_input.notna() & renovation_year.isna()
    if invalid_renovation.any():
        raise ValueError("Column 'yr_renovated' must contain numeric years.")
    renovation_year = renovation_year.fillna(0)
    if np.isinf(renovation_year.to_numpy(dtype="float64")).any():
        raise ValueError("Column 'yr_renovated' cannot contain infinite values.")
    frame["renovated"] = (renovation_year > 0).astype("int8")
    frame["years_since_renovation"] = np.where(
        renovation_year == 0,
        frame["house_age"],
        frame["sale_year"] - renovation_year,
    )

    frame["log_sqft_living"] = np.log1p(frame["sqft_living"].clip(lower=0))
    frame["grade_living"] = frame["grade"] * frame["sqft_living"]

    # Keep the same feature order for training and prediction.
    return frame.loc[:, FEATURE_COLUMNS].copy()


def validate_training_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate the labelled source file and remove exact record duplicates.

    The source's unique ``id`` is excluded from duplicate detection, since it
    is an identifier and not a predictive signal.
    """
    required = set(RAW_FEATURE_COLUMNS) | {"yr_renovated", TARGET_COLUMN}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Training CSV is missing required columns: {missing}")

    frame = data.loc[:, RAW_FEATURE_COLUMNS + ["yr_renovated", TARGET_COLUMN]].copy()
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    valid_target = np.isfinite(
        frame[TARGET_COLUMN].to_numpy(dtype="float64", na_value=np.nan)
    )
    frame = frame.loc[valid_target].dropna(subset=[TARGET_COLUMN])
    if frame.empty:
        raise ValueError("No rows with a valid target price were found.")
    if (frame[TARGET_COLUMN] <= 0).any():
        raise ValueError("All target prices must be greater than zero.")

    duplicate_columns = [column for column in frame.columns if column != "id"]
    frame = frame.drop_duplicates(subset=duplicate_columns).reset_index(drop=True)
    return frame
