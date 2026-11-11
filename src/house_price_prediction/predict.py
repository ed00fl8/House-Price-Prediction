"""CSV batch inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .features import engineer_features
from .model import load_model


def predict_dataframe(
    data: pd.DataFrame,
    model_path: str | Path | None = None,
) -> np.ndarray:
    """Predict sale prices in USD from raw King County property records."""
    features = engineer_features(data)
    model = load_model(model_path)
    predicted = np.expm1(np.asarray(model.predict(features), dtype=float).reshape(-1))
    if not np.isfinite(predicted).all():
        raise RuntimeError("The model returned a non-finite prediction.")
    return np.maximum(predicted, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict house prices from a CSV file.")
    parser.add_argument("--input", required=True, type=Path, help="CSV with raw property features.")
    parser.add_argument(
        "--output", required=True, type=Path, help="Destination CSV for predictions."
    )
    parser.add_argument(
        "--model", type=Path, default=None, help="Optional path to a trained .cbm model."
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input CSV not found: {args.input}")

    try:
        raw = pd.read_csv(args.input, low_memory=False)
        predictions = predict_dataframe(raw, args.model)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")

    result = pd.DataFrame(index=raw.index)
    if "id" in raw.columns:
        result["id"] = raw["id"]
    if "price" in raw.columns:
        result["actual_price_usd"] = raw["price"]
    result["predicted_price_usd"] = predictions

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
