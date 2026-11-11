# King County House Price Prediction

A machine learning project that predicts residential property prices in King County, Washington, USA, using Python and CatBoost.

Automatically downloads the original Kaggle dataset, trains the model, evaluates its performance, and generates predictions.

## Requirements

- Python 3.13 (managed by uv)
- Internet connection
- Windows, macOS, or Linux

## Installation

1. Download and extract the repository.
2. Open a terminal in the project directory.
3. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS, Linux, and WSL:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

Restart the terminal if `uv` is not recognized.

4. Install Python and project dependencies:

```bash
uv python install 3.13
uv sync --locked --extra dev
```

## Train the model

```bash
uv run --locked --extra dev python -m house_price_prediction.train
```

The project automatically downloads and verifies the original King County dataset from Kaggle.

The model, evaluation metrics, predictions, and reports are generated locally.

## Make predictions

Run after training:

```bash
uv run --locked --extra dev python -m house_price_prediction.predict --input samples/example_input.csv --output reports/sample_predictions.csv
```

Predictions are saved in USD. The example contains synthetic records.

## SHAP explainability (optional)

Install the optional dependency:

```bash
uv sync --locked --extra dev --extra explainability
```

Generate explanations:

```bash
uv run --locked --extra dev --extra explainability python -m house_price_prediction.train --shap
```

The SHAP summary is saved in `reports/figures/`.

## Tests

```bash
uv run --locked --extra dev python -m pytest -v
```

## Dataset

Source: [King County House Sales — Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction).

The dataset is downloaded automatically and checked against its pinned SHA-256 checksum.

## Limitations

- The model is trained for King County and is not validated for other housing markets.
- Predictions are estimates, not professional property appraisals.
- Historical relationships may not represent current market conditions.
- Feature importance and SHAP values describe model behavior, not causation.

## Technologies

Python · CatBoost · Pandas · NumPy · Scikit-learn · SHAP · Pytest