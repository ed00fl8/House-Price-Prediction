import pytest

from house_price_prediction.metrics import regression_metrics


def test_metrics_are_on_original_target_scale():
    result = regression_metrics([100_000, 200_000], [110_000, 180_000])
    assert result["mae_usd"] == pytest.approx(15_000)
    assert result["rmse_usd"] == pytest.approx((250_000_000) ** 0.5)
    assert 0 <= result["mape_pct"]


def test_mape_requires_positive_actual_prices():
    with pytest.raises(ValueError, match="greater than zero"):
        regression_metrics([0, 100], [5, 90])
