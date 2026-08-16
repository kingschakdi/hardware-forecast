"""
metrics.py
----------
Forecast-accuracy metrics.
"""

import numpy as np


# ---------------------------------------------------------------------------
# RMSE 
# ---------------------------------------------------------------------------
def rmse(actual, forecast) -> float:
    """
    Root Mean Squared Error.

    Penalises large deviations more heavily than small ones; the standard
    accuracy metric used in the spare-parts forecasting comparator studies.
    """
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if a.shape != f.shape:
        raise ValueError(f"shape mismatch: actual {a.shape}, forecast {f.shape}")
    return float(np.sqrt(np.mean((a - f) ** 2)))


# ---------------------------------------------------------------------------
# MAE 
# ---------------------------------------------------------------------------
def mae(actual, forecast) -> float:
    """
    Mean Absolute Error: the average absolute difference between actual and
    forecast values. Interpretable directly in price units (USD).
    """
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if a.shape != f.shape:
        raise ValueError(f"shape mismatch: actual {a.shape}, forecast {f.shape}")
    return float(np.mean(np.abs(a - f)))


# ---------------------------------------------------------------------------
# MASE 
# ---------------------------------------------------------------------------
def mase(actual, forecast, training_series) -> float:
    """
    Mean Absolute Scaled Error (Hyndman & Koehler, 2006).

    Scales forecast MAE by the MAE of a naive one-step forecast on the
    training data. MASE < 1 beats the naive forecast; > 1 is worse.
    Scale-free, so it allows comparison across GPUs at different price levels.
    """
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    train = np.asarray(training_series, dtype=float)

    if a.shape != f.shape:
        raise ValueError(f"shape mismatch: actual {a.shape}, forecast {f.shape}")
    if len(train) < 2:
        raise ValueError("training_series needs at least 2 points to scale MASE")

    scale = np.mean(np.abs(np.diff(train)))
    if scale == 0:
        return float("nan")

    return float(np.mean(np.abs(a - f)) / scale)


if __name__ == "__main__":
    actual = [100, 105, 110]
    forecast = [102, 102, 102]
    training = [90, 95, 92, 98, 100]
    print(f"RMSE: {rmse(actual, forecast):.3f}")
    print(f"MAE:  {mae(actual, forecast):.3f}")
    print(f"MASE: {mase(actual, forecast, training):.3f}")