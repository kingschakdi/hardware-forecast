"""
run_feature_comparison.py
-------------------------
Compares two feature sets for the ML models:
    "simple" : lag features only
    "rich"   : lags + rolling mean/std + month

Answers whether richer engineered features improve accuracy on these short
series, or whether simple lags are sufficient. Only models that use the
feature switch are included (LSTM uses raw sequences, so it is excluded).
"""

import numpy as np
import pandas as pd

from data_loader import load_gpu_prices
from metrics import rmse, mae, mase
from ml_models import (random_forest_forecast, gradient_boosting_forecast,
                       mlp_forecast)

HORIZON = 3


def evaluate(gpu_df, forecast_fn, feature_set):
    forecasts, holdouts = forecast_fn(gpu_df, feature_set=feature_set, horizon=HORIZON)
    R, M, S = [], [], []
    for model, fc in forecasts.items():
        train, actual = holdouts[model]
        R.append(rmse(actual, fc))
        M.append(mae(actual, fc))
        S.append(mase(actual, fc, train.values))
    return np.nanmean(R), np.nanmean(M), np.nanmean(S)


def main():
    gpu = load_gpu_prices()
    print("Comparing simple vs rich features (RMSE / MAE / MASE)...\n")

    methods = [
        ("Random Forest", random_forest_forecast),
        ("Gradient Boosting", gradient_boosting_forecast),
        ("MLP", mlp_forecast),
    ]

    rows = []
    for name, fn in methods:
        print(f"  {name}...")
        for fs in ("simple", "rich"):
            r, m, s = evaluate(gpu, fn, fs)
            rows.append((name, fs, r, m, s))

    print("\n" + "=" * 68)
    print(f"{'Method':20s} {'Features':10s} {'RMSE':>8s} {'MAE':>8s} {'MASE':>8s}")
    print("-" * 68)
    for name, fs, r, m, s in rows:
        print(f"{name:20s} {fs:10s} {r:8.2f} {m:8.2f} {s:8.3f}")
    print("=" * 68)
    print("Compare each method's simple vs rich rows: does richer help?")

    pd.DataFrame(rows, columns=["method", "feature_set", "rmse", "mae", "mase"]
                 ).to_csv("outputs/feature_comparison.csv", index=False)
    print("\nSaved to outputs/feature_comparison.csv")


if __name__ == "__main__":
    main()