"""
run_comparison.py
-----------------
Runs every forecasting method on the GPU price data and reports RMSE, MAE
and MASE side by side in one table. This is the core results output for the
dissertation's comparison of traditional vs machine-learning methods.

Baselines (models.py) forecast each series from its own history.
ML models (ml_models.py) are trained pooled across all series.
Both are evaluated on the same 3-month hold-out per series, then averaged.
"""

import numpy as np
import pandas as pd

from data_loader import load_gpu_prices, load_ram_prices, load_ssd_prices, load_cpu_prices
from metrics import rmse, mae, mase
from models import moving_average, exponential_smoothing, crostons_method
from ml_models import (random_forest_forecast, gradient_boosting_forecast,
                       mlp_forecast, lstm_forecast)

HORIZON = 3


def evaluate_baseline(gpu_df, forecast_fn, **kwargs):
    """
    Evaluate a per-series baseline. For each GPU with enough history, hold
    out the last HORIZON points, forecast them, and collect the three metrics.
    Returns lists of per-series (rmse, mae, mase).
    """
    rmses, maes, mases = [], [], []
    for model in gpu_df["model"].unique():
        s = gpu_df[gpu_df["model"] == model].sort_values("date")["price"]
        if len(s) < HORIZON + 4:
            continue
        train = s.iloc[:-HORIZON].values
        actual = s.iloc[-HORIZON:].values
        forecast = forecast_fn(train, horizon=HORIZON, **kwargs)
        rmses.append(rmse(actual, forecast))
        maes.append(mae(actual, forecast))
        mases.append(mase(actual, forecast, train))
    return rmses, maes, mases


def evaluate_ml(gpu_df, forecast_fn):
    """
    Evaluate a pooled ML model. It returns {model: forecast} plus holdouts;
    we compute the three metrics per series against the held-out actuals.
    """
    forecasts, holdouts = forecast_fn(gpu_df, horizon=HORIZON)
    rmses, maes, mases = [], [], []
    for model, fc in forecasts.items():
        train, actual = holdouts[model]
        rmses.append(rmse(actual, fc))
        maes.append(mae(actual, fc))
        mases.append(mase(actual, fc, train.values))
    return rmses, maes, mases


def run_for_component(df, component_name, csv_suffix):
    """Run all methods on one component's data, print a table, save a CSV."""
    print(f"\n{'#'*70}\n# {component_name.upper()} — all methods (3-month hold-out)\n{'#'*70}")

    results = []

    baseline_methods = [
        ("Moving Average", moving_average, {"window": 3}),
        ("Exponential Smoothing", exponential_smoothing, {"alpha": 0.3}),
        ("Croston's", crostons_method, {}),
        ("Croston's (SBA)", crostons_method, {"sba": True}),
    ]
    for name, fn, kw in baseline_methods:
        r, m, ms = evaluate_baseline(df, fn, **kw)
        results.append((name, np.nanmean(r), np.nanmean(m), np.nanmean(ms), len(r)))

    ml_methods = [
        ("Random Forest", random_forest_forecast),
        ("Gradient Boosting", gradient_boosting_forecast),
        ("MLP", mlp_forecast),
        ("LSTM", lstm_forecast),
    ]
    for name, fn in ml_methods:
        print(f"  training {name}...")
        r, m, ms = evaluate_ml(df, fn)
        results.append((name, np.nanmean(r), np.nanmean(m), np.nanmean(ms), len(r)))

    results.sort(key=lambda x: x[1])
    print("\n" + "=" * 70)
    print(f"{component_name} results")
    print(f"{'Method':22s} {'RMSE':>8s} {'MAE':>8s} {'MASE':>8s} {'n':>5s}")
    print("-" * 70)
    for name, r, m, ms, n in results:
        print(f"{name:22s} {r:8.2f} {m:8.2f} {ms:8.3f} {n:5d}")
    print("=" * 70)
    print("RMSE/MAE in the component's price units. MASE scale-free (<1 beats naive).")

    df_out = pd.DataFrame(results, columns=["method", "rmse", "mae", "mase", "n_series"])
    path = f"outputs/model_comparison_{csv_suffix}.csv"
    df_out.to_csv(path, index=False)
    print(f"Saved to {path}")


def main():
    # GPU
    gpu = load_gpu_prices()
    run_for_component(gpu, "GPU", "gpu")
    # keep the dashboard's expected filename working
    import shutil
    shutil.copy("outputs/model_comparison_gpu.csv", "outputs/model_comparison.csv")

    # RAM (limited web data: 4 series, ~9-10 monthly points each)
    try:
        ram = load_ram_prices()
        run_for_component(ram, "RAM", "ram")
    except Exception as e:
        print(f"\n[RAM skipped: {e}]")

    # SSD (single index, estimated USD/GB, 497 monthly points)
    try:
        ssd = load_ssd_prices()
        run_for_component(ssd, "SSD", "ssd")
    except Exception as e:
        print(f"\n[SSD skipped: {e}]")

    # CPU (simulated: 6 series, 24 monthly points each)
    try:
        cpu = load_cpu_prices()
        run_for_component(cpu, "CPU", "cpu")
    except Exception as e:
        print(f"\n[CPU skipped: {e}]")


if __name__ == "__main__":
    main()