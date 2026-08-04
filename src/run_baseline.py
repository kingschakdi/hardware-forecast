"""
run_baseline.py
---------------
End-to-end demonstration of the COMPLETED baseline pipeline on real GPU data:

    load real data -> hold out last 3 months -> Moving Average forecast -> RMSE

This proves the core pipeline works end to end. The more advanced models
(Croston's, RF, MLP, LSTM) and metrics (MAE, MASE) are stubbed in their
modules and will be slotted into this same harness in later phases.

Run:  python src/run_baseline.py
"""

import pandas as pd

from data_loader import load_gpu_prices
from models import moving_average
from metrics import rmse


def evaluate_one_model(series: pd.Series, horizon: int = 3, window: int = 3):
    """
    Hold out the last `horizon` points, forecast them with Moving Average,
    and compute RMSE against the actual held-out values.
    """
    if len(series) < horizon + window:
        return None  # not enough data for a fair hold-out

    train = series.iloc[:-horizon]
    actual = series.iloc[-horizon:].values

    forecast = moving_average(train.values, horizon=horizon, window=window)
    error = rmse(actual, forecast)
    return {"train_n": len(train), "rmse": error,
            "forecast": forecast, "actual": actual}


def main():
    print("=" * 60)
    print("BASELINE FORECAST DEMO — Moving Average on real GPU data")
    print("=" * 60)

    gpu = load_gpu_prices()

    # Evaluate the baseline on the models that have enough history.
    results = []
    for model in gpu["model"].unique():
        series = gpu[gpu["model"] == model].sort_values("date")["price"]
        outcome = evaluate_one_model(series, horizon=3, window=3)
        if outcome:
            results.append((model, outcome["rmse"]))

    if not results:
        print("No models had enough history for a 3-step hold-out.")
        return

    results.sort(key=lambda x: x[1])
    print(f"\nMoving Average baseline evaluated on {len(results)} GPU models.")
    print("(3-month hold-out, RMSE in USD — lower is better)\n")
    print(f"  {'Best 5 models':32s} {'RMSE':>8s}")
    print("  " + "-" * 42)
    for model, err in results[:5]:
        print(f"  {model:32s} {err:8.2f}")

    avg = sum(e for _, e in results) / len(results)
    print(f"\nAverage baseline RMSE across all evaluated models: {avg:.2f} USD")
    print("\nThis is the baseline the advanced models will be compared against")
    print("in later phases (Croston's, Random Forest, MLP, LSTM).")
    print("\nDone.")


if __name__ == "__main__":
    main()
