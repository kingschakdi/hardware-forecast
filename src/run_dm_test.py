"""
run_dm_test.py
--------------
Diebold-Mariano test (Diebold & Mariano, 1995) for pairwise comparison of
forecast accuracy, per component.

Each method's forecast errors are pooled across all series into one error
vector; the best method (by MSE) is tested against each other method.

NOTE: RAM has very few forecast points (4 series x 3 = ~12 errors), so the
DM test has low statistical power for RAM and results are indicative only.
This is a documented limitation.
"""

import numpy as np
import pandas as pd
from scipy import stats

from data_loader import load_gpu_prices, load_ram_prices, load_ssd_prices, load_cpu_prices
from models import moving_average, exponential_smoothing, crostons_method
from ml_models import (random_forest_forecast, gradient_boosting_forecast,
                       mlp_forecast, lstm_forecast)

HORIZON = 3


def diebold_mariano(errors_a, errors_b, power=2):
    ea = np.asarray(errors_a, dtype=float)
    eb = np.asarray(errors_b, dtype=float)
    mask = ~(np.isnan(ea) | np.isnan(eb))
    ea, eb = ea[mask], eb[mask]
    d = np.abs(ea) ** power - np.abs(eb) ** power
    n = len(d)
    if n < 2:
        return np.nan, np.nan
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    if d_var == 0:
        return np.nan, np.nan
    dm_stat = d_mean / np.sqrt(d_var / n)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value


def collect_baseline_errors(df, forecast_fn, **kwargs):
    errs = []
    for model in df["model"].unique():
        s = df[df["model"] == model].sort_values("date")["price"]
        if len(s) < HORIZON + 4:
            continue
        train = s.iloc[:-HORIZON].values
        actual = s.iloc[-HORIZON:].values
        fc = forecast_fn(train, horizon=HORIZON, **kwargs)
        errs.extend(list(np.asarray(actual, float) - np.asarray(fc, float)))
    return np.array(errs)


def collect_ml_errors(df, forecast_fn):
    try:
        forecasts, holdouts = forecast_fn(df, horizon=HORIZON)
    except Exception:
        return np.array([])  # model couldn't run (e.g. LSTM on short RAM)
    errs = []
    for model, fc in forecasts.items():
        _train, actual = holdouts[model]
        errs.extend(list(np.asarray(actual, float) - np.asarray(fc, float)))
    return np.array(errs)


def run_dm_for_component(df, component_name, csv_suffix):
    print(f"\n{'#'*72}\n# {component_name.upper()} — Diebold-Mariano test\n{'#'*72}")

    errors = {}
    errors["Moving Average"] = collect_baseline_errors(df, moving_average, window=3)
    errors["Exponential Smoothing"] = collect_baseline_errors(df, exponential_smoothing, alpha=0.3)
    errors["Croston's"] = collect_baseline_errors(df, crostons_method)
    errors["Croston's (SBA)"] = collect_baseline_errors(df, crostons_method, sba=True)
    for name, fn in [("Random Forest", random_forest_forecast),
                     ("Gradient Boosting", gradient_boosting_forecast),
                     ("MLP", mlp_forecast),
                     ("LSTM", lstm_forecast)]:
        print(f"  training {name}...")
        errors[name] = collect_ml_errors(df, fn)

    # Drop methods that produced no errors (couldn't run on this component)
    errors = {k: v for k, v in errors.items() if len(v) > 0}

    mse = {k: np.nanmean(np.asarray(v) ** 2) for k, v in errors.items()}
    best = min(mse, key=mse.get)
    print(f"\nBest method by MSE: {best}")
    print(f"Testing whether {best} is significantly better than each other method:\n")

    rows = []
    print("=" * 72)
    print(f"{'Comparison':40s} {'DM stat':>9s} {'p-value':>9s} {'sig?':>6s}")
    print("-" * 72)
    for name, errs in errors.items():
        if name == best:
            continue
        dm, p = diebold_mariano(errors[best], errs, power=2)
        sig = "yes" if (not np.isnan(p) and p < 0.05) else "no"
        p_disp = f"{p:.4f}" if not np.isnan(p) else "   nan"
        dm_disp = f"{dm:9.3f}" if not np.isnan(dm) else "      nan"
        print(f"{best} vs {name:22s} {dm_disp} {p_disp:>9s} {sig:>6s}")
        rows.append((f"{best} vs {name}", dm, p, sig))
    print("=" * 72)
    print("Negative DM = the best method has lower loss. p<0.05 = significant.")

    pd.DataFrame(rows, columns=["comparison", "dm_stat", "p_value", "significant"]
                 ).to_csv(f"outputs/dm_test_{csv_suffix}.csv", index=False)
    print(f"Saved to outputs/dm_test_{csv_suffix}.csv")


def main():
    gpu = load_gpu_prices()
    run_dm_for_component(gpu, "GPU", "gpu")

    try:
        ram = load_ram_prices()
        run_dm_for_component(ram, "RAM", "ram")
    except Exception as e:
        print(f"\n[RAM DM test skipped: {e}]")

    try:
        ssd = load_ssd_prices()
        run_dm_for_component(ssd, "SSD", "ssd")
    except Exception as e:
        print(f"\n[SSD DM test skipped: {e}]")

    try:
        cpu = load_cpu_prices()
        run_dm_for_component(cpu, "CPU", "cpu")
    except Exception as e:
        print(f"\n[CPU DM test skipped: {e}]")


if __name__ == "__main__":
    main()