"""
run_cv.py
---------
Rolling-origin cross-validation for all forecasting methods, per component.

Each method is tested on several successive time windows (folds), training
only on data before each window. This gives a more robust estimate of
forecast accuracy and its variability than a single hold-out.

GPU series are short (~24 monthly points) so a few folds are feasible.
RAM series are much shorter (~9-10 points) so RAM uses fewer folds and its 
cross-validation is correspondingly limited — a documented limitation.
"""

import numpy as np
import pandas as pd

from data_loader import load_gpu_prices, load_ram_prices, load_ssd_prices, load_cpu_prices
from metrics import rmse, mae, mase
from models import moving_average, exponential_smoothing, crostons_method
from ml_models import build_training_table, _forecast_series
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

HORIZON = 3
N_LAGS = 3


def series_folds(s, horizon, n_folds, min_train):
    """Yield (train, actual) rolling-origin folds for one series."""
    for k in range(n_folds):
        end = len(s) - k * horizon
        train = s.iloc[:end - horizon]
        actual = s.iloc[end - horizon:end].values
        if len(train) >= min_train and len(actual) == horizon:
            yield train, actual


def cv_baseline(df, forecast_fn, n_folds, **kwargs):
    R, M, S = [], [], []
    for model in df["model"].unique():
        s = df[df["model"] == model].sort_values("date")["price"]
        for train, actual in series_folds(s, HORIZON, n_folds, min_train=N_LAGS + 2):
            fc = forecast_fn(train.values, horizon=HORIZON, **kwargs)
            R.append(rmse(actual, fc))
            M.append(mae(actual, fc))
            S.append(mase(actual, fc, train.values))
    return R, M, S


def _fit_tree(df, fold_k, Model):
    trimmed_rows = []
    for model in df["model"].unique():
        s = df[df["model"] == model].sort_values("date")
        cut = len(s) - fold_k * HORIZON
        trimmed_rows.append(s.iloc[:cut])
    trimmed = pd.concat(trimmed_rows, ignore_index=True)
    X, y, holdouts = build_training_table(trimmed, N_LAGS, "simple", HORIZON)
    reg = Model(n_estimators=200, random_state=42)
    reg.fit(X, y)
    return reg, holdouts


def cv_tree(df, Model, n_folds):
    R, M, S = [], [], []
    for k in range(n_folds):
        try:
            reg, holdouts = _fit_tree(df, k, Model)
        except Exception:
            continue  # not enough data for this fold (short series)
        for name, (train, actual) in holdouts.items():
            fc = _forecast_series(reg, train, N_LAGS, "simple", HORIZON)
            R.append(rmse(actual, fc))
            M.append(mae(actual, fc))
            S.append(mase(actual, fc, train.values))
    return R, M, S


def summarize(name, R, M, S):
    if len(R) == 0:
        return (name, float("nan"), float("nan"), float("nan"), float("nan"), 0)
    return (name, np.nanmean(R), np.nanstd(R), np.nanmean(M), np.nanmean(S), len(R))


def run_cv_for_component(df, component_name, n_folds, csv_suffix):
    print(f"\n{'#'*78}\n# {component_name.upper()} — rolling-origin CV "
          f"({n_folds} fold(s), horizon {HORIZON})\n{'#'*78}")
    results = []

    for name, fn, kw in [
        ("Moving Average", moving_average, {"window": 3}),
        ("Exponential Smoothing", exponential_smoothing, {"alpha": 0.3}),
        ("Croston's", crostons_method, {}),
        ("Croston's (SBA)", crostons_method, {"sba": True}),
    ]:
        R, M, S = cv_baseline(df, fn, n_folds, **kw)
        results.append(summarize(name, R, M, S))

    print("  cross-validating Random Forest...")
    results.append(summarize("Random Forest", *cv_tree(df, RandomForestRegressor, n_folds)))
    print("  cross-validating Gradient Boosting...")
    results.append(summarize("Gradient Boosting", *cv_tree(df, GradientBoostingRegressor, n_folds)))

    results.sort(key=lambda x: (np.isnan(x[1]), x[1]))
    print("\n" + "=" * 78)
    print(f"{component_name} CV results")
    print(f"{'Method':22s} {'RMSE':>8s} {'±std':>7s} {'MAE':>8s} {'MASE':>8s} {'evals':>6s}")
    print("-" * 78)
    for name, rmean, rstd, mmean, smean, n in results:
        print(f"{name:22s} {rmean:8.2f} {rstd:7.2f} {mmean:8.2f} {smean:8.3f} {n:6d}")
    print("=" * 78)
    print("Averaged across all series and folds. ±std = spread of RMSE across evals.")

    pd.DataFrame(results, columns=["method", "rmse_mean", "rmse_std",
                                   "mae_mean", "mase_mean", "n_evals"]
                 ).to_csv(f"outputs/cv_comparison_{csv_suffix}.csv", index=False)
    print(f"Saved to outputs/cv_comparison_{csv_suffix}.csv")


def main():
    gpu = load_gpu_prices()
    run_cv_for_component(gpu, "GPU", 3, "gpu")

    try:
        ram = load_ram_prices()
        # RAM series are very short, so only 1 fold is realistic
        run_cv_for_component(ram, "RAM", 1, "ram")
    except Exception as e:
        print(f"\n[RAM CV skipped: {e}]")

    try:
        ssd = load_ssd_prices()
        # SSD has long history (497 points), so full 3 folds are feasible
        run_cv_for_component(ssd, "SSD", 3, "ssd")
    except Exception as e:
        print(f"\n[SSD CV skipped: {e}]")

    try:
        cpu = load_cpu_prices()
        # CPU has 6 series x 24 points, so full 3 folds are feasible
        run_cv_for_component(cpu, "CPU", 3, "cpu")
    except Exception as e:
        print(f"\n[CPU CV skipped: {e}]")

if __name__ == "__main__":
    main()