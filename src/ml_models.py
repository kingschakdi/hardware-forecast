"""
ml_models.py
------------
Machine-learning forecasters for hardware price series.

Unlike the statistical baselines in models.py (which forecast each series
from its own history), these models are trained POOLED: one model learns
from lag-based features across ALL GPU series at once. This is necessary
because each individual series is short (~26 monthly points), too little
to train a tree ensemble per series.

Feature engineering is controlled by a switch:
    feature_set="simple" : lag features only
    feature_set="rich"   : lags + rolling mean/std + month
This lets us start simple and later compare the two (see evaluation phase).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def make_features(series: pd.Series, n_lags: int = 3, feature_set: str = "simple"):
    """
    Turn a single price series into a supervised-learning table.

    Each row is one time point; columns are its predictors (lagged prices,
    and optionally rolling stats + month), and the target is that point's
    price. Rows without enough history for all lags are dropped.

    Parameters
    ----------
    series : pd.Series   time-indexed price series (chronological)
    n_lags : int         how many previous months to use as predictors
    feature_set : str    "simple" (lags only) or "rich" (lags + extras)

    Returns
    -------
    (X, y) : (pd.DataFrame, pd.Series)   features and target
    """
    s = series.sort_index()
    df = pd.DataFrame({"price": s.values}, index=s.index)

    # Lag features: price 1, 2, ... n_lags months ago
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["price"].shift(lag)

    if feature_set == "rich":
        # Rolling statistics over the last n_lags months (shifted so they
        # only use PAST data — no leakage of the current value)
        df["roll_mean"] = df["price"].shift(1).rolling(n_lags).mean()
        df["roll_std"] = df["price"].shift(1).rolling(n_lags).std()
        # Month as a simple seasonal signal
        df["month"] = df.index.month

    df = df.dropna()  # drop early rows without full history

    feature_cols = [c for c in df.columns if c != "price"]
    return df[feature_cols], df["price"]


def build_training_table(gpu_df: pd.DataFrame, n_lags: int = 3,
                         feature_set: str = "simple", horizon: int = 3):
    """
    Build ONE pooled training table from all GPU series.

    For each model, hold out the last `horizon` points (so they are never
    seen in training), turn the remaining history into features, and stack
    everything into a single X / y. Returns the pooled training data plus a
    per-series record of held-out actuals for later evaluation.
    """
    X_parts, y_parts = [], []
    holdouts = {}  # model_name -> (train_series, actual_values)

    for model in gpu_df["model"].unique():
        s = gpu_df[gpu_df["model"] == model].sort_values("date").set_index("date")["price"]
        if len(s) < horizon + n_lags + 1:
            continue  # not enough history for features + a fair hold-out

        train = s.iloc[:-horizon]
        actual = s.iloc[-horizon:].values
        holdouts[model] = (train, actual)

        X, y = make_features(train, n_lags=n_lags, feature_set=feature_set)
        if len(X) > 0:
            X_parts.append(X)
            y_parts.append(y)

    if not X_parts:
        raise ValueError("no series had enough data to build features")

    X_all = pd.concat(X_parts, ignore_index=True)
    y_all = pd.concat(y_parts, ignore_index=True)
    return X_all, y_all, holdouts


# ---------------------------------------------------------------------------
# Recursive forecasting helper
# ---------------------------------------------------------------------------
def _forecast_series(model, train_series, n_lags, feature_set, horizon):
    """
    Forecast `horizon` steps ahead for one series by feeding each prediction
    back in as the newest lag (recursive multi-step forecasting).
    """
    history = list(train_series.sort_index().values)
    idx = train_series.sort_index().index
    last_month = idx[-1].month if hasattr(idx[-1], "month") else 1

    preds = []
    for h in range(horizon):
        row = {}
        for lag in range(1, n_lags + 1):
            row[f"lag_{lag}"] = history[-lag]
        if feature_set == "rich":
            recent = history[-n_lags:]
            row["roll_mean"] = np.mean(recent)
            row["roll_std"] = np.std(recent, ddof=1) if len(recent) > 1 else 0.0
            row["month"] = ((last_month + h) % 12) + 1
        X_row = pd.DataFrame([row])
        yhat = float(model.predict(X_row)[0])
        preds.append(yhat)
        history.append(yhat)  # feed prediction back in
    return np.array(preds)


# ---------------------------------------------------------------------------
# The two tree models
# ---------------------------------------------------------------------------
def random_forest_forecast(gpu_df, n_lags: int = 3, feature_set: str = "simple",
                           horizon: int = 3, random_state: int = 42):
    """
    Train ONE Random Forest on pooled features from all GPU series, then
    forecast each series' held-out horizon. Returns {model_name: forecast}.
    """
    X, y, holdouts = build_training_table(gpu_df, n_lags, feature_set, horizon)
    rf = RandomForestRegressor(n_estimators=200, random_state=random_state)
    rf.fit(X, y)

    forecasts = {}
    for model, (train, _actual) in holdouts.items():
        forecasts[model] = _forecast_series(rf, train, n_lags, feature_set, horizon)
    return forecasts, holdouts


def gradient_boosting_forecast(gpu_df, n_lags: int = 3, feature_set: str = "simple",
                               horizon: int = 3, random_state: int = 42):
    """
    Same as random_forest_forecast but with Gradient Boosting.
    """
    X, y, holdouts = build_training_table(gpu_df, n_lags, feature_set, horizon)
    gb = GradientBoostingRegressor(n_estimators=200, random_state=random_state)
    gb.fit(X, y)

    forecasts = {}
    for model, (train, _actual) in holdouts.items():
        forecasts[model] = _forecast_series(gb, train, n_lags, feature_set, horizon)
    return forecasts, holdouts

# ---------------------------------------------------------------------------
# Multi-Layer Perceptron (Keras) — pooled training
# ---------------------------------------------------------------------------
def mlp_forecast(gpu_df, n_lags: int = 3, feature_set: str = "simple",
                 horizon: int = 3, random_state: int = 42,
                 epochs: int = 100, verbose: int = 0):
    """
    Train ONE Multi-Layer Perceptron on pooled features from all GPU series,
    then forecast each series' held-out horizon.

    Neural networks are scale-sensitive, so features are standardised before
    training. Returns ({model_name: forecast}, holdouts) to match the tree
    models' interface.
    """
    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    X, y, holdouts = build_training_table(gpu_df, n_lags, feature_set, horizon)

    # Standardise features (fit the scaler on training data only)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    n_features = X_scaled.shape[1]

    # A small MLP: two hidden layers. Kept modest because the dataset is small.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),  # single price output
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_scaled, y.values, epochs=epochs, verbose=verbose)

    # Recursive forecast per series, scaling each feature row the same way
    forecasts = {}
    for name, (train, _actual) in holdouts.items():
        forecasts[name] = _forecast_series_scaled(
            model, scaler, train, n_lags, feature_set, horizon
        )
    return forecasts, holdouts


def _forecast_series_scaled(model, scaler, train_series, n_lags, feature_set, horizon):
    """
    Like _forecast_series, but applies the fitted scaler to each feature row
    before predicting (needed for the neural network).
    """
    history = list(train_series.sort_index().values)
    idx = train_series.sort_index().index
    last_month = idx[-1].month if hasattr(idx[-1], "month") else 1

    preds = []
    for h in range(horizon):
        row = {}
        for lag in range(1, n_lags + 1):
            row[f"lag_{lag}"] = history[-lag]
        if feature_set == "rich":
            recent = history[-n_lags:]
            row["roll_mean"] = np.mean(recent)
            row["roll_std"] = np.std(recent, ddof=1) if len(recent) > 1 else 0.0
            row["month"] = ((last_month + h) % 12) + 1
        X_row = pd.DataFrame([row])
        X_row_scaled = scaler.transform(X_row.values)
        yhat = float(model.predict(X_row_scaled, verbose=0)[0][0])
        preds.append(yhat)
        history.append(yhat)
    return np.array(preds)

# ---------------------------------------------------------------------------
# LSTM (Keras) — pooled training on sequences
# ---------------------------------------------------------------------------
def _make_sequences(series_values, window):
    """
    Turn a 1-D price series into overlapping (input_window -> next_value)
    pairs for LSTM training.

    Returns X of shape (n_samples, window, 1) and y of shape (n_samples,).
    """
    X, y = [], []
    for i in range(len(series_values) - window):
        X.append(series_values[i:i + window])
        y.append(series_values[i + window])
    if not X:
        return np.empty((0, window, 1)), np.empty((0,))
    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y)
    return X, y


def lstm_forecast(gpu_df, window: int = 4, horizon: int = 3,
                  random_state: int = 42, epochs: int = 100, verbose: int = 0):
    """
    Train ONE LSTM on pooled sequences from all GPU series, then forecast
    each series' held-out horizon recursively.

    Prices are scaled to [0, 1] per the pooled training set before training,
    since LSTMs (like all neural nets) are scale-sensitive. Returns
    ({model_name: forecast}, holdouts) to match the other models.
    """
    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    # Build hold-outs the same way as the other models
    holdouts = {}
    for model in gpu_df["model"].unique():
        s = gpu_df[gpu_df["model"] == model].sort_values("date").set_index("date")["price"]
        if len(s) < horizon + window + 1:
            continue
        train = s.iloc[:-horizon]
        actual = s.iloc[-horizon:].values
        holdouts[model] = (train, actual)

    if not holdouts:
        raise ValueError("no series had enough data for the LSTM window + hold-out")

    # Scale all training prices to [0, 1] using one global min/max
    all_train_vals = np.concatenate([t.values for t, _ in holdouts.values()])
    p_min, p_max = float(all_train_vals.min()), float(all_train_vals.max())
    rng = (p_max - p_min) or 1.0

    def scale(v):
        return (np.asarray(v, dtype=float) - p_min) / rng

    def unscale(v):
        return np.asarray(v, dtype=float) * rng + p_min

    # Build pooled training sequences from all series (scaled)
    X_parts, y_parts = [], []
    for train, _actual in holdouts.values():
        Xs, ys = _make_sequences(scale(train.values), window)
        if len(Xs) > 0:
            X_parts.append(Xs)
            y_parts.append(ys)

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)

    # A small LSTM, kept modest for the small dataset
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, 1)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_all, y_all, epochs=epochs, verbose=verbose)

    # Recursive forecast per series
    forecasts = {}
    for name, (train, _actual) in holdouts.items():
        scaled_hist = list(scale(train.values))
        preds_scaled = []
        for _ in range(horizon):
            window_in = np.array(scaled_hist[-window:]).reshape(1, window, 1)
            yhat = float(model.predict(window_in, verbose=0)[0][0])
            preds_scaled.append(yhat)
            scaled_hist.append(yhat)
        forecasts[name] = unscale(np.array(preds_scaled))

    return forecasts, holdouts



