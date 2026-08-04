import numpy as np
import pandas as pd

RANDOM_SEED = 42  # fixed for reproducibility (NFR4)


# ---------------------------------------------------------------------------
# Moving Average - COMPLETED/MIGHT NEED A RECHECK LATER
# ---------------------------------------------------------------------------
def moving_average(history, horizon: int = 3, window: int = 3):
    """
    Simple Moving Average forecast.

    Uses the mean of the last `window` observations as the forecast for every
    step in the horizon (a flat forecast — the standard naive MA baseline).

    Parameters
    ----------
    history : sequence of float   historical price values (chronological)
    horizon : int                 number of future steps to forecast
    window  : int                 number of recent points to average

    Returns
    -------
    np.ndarray of length `horizon`
    """
    hist = pd.Series(history).dropna().values
    if len(hist) == 0:
        raise ValueError("history is empty")

    window = min(window, len(hist))
    forecast_value = hist[-window:].mean()
    return np.repeat(forecast_value, horizon)


# ---------------------------------------------------------------------------
# Simple Exponential Smoothing
# ---------------------------------------------------------------------------
def exponential_smoothing(history, horizon: int = 3, alpha: float = 0.3):
    """
    Simple Exponential Smoothing (SES) forecast.

    Produces a smoothed level by weighting recent observations more heavily
    than older ones, controlled by alpha. The final smoothed level is used
    as a flat forecast across the horizon, matching the other baselines.

    Parameters
    ----------
    history : sequence of float   historical values (chronological)
    horizon : int                 number of future steps to forecast
    alpha   : float               smoothing factor, 0 < alpha < 1
                                  (higher = more weight on recent points)

    Returns
    -------
    np.ndarray of length `horizon`
    """
    hist = pd.Series(history).dropna().values
    if len(hist) == 0:
        raise ValueError("history is empty")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

    # Initialise the level with the first observation, then update it
    # one step at a time using the SES recurrence:
    #   level = alpha * observation + (1 - alpha) * previous level
    level = float(hist[0])
    for value in hist[1:]:
        level = alpha * float(value) + (1 - alpha) * level

    return np.repeat(level, horizon)



# ---------------------------------------------------------------------------
# Croston's method — IN PROGRESS
# ---------------------------------------------------------------------------
def crostons_method(history, horizon: int = 3, alpha: float = 0.1, sba: bool = False):
    """
    Croston's method for intermittent demand (Croston, 1972), with an
    optional Syntetos-Boylan Approximation bias correction (Syntetos &
    Boylan, 2005).

    Separates the series into non-zero demand sizes and the intervals
    between non-zero demands, smooths each with simple exponential
    smoothing, and forecasts size / interval. Returns a flat forecast
    over the horizon, matching the other baselines.

    Parameters
    ----------
    history : sequence of float   historical values (chronological)
    horizon : int                 number of future steps to forecast
    alpha   : float               smoothing parameter (0 < alpha < 1)
    sba     : bool                if True, apply the SBA bias correction

    Returns
    -------
    np.ndarray of length `horizon`
    """
    hist = pd.Series(history).dropna().values
    if len(hist) == 0:
        raise ValueError("history is empty")

    # Indices where demand is non-zero
    nonzero_idx = np.flatnonzero(hist)

    # If there is no non-zero demand at all, the forecast is zero.
    if len(nonzero_idx) == 0:
        return np.repeat(0.0, horizon)

    # Demand sizes at the non-zero points
    sizes = hist[nonzero_idx].astype(float)

    # Intervals between consecutive non-zero demands.
    # The first interval is the position of the first non-zero demand + 1.
    intervals = np.diff(nonzero_idx).astype(float)
    first_interval = float(nonzero_idx[0] + 1)
    intervals = np.insert(intervals, 0, first_interval)

    # Simple exponential smoothing, initialised with the first observed value
    z = sizes[0]        # smoothed demand size
    x = intervals[0]    # smoothed interval
    for i in range(1, len(sizes)):
        z = z + alpha * (sizes[i] - z)
        x = x + alpha * (intervals[i] - x)

    # Croston's forecast = smoothed size / smoothed interval
    forecast_value = z / x

    # Optional Syntetos-Boylan bias correction
    if sba:
        forecast_value *= (1.0 - alpha / 2.0)

    return np.repeat(forecast_value, horizon)


# ---------------------------------------------------------------------------
# Advanced models — IN PROGRESS
# ---------------------------------------------------------------------------
def random_forest_forecast(*args, **kwargs):
    """REMAINING: Random Forest regressor with lag features (later phase)."""
    raise NotImplementedError("Random Forest model is a later-phase task.")


def mlp_forecast(*args, **kwargs):
    """REMAINING: Multi-Layer Perceptron (later phase, TensorFlow/Keras)."""
    raise NotImplementedError("MLP model is a later-phase task.")


def lstm_forecast(*args, **kwargs):
    """REMAINING: LSTM network (later phase, TensorFlow/Keras)."""
    raise NotImplementedError("LSTM model is a later-phase task.")


if __name__ == "__main__":
    import numpy as np
    # Intermittent series (Croston's natural case)
    intermittent = [0, 0, 5, 0, 0, 0, 8, 0, 4]
    print("Croston (intermittent):", crostons_method(intermittent, horizon=3))
    print("SBA     (intermittent):", crostons_method(intermittent, horizon=3, sba=True))
    # Continuous price-like series (your actual case)
    prices = [300, 310, 305, 320, 315, 330]
    print("ExpSmoothing (a=0.3):", exponential_smoothing(prices, horizon=3, alpha=0.3))
    print("ExpSmoothing (a=0.7):", exponential_smoothing(prices, horizon=3, alpha=0.7))
    