"""
app.py
------
Streamlit dashboard for the Hardware Component Price Forecasting Tool.

Pick a component (GPU or RAM), a model and a forecasting method to see a
short-term price forecast alongside recent history. ML models produce
recursive, non-flat forecasts; baselines are flat by design.

Run with:  streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import streamlit as st

from data_loader import load_gpu_prices, load_ram_prices, load_ssd_prices, load_cpu_prices
from models import moving_average, exponential_smoothing, crostons_method
from ml_models import (random_forest_forecast, gradient_boosting_forecast,
                       mlp_forecast, lstm_forecast)

st.set_page_config(page_title="Hardware Price Forecasting", layout="centered")
st.title("Hardware Component Price Forecasting Tool")
st.write(
    "Select a component, model and forecasting method to see a short-term "
    "price forecast based on historical data."
)

FORECAST_HORIZON = 6  # dashboard always computes up to 6 months

# Each component: its loader, results CSV, and price-unit label.
COMPONENTS = {
    "GPU": {"loader": load_gpu_prices, "csv": "model_comparison_gpu.csv",
            "unit": "USD per card"},
    "RAM": {"loader": load_ram_prices, "csv": "model_comparison_ram.csv",
            "unit": "USD per GB"},
    "SSD": {"loader": load_ssd_prices, "csv": "model_comparison_ssd.csv",
            "unit": "estimated USD per GB"},
    "CPU": {"loader": load_cpu_prices, "csv": "model_comparison_cpu.csv",
            "unit": "USD (simulated)"},
}


@st.cache_data
def load_accuracy(csv_name):
    path = os.path.join(os.path.dirname(__file__), "outputs", csv_name)
    try:
        df = pd.read_csv(path)
        acc = {}
        for _, row in df.iterrows():
            acc[row["method"]] = {"rmse": row["rmse"],
                                  "mae": row["mae"],
                                  "mase": row["mase"]}
        return acc
    except (FileNotFoundError, OSError):
        return {}


@st.cache_data
def get_data(component):
    return COMPONENTS[component]["loader"]()


# ML models train once PER COMPONENT and are cached.
@st.cache_resource(show_spinner="Training machine-learning models (first load per component)...")
def get_ml_forecasts(component, horizon):
    df = get_data(component)
    ml = {}
    for label, fn in [("Random Forest", random_forest_forecast),
                      ("Gradient Boosting", gradient_boosting_forecast),
                      ("MLP", mlp_forecast),
                      ("LSTM", lstm_forecast)]:
        try:
            forecasts, _holdouts = fn(df, horizon=horizon)
            ml[label] = forecasts
        except Exception:
            # Some models (e.g. LSTM) need more history than short series
            # like RAM provide. Skip them gracefully for that component.
            ml[label] = {}
    return ml


# --- Component selector (drives everything below) ---
component = st.selectbox("Component", list(COMPONENTS.keys()))

data = get_data(component)
unit = COMPONENTS[component]["unit"]
METHOD_ACCURACY = load_accuracy(COMPONENTS[component]["csv"])

models = sorted(data["model"].unique())
chosen_model = st.selectbox("Model", models)
method_name = st.selectbox(
    "Forecasting method",
    ["Moving Average", "Exponential Smoothing", "Croston's",
     "Random Forest", "Gradient Boosting", "MLP", "LSTM"],
)
months_to_show = st.slider("Months to forecast", min_value=1, max_value=6, value=3)

series = (
    data[data["model"] == chosen_model]
    .sort_values("date")
    .set_index("date")["price"]
)
history = series.values

# Always compute the full 6-month horizon, then show only what the slider asks
if method_name == "Moving Average":
    full_forecast = moving_average(history, horizon=FORECAST_HORIZON, window=3)
elif method_name == "Exponential Smoothing":
    full_forecast = exponential_smoothing(history, horizon=FORECAST_HORIZON, alpha=0.3)
elif method_name == "Croston's":
    full_forecast = crostons_method(history, horizon=FORECAST_HORIZON)
else:
    ml_forecasts = get_ml_forecasts(component, FORECAST_HORIZON)
    method_dict = ml_forecasts.get(method_name, {})
    full_forecast = method_dict.get(chosen_model, None)

st.subheader(f"{chosen_model} ({component}) — {method_name}")

if full_forecast is None:
    st.warning(
        "This model does not have enough history for the machine-learning "
        "methods. Try a traditional method, or pick another model."
    )
else:
    forecast = np.asarray(full_forecast)[:months_to_show]  # slider trims it

    last_date = series.index[-1]
    last_price = series.values[-1]
    future_dates = pd.date_range(last_date, periods=len(forecast) + 1, freq="MS")[1:]

    # History column
    hist_df = pd.DataFrame({"price": series.values}, index=series.index)

    # Forecast column: start it at the last actual point so the lines connect
    fcst_index = [last_date] + list(future_dates)
    fcst_values = [last_price] + list(forecast)
    fcst_df = pd.DataFrame({"forecast": fcst_values}, index=fcst_index)

    chart_df = pd.concat([hist_df, fcst_df], axis=1)
    st.line_chart(chart_df)

    direction = "flat"
    if forecast[-1] > forecast[0] * 1.01:
        direction = "rising"
    elif forecast[-1] < forecast[0] * 0.99:
        direction = "falling"
    st.caption(f"Forecast trend: **{direction}** "
               f"(from ${forecast[0]:.2f} to ${forecast[-1]:.2f} over "
               f"{len(forecast)} months, {unit})")

    # --- Accuracy of this method (from evaluation on held-out data) ---
    acc = METHOD_ACCURACY.get(method_name)
    if acc:
        st.write("**How reliable is this method?**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg error (RMSE)", f"{acc['rmse']:.2f}")
        c2.metric("Avg error (MAE)", f"{acc['mae']:.2f}")
        c3.metric("MASE", f"{acc['mase']:.2f}")
        naive_note = ("better than" if acc["mase"] < 1 else "worse than")
        st.caption(
            f"On tested data, this method's forecasts were off by about "
            f"{acc['mae']:.2f} ({unit}) on average. A MASE of {acc['mase']:.2f} "
            f"means it performed {naive_note} a simple 'next month = this month' "
            f"guess. All figures are indicative; forecasts are estimates, not guarantees."
        )

    st.write("**Forecast values:**")
    st.dataframe(
        pd.DataFrame(
            {"month": future_dates.strftime("%Y-%m"),
             f"forecast price ({unit})": np.round(forecast, 2)}
        ),
        hide_index=True,
    )
    export_df = pd.DataFrame({
        "component": component,
        "model": chosen_model,
        "method": method_name,
        "month": future_dates.strftime("%Y-%m"),
        f"forecast_price ({unit})": np.round(forecast, 4),
    })
    st.download_button(
        label="Download forecast as CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"forecast_{component}_{chosen_model}_{method_name}.csv".replace(" ", "_"),
        mime="text/csv",
    )