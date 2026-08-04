"""
data_loader.py
--------------
Loads and cleans hardware component price datasets into a common tidy format:
    columns = [component, model, date, price, frequency, source, data_type]
    where data_type is 'real' or 'simulated'.

STATUS (PROM06 Assignment 1 — early stage):
    - load_gpu_prices()  : COMPLETED (real Kaggle data)
    - load_ram_prices()  : IN PROGRESS (real data on disk, parsing TODO)
    - load_ssd_prices()  : IN PROGRESS (real data on disk, parsing TODO)
    - load_cpu_prices()  : REMAINING (no real series; will be simulated later)
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


# ---------------------------------------------------------------------------
# GPU — COMPLETED (real data)
# ---------------------------------------------------------------------------
def load_gpu_prices() -> pd.DataFrame:
    """
    Load and clean the Kaggle GPU price-history dataset.

    Returns a tidy DataFrame with one row per (model, date):
        component | model | date | price | frequency | source | data_type

    Cleaning steps applied:
        - parse the DD-MM-YY date strings into proper datetimes
        - drop rows where retail price is 0 (these are missing-value placeholders)
        - keep retail price as the forecast variable (used price dropped for now)
    """
    path = RAW_DIR / "gpu_price_history.csv"
    df = pd.read_csv(path)

    # The raw columns are: Date, Name, Retail Price, Used Price
    df = df.rename(columns={
        "Name": "model",
        "Retail Price": "price",
        "Date": "date_raw",
    })

    # Dates look like "01-01-24" (DD-MM-YY). Parse explicitly to avoid ambiguity.
    df["date"] = pd.to_datetime(df["date_raw"], format="%d-%m-%y", errors="coerce")

    # Retail price: coerce to numeric, then drop the 0 / missing placeholders.
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"].fillna(0) > 0].copy()

    df["component"] = "GPU"
    df["frequency"] = "monthly"
    df["source"] = "Kaggle: historical-gpu-prices-nvidia-and-amd"
    df["data_type"] = "real"

    tidy = df[["component", "model", "date", "price",
               "frequency", "source", "data_type"]]
    tidy = tidy.sort_values(["model", "date"]).reset_index(drop=True)
    return tidy


def get_gpu_series(model_name: str) -> pd.Series:
    """
    Convenience helper: return a single GPU model's price as a time-indexed Series.
    Used by the baseline demo and the intermittency classifier.
    """
    df = load_gpu_prices()
    one = df[df["model"] == model_name].set_index("date")["price"]
    return one.sort_index()


# ---------------------------------------------------------------------------
# RAM — IN PROGRESS
# ---------------------------------------------------------------------------
def load_ram_prices() -> pd.DataFrame:
    """
    Load RAM price data from the RamRadar price index.

    Source is a daily index of average price-per-GB, broken down by RAM type
    (DDR4/DDR5) and form factor (DIMM/SODIMM). Each type+form-factor pair is
    treated as a separate 'model' series, resampled from daily to monthly to
    match the GPU data's frequency.

    NOTE: available history is short (roughly Oct 2025 - Jul 2026), so each
    RAM series has only ~9-10 monthly points. This limited web-available
    history is a documented limitation of the RAM analysis.

    Returns a tidy DataFrame with columns:
        component, model, date, price, frequency, source, data_type
    """
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "data", "raw",
                        "ramradar-price-index.csv")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), "..", "ramradar-price-index.csv")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["model"] = df["ram_type"].astype(str) + " " + df["form_factor"].astype(str)

    rows = []
    for model, grp in df.groupby("model"):
        monthly = (grp.set_index("date")["avg_price_per_gb"]
                      .resample("MS").mean().dropna())
        for date, price in monthly.items():
            rows.append({
                "component": "RAM",
                "model": model,
                "date": date,
                "price": round(float(price), 4),
                "frequency": "monthly",
                "source": "RamRadar price index",
                "data_type": "real",
            })

    out = pd.DataFrame(rows).sort_values(["model", "date"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# SSD — IN PROGRESS
# ---------------------------------------------------------------------------
def load_ssd_prices() -> pd.DataFrame:
    """
    Load SSD price data derived from the FRED producer price index PCU33443344.

    The source is a monthly price INDEX (not dollar prices), so it is converted
    to an approximate USD-per-GB series by anchoring it to a known reference
    retail price. A mainstream 1TB NVMe SSD cost roughly $0.09/GB in early 2026
    (retail price trackers, e.g. BuyPerUnit/Tom's Hardware, 2026). The index at
    its most recent point is scaled to that anchor, and all other months follow
    the index's relative movements.

    IMPORTANT: the resulting prices are ESTIMATES, not measured retail prices.
    They preserve the index's relative changes but express them in approximate
    dollar terms. The anchor was taken during an atypical NAND-shortage price
    spike, so absolute figures reflect inflated 2026 market conditions.

    This is a SINGLE economy-wide series (not per-model), unlike the GPU data.

    Returns a tidy DataFrame with columns:
        component, model, date, price, frequency, source, data_type
    """
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "data", "raw",
                        "PCU33443344.csv")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), "..", "PCU33443344.csv")

    df = pd.read_csv(path)
    df.columns = ["date", "index_value"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Anchor: scale the index so the most recent point equals the reference
    # USD/GB price of a 1TB NVMe SSD in early 2026.
    ANCHOR_USD_PER_GB = 0.09
    latest_index = df["index_value"].iloc[-1]
    scale = ANCHOR_USD_PER_GB / latest_index

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "component": "SSD",
            "model": "SSD (NAND price index)",
            "date": r["date"],
            "price": round(float(r["index_value"]) * scale, 4),  # est. USD/GB
            "frequency": "monthly",
            "source": "FRED PCU33443344, anchored to ~$0.09/GB (1TB NVMe, 2026)",
            "data_type": "estimated",
        })

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CPU — REMAINING (simulation, later phase)
# ---------------------------------------------------------------------------

def load_cpu_prices() -> pd.DataFrame:
    """
    Generate SIMULATED CPU price data.

    No real per-model CPU price time-series was available (only specification
    sheets and launch prices, or aggregate producer-price indices with no
    brand/model detail). CPU prices are therefore simulated so the tool can
    demonstrate multi-series forecasting for this component. Extending the
    study with genuine per-model CPU price data is noted as future work.

    Design: several fictional CPU models grouped into two nominal brand
    categories ("Intel-class" and "AMD-class"), each simulated as a mild
    downward price trend (hardware depreciation) plus random monthly noise.
    A fixed seed makes the data reproducible. Brand differences are
    illustrative only and do NOT reflect real Intel/AMD market behaviour.

    Returns a tidy DataFrame with columns:
        component, model, date, price, frequency, source, data_type
    """
    rng = np.random.default_rng(42)  # fixed seed for reproducibility

    # (brand label, model name, starting price USD, monthly trend, noise sd)
    specs = [
        ("Intel class", "Sim Core i3 class", 180, -1.2, 6),
        ("Intel class", "Sim Core i5 class", 260, -1.8, 8),
        ("Intel class", "Sim Core i7 class", 380, -2.5, 11),
        ("AMD class",   "Sim Ryzen 3 class", 160, -1.0, 7),
        ("AMD class",   "Sim Ryzen 5 class", 240, -1.6, 9),
        ("AMD class",   "Sim Ryzen 7 class", 350, -2.3, 12),
            ]

    n_months = 24
    dates = pd.date_range("2023-01-01", periods=n_months, freq="MS")

    rows = []
    for brand, model, start, trend, noise_sd in specs:
        price = float(start)
        for i, date in enumerate(dates):
            # trend applied cumulatively, plus random monthly shock
            value = start + trend * i + rng.normal(0, noise_sd)
            value = max(value, 20.0)  # floor so prices never go absurdly low
            rows.append({
                "component": "CPU",
                "model": f"{model} ({brand})",
                "date": date,
                "price": round(float(value), 2),
                "frequency": "monthly",
                "source": "SIMULATED (no real CPU price series available)",
                "data_type": "simulated",
            })

    return pd.DataFrame(rows).sort_values(["model", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    # Quick smoke test of the completed GPU loader.
    gpu = load_gpu_prices()
    print(f"Loaded {len(gpu)} GPU price rows across "
          f"{gpu['model'].nunique()} models.")
    print(gpu.head())
