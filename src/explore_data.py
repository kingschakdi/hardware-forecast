"""
explore_data.py
---------------
Exploratory data analysis on the real GPU price dataset.

(Note to self on how to use this again)
Run:  python src/explore_data.py
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend, safe without a display
import matplotlib.pyplot as plt

from data_loader import load_gpu_prices
from intermittency import classify_series

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("GPU PRICE DATA — EXPLORATORY ANALYSIS")
    print("=" * 60)

    gpu = load_gpu_prices()

    # 1. Summary
    print(f"\nTotal price observations : {len(gpu)}")
    print(f"Unique GPU models        : {gpu['model'].nunique()}")
    print(f"Date range               : {gpu['date'].min().date()} "
          f"to {gpu['date'].max().date()}")
    print(f"Price range (USD)        : {gpu['price'].min():.0f} "
          f"to {gpu['price'].max():.0f}")

    # 2. Demand-pattern classification across all models
    print("\nDemand-pattern classification (first 10 models):")
    print("-" * 60)
    patterns = {}
    for model in gpu["model"].unique():
        series = gpu[gpu["model"] == model].sort_values("date")["price"].values
        result = classify_series(series)
        patterns[model] = result["pattern"]

    for model in list(patterns)[:10]:
        print(f"  {model:30s} -> {patterns[model]}")

    # Count of each pattern across the whole dataset
    counts = pd.Series(patterns).value_counts()
    print("\nPattern distribution across all models:")
    for pattern, n in counts.items():
        print(f"  {pattern:14s}: {n}")

    # 3. Plot one sample model's price history
    sample_model = gpu["model"].value_counts().index[0]  # most-observed model
    sample = gpu[gpu["model"] == sample_model].sort_values("date")

    plt.figure(figsize=(9, 4))
    plt.plot(sample["date"], sample["price"], marker="o", linewidth=1.5)
    plt.title(f"GPU retail price history — {sample_model}")
    plt.xlabel("Date")
    plt.ylabel("Retail price (USD)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "sample_gpu_price_history.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved sample price plot to: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
