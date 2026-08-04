"""
intermittency.py
----------------
Classifies a demand/price series into one of the four demand patterns from the
literature (Syntetos & Boylan; used by Lucht et al., 2022):

    - smooth        : regular timing, low variability
    - erratic       : regular timing, high variability
    - intermittent  : irregular timing, low variability
    - lumpy         : irregular timing, high variability

Classification uses two standard statistics:
    - ADI  (Average Demand Interval): mean gap between non-zero observations
    - CV^2 (squared Coefficient of Variation of non-zero values)

Standard cut-offs from the literature:
    ADI  cut-off = 1.32
    CV^2 cut-off = 0.49

STATUS: COMPLETED for Assignment 1.
"""

import numpy as np
import pandas as pd

ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49


def classify_series(values) -> dict:
    """
    Classify a 1-D sequence of demand/price values into a demand pattern.

    Returns a dict with adi, cv2, and the pattern label. This is the same
    quadrant scheme discussed in the literature review (Lucht et al., 2022).
    """
    s = pd.Series(values).dropna()
    s = s[s != 0]  # treat zeros as "no demand" periods

    if len(s) < 2:
        return {"adi": np.nan, "cv2": np.nan, "pattern": "insufficient_data"}

    n_periods = len(pd.Series(values).dropna())
    n_nonzero = len(s)
    adi = n_periods / n_nonzero if n_nonzero else np.nan

    mean = s.mean()
    std = s.std(ddof=0)
    cv2 = (std / mean) ** 2 if mean else np.nan

    if adi < ADI_CUTOFF and cv2 < CV2_CUTOFF:
        pattern = "smooth"
    elif adi < ADI_CUTOFF and cv2 >= CV2_CUTOFF:
        pattern = "erratic"
    elif adi >= ADI_CUTOFF and cv2 < CV2_CUTOFF:
        pattern = "intermittent"
    else:
        pattern = "lumpy"

    return {"adi": round(adi, 3), "cv2": round(cv2, 3), "pattern": pattern}


if __name__ == "__main__":
    # Demo on real GPU data.
    from data_loader import load_gpu_prices

    gpu = load_gpu_prices()
    print("Demand-pattern classification for first 5 GPU models:\n")
    for model in gpu["model"].unique()[:5]:
        series = gpu[gpu["model"] == model]["price"].values
        result = classify_series(series)
        print(f"  {model:28s} -> {result['pattern']:14s} "
              f"(ADI={result['adi']}, CV2={result['cv2']})")
