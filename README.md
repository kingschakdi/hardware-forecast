# Hardware Component Price Forecasting Tool (PROM06 Project)

**MSc Computer Science — University of Sunderland**
**Status: Early-stage development (PROM06 Assignment 1)**

A machine-learning tool to forecast hardware component price trends as an indicator of
demand pressure, for GPUs, CPUs, RAM and SSDs. Because public unit-demand data for hardware
components is not available, price is used as an indicator of demand (see methodology notes).

---


## Data sources

| Component | Dataset | Type | Status |
|---|---|---|---|
| GPU | Kaggle GPU price history (48 models, monthly 2023–24) | Real | Loaded |
| RAM | McCallum memory prices (DDR-type level) | Real | On disk, loader TODO |
| SSD | FRED storage PPI (PCU334112) + OWID SSD trend | Real | On disk, loader TODO |
| CPU | (no time-series available) | Simulated | Remaining |

**Note on price vs demand:** Public unit-demand data for hardware components is not available.
Price is used as an indicator of demand pressure, justified by the supply-constrained nature
of the hardware market (semiconductor shortages, crypto-mining and AI-driven demand spikes).
This limitation is documented in the project methodology.

---

## Setup

Requires Python 3.10 or newer.

```bash
# 1. Clone / download this repository
# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the data exploration script (works on real GPU data)
python src/explore_data.py

# 6. Run the baseline forecast demo
python src/run_baseline.py
```

---

## Repository structure

```
hardware-forecast/
├── README.md                 # this file
├── requirements.txt          # pinned dependencies
├── data/
│   ├── raw/                  # original datasets (read-only)
│   └── processed/            # cleaned outputs
├── src/
│   ├── data_loader.py        # load + clean datasets (GPU done; others TODO)
│   ├── intermittency.py      # demand-pattern classification (done)
│   ├── models.py             # forecasting models (MA done; others stubbed)
│   ├── metrics.py            # accuracy metrics (RMSE done; others stubbed)
│   ├── explore_data.py       # exploratory data analysis script (done)
│   └── run_baseline.py       # end-to-end baseline demo (done)
├── notebooks/                # exploratory notebooks (optional)
└── outputs/                  # generated figures / results
```

---

## Reproducibility

All random operations use a fixed seed (`RANDOM_SEED = 42`) and dependencies are pinned in
`requirements.txt`, in line with the reproducibility requirement (NFR4) from the project plan.
