from data_loader import load_gpu_prices
from ml_models import (random_forest_forecast, gradient_boosting_forecast,
                       mlp_forecast, lstm_forecast)
from metrics import rmse
import numpy as np

gpu = load_gpu_prices()

for name, fn in [("Random Forest", random_forest_forecast),
                 ("Gradient Boosting", gradient_boosting_forecast),
                 ("MLP", mlp_forecast),
                 ("LSTM", lstm_forecast)]:
    forecasts, holdouts = fn(gpu, horizon=3)
    errors = [rmse(holdouts[m][1], forecasts[m]) for m in forecasts]
    print(f"{name:20s} avg RMSE: {np.mean(errors):.2f} USD  ({len(errors)} models)")