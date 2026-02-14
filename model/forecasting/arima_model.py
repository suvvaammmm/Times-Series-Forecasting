import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox


def run_arima(series, forecast_steps=10):

    best_aic = float("inf")
    best_order = None
    best_model = None

    # Grid search
    for p in range(0, 3):
       for d in range(0, 2):
          for q in range(0, 3):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    result = model.fit()

                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_order = (p, d, q)
                        best_model = result

                except:
                    continue

    if best_model is None:
        raise Exception("ARIMA auto selection failed")

    print("Best ARIMA Order:", best_order)

    # Forecast
    forecast_result = best_model.get_forecast(steps=forecast_steps)
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    lower = conf_int.iloc[:, 0]
    upper = conf_int.iloc[:, 1]

    # ---------------------------
    # RESIDUAL DIAGNOSTICS
    # ---------------------------

    residuals = best_model.resid

    # Bias check
    residual_mean = np.mean(residuals)

    # Ljung-Box test (autocorrelation check)
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_pvalue = lb_test["lb_pvalue"].values[0]

    return forecast, lower, upper, best_aic, residual_mean, lb_pvalue
