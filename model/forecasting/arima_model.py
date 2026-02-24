import warnings
warnings.filterwarnings("ignore")

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox


def run_arima(series, forecast_steps=10):

    best_aic = float("inf")
    best_order = None
    best_model = None

    # ---------------------------
    # 1️⃣ Grid Search
    # ---------------------------
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

    # ---------------------------
    # 2️⃣ Forecast
    # ---------------------------
    forecast_result = best_model.get_forecast(steps=forecast_steps)

    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    lower = conf_int.iloc[:, 0]
    upper = conf_int.iloc[:, 1]

    # ---------------------------
    # 3️⃣ Residual Diagnostics
    # ---------------------------
    residuals = best_model.resid

    residual_mean = np.mean(residuals)

    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_pvalue = lb_test["lb_pvalue"].values[0]

    aic = best_aic  # ✅ correct AIC

    # ---------------------------
    # 4️⃣ Safe Rounding
    # ---------------------------
    forecast = np.round(forecast, 2)
    lower = np.round(lower, 2)
    upper = np.round(upper, 2)

    try:
        aic = round(float(aic), 2)
    except:
        aic = 0.0

    try:
        residual_mean = round(float(residual_mean), 2)
    except:
        residual_mean = 0.0

    try:
        lb_pvalue = round(float(lb_pvalue), 2)
    except:
        lb_pvalue = 0.0

    return forecast, lower, upper, aic, residual_mean, lb_pvalue