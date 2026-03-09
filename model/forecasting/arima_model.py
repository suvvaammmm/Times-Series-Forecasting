import warnings
warnings.filterwarnings("ignore")

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox


def run_arima(series, forecast_steps=5):

    # ---------------------------
    # 0️⃣ Clean Series
    # ---------------------------
    series = series.astype(float).dropna()

    # Use log transform (percentage growth modeling)
    log_series = np.log(series)

    best_aic = float("inf")
    best_order = None
    best_model = None

    # ---------------------------
    # 1️⃣ Grid Search with Drift
    # ---------------------------
    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    model = ARIMA(
                        log_series,
                        order=(p, d, q),
                        trend="t"  # 👈 Drift added
                    )
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
    # 2️⃣ Forecast (Log Scale)
    # ---------------------------
    forecast_result = best_model.get_forecast(steps=forecast_steps)

    forecast_log = forecast_result.predicted_mean
    conf_int_log = forecast_result.conf_int()

    # Convert back to price scale
    forecast = np.exp(forecast_log)
    lower = np.exp(conf_int_log.iloc[:, 0])
    upper = np.exp(conf_int_log.iloc[:, 1])

    # ---------------------------
    # 3️⃣ Residual Diagnostics
    # ---------------------------
    residuals = best_model.resid
    residual_mean = float(np.mean(residuals))

    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_pvalue = float(lb_test["lb_pvalue"].values[0])

    aic = float(best_aic)

    # ---------------------------
    # 4️⃣ Safe Rounding
    # ---------------------------
    forecast = np.round(forecast.values.astype(float), 2)
    lower = np.round(lower.values.astype(float), 2)
    upper = np.round(upper.values.astype(float), 2)

    aic = round(aic, 2)
    residual_mean = round(residual_mean, 4)
    lb_pvalue = round(lb_pvalue, 4)

    return forecast, lower, upper, aic, residual_mean, lb_pvalue