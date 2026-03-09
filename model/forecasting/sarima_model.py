from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np


def run_sarima(data, forecast_steps=5):

    # Default safe values
    aic = 0.0
    residual_mean = 0.0
    lb_pvalue = 0.0

    try:
        model = SARIMAX(
            data,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 5),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        result = model.fit(disp=False)

        # AIC
        aic = result.aic

        # Forecast
        forecast_obj = result.get_forecast(steps=forecast_steps)
        forecast = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int()

        lower = conf_int.iloc[:, 0]
        upper = conf_int.iloc[:, 1]

        # Residual diagnostics
        residuals = result.resid
        residual_mean = np.mean(residuals)

        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        lb_pvalue = lb_test["lb_pvalue"].values[0]

    except Exception as e:
        print("SARIMA ERROR:", e)

        # Fallback safe outputs
        forecast = np.zeros(forecast_steps)
        lower = np.zeros(forecast_steps)
        upper = np.zeros(forecast_steps)

    # -------------------------
    # Round outputs safely
    # -------------------------
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