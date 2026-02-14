from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np

def run_sarima(data, forecast_steps=10):

    model = SARIMAX(
        data,
        order=(1,1,1),
        seasonal_order=(1,0,1,5),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    result = model.fit(disp=False)

    forecast_obj = result.get_forecast(steps=forecast_steps)

    forecast = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()

    lower = conf_int.iloc[:, 0]
    upper = conf_int.iloc[:, 1]

    # --- Residual Diagnostics ---
    residuals = result.resid
    residual_mean = np.mean(residuals)

    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_pvalue = lb_test['lb_pvalue'].values[0]

    return forecast, lower, upper, result.aic, residual_mean, lb_pvalue