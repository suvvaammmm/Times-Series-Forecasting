import warnings
warnings.filterwarnings("ignore")

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.diagnostic import acorr_ljungbox


def run_ets(series, forecast_steps=5):
    series = series.astype(float).dropna()

    best_aic = float("inf")
    best_model = None

    # Grid search over trend and seasonal configs
    for trend in ["add", "mul", None]:
        for seasonal in ["add", "mul", None]:
            try:
                model = ExponentialSmoothing(
                    series,
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=5,   # 5-day week for stocks
                    initialization_method="estimated"
                )
                result = model.fit(optimized=True, remove_bias=True)

                if result.aic < best_aic:
                    best_aic = result.aic
                    best_model = result
            except:
                continue

    if best_model is None:
        raise Exception("ETS model fitting failed")

    forecast = best_model.forecast(forecast_steps)

    # Confidence interval via simulation (bootstrap residuals)
    residuals = best_model.resid
    std_resid = np.std(residuals)
    lower = forecast - 1.96 * std_resid
    upper = forecast + 1.96 * std_resid

    aic = round(float(best_aic), 2)
    residual_mean = round(float(np.mean(residuals)), 4)

    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_pvalue = round(float(lb_test["lb_pvalue"].values[0]), 4)

    forecast = np.round(forecast.values.astype(float), 2)
    lower = np.round(lower.values.astype(float), 2)
    upper = np.round(upper.values.astype(float), 2)

    return forecast, lower, upper, aic, residual_mean, lb_pvalue