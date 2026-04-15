"""
ETS (Error, Trend, Seasonality) — Exponential Smoothing model.
Uses statsmodels ExponentialSmoothing with auto-tuned parameters.
Returns the same interface as ARIMA/SARIMA/Ridge models.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def run_ets(series: pd.Series, steps: int = 10):
    """
    Fit an ETS model and produce a multi-step forecast.

    Parameters
    ----------
    series : pd.Series   – historical values
    steps  : int         – number of forecast steps (default 10)

    Returns
    -------
    forecast      : list[float]
    lower_ci      : list[float]
    upper_ci      : list[float]
    aic           : float
    residual_mean : float
    lb_pvalue     : float   (Ljung-Box p-value on residuals, or 0.5 fallback)
    """
    series = pd.Series(series).dropna().reset_index(drop=True)

    # ── Choose trend/seasonal components based on series length
    seasonal   = None
    seasonal_p = None
    if len(series) >= 24:
        seasonal   = "add"
        seasonal_p = 12

    model = ExponentialSmoothing(
        series,
        trend=("add" if len(series) >= 10 else None),
        seasonal=seasonal,
        seasonal_periods=seasonal_p,
        initialization_method="estimated",
    )
    fitted = model.fit(optimized=True, remove_bias=True)

    # ── Forecast
    fc_values = fitted.forecast(steps)
    forecast  = list(fc_values.astype(float))

    # ── Confidence interval: ±1.96 * residual std
    residuals = fitted.resid.dropna()
    resid_std = float(residuals.std()) if len(residuals) > 1 else 0.0
    z         = 1.96
    lower_ci  = [f - z * resid_std for f in forecast]
    upper_ci  = [f + z * resid_std for f in forecast]

    # ── AIC
    aic = float(fitted.aic) if hasattr(fitted, "aic") else 0.0

    # ── Residual diagnostics
    residual_mean = float(residuals.mean()) if len(residuals) > 0 else 0.0

    # Ljung-Box test (graceful fallback if statsmodels version differs)
    lb_pvalue = 0.5
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
        lb_pvalue = float(lb_result["lb_pvalue"].iloc[0])
    except Exception:
        pass

    return forecast, lower_ci, upper_ci, aic, residual_mean, lb_pvalue
