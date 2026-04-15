import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox


def _build_features(series_values, window=10):
    """
    Build a feature matrix from a 1-D numpy array.
    Each row = one training sample. Features per row:
      - lag_1 … lag_window      : last `window` prices
      - rolling_mean_5          : mean of last 5 prices
      - rolling_std_5           : std  of last 5 prices
      - rolling_mean_10         : mean of last 10 prices
      - momentum_3              : price[t-1] - price[t-4]  (3-step momentum)
      - trend                   : linear index normalised to [0,1]
    """
    n = len(series_values)
    X, y = [], []

    for i in range(window, n):
        lag_features = series_values[i - window: i].tolist()

        rm5  = np.mean(series_values[max(0, i - 5): i])
        rs5  = np.std(series_values[max(0, i - 5): i]) if (i - max(0, i-5)) > 1 else 0.0
        rm10 = np.mean(series_values[max(0, i - 10): i])
        mom3 = series_values[i - 1] - series_values[i - 4] if i >= 4 else 0.0
        trend = i / n

        row = lag_features + [rm5, rs5, rm10, mom3, trend]
        X.append(row)
        y.append(series_values[i])

    return np.array(X, dtype=float), np.array(y, dtype=float)


def run_ridge(series, forecast_steps=5):

    # ── 0. Clean ──────────────────────────────────────────────────────────
    series = series.astype(float).dropna()
    values = series.values
    n = len(values)

    WINDOW = min(10, n // 3)          # adapt window to series length
    if WINDOW < 3:
        raise Exception("Ridge: series too short (need at least 9 points)")

    # ── 1. Log-transform (mirrors ARIMA treatment) ────────────────────────
    log_values = np.log(values)

    # ── 2. Build features ─────────────────────────────────────────────────
    X, y = _build_features(log_values, window=WINDOW)

    # ── 3. Train / validation split for residual estimation ───────────────
    split = max(int(len(X) * 0.8), len(X) - 20)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s   = scaler_X.transform(X_val)

    # ── 4. Fit Ridge (alpha tuned via simple CV over [0.01, 0.1, 1, 10]) ──
    best_alpha = 1.0
    best_val_mse = float("inf")

    for alpha in [0.01, 0.1, 1.0, 10.0, 50.0]:
        m = Ridge(alpha=alpha)
        m.fit(X_train_s, y_train)
        preds = m.predict(X_val_s)
        mse = np.mean((y_val - preds) ** 2)
        if mse < best_val_mse:
            best_val_mse = mse
            best_alpha = alpha

    # Refit on full data with best alpha
    X_all_s = scaler_X.fit_transform(X)
    model = Ridge(alpha=best_alpha)
    model.fit(X_all_s, y)

    # ── 5. Multi-step forecast (iterative / recursive) ────────────────────
    history = log_values.tolist()
    preds_log = []

    for step in range(forecast_steps):
        window_vals = np.array(history[-WINDOW:])
        tail5 = history[-5:] if len(history) >= 5 else history
        rm5  = np.mean(tail5)
        rs5  = np.std(tail5) if len(tail5) > 1 else 0.0
        tail10 = history[-10:] if len(history) >= 10 else history
        rm10 = np.mean(tail10)
        mom3 = history[-1] - history[-4] if len(history) >= 4 else 0.0
        trend = (n + step) / (n + forecast_steps)

        row = window_vals.tolist() + [rm5, rs5, rm10, mom3, trend]
        row_s = scaler_X.transform([row])
        pred_log = model.predict(row_s)[0]

        preds_log.append(pred_log)
        history.append(pred_log)

    # ── 6. Confidence interval (±1.96σ of val residuals, log scale) ───────
    val_preds_log = model.predict(X_val_s)
    residuals_log = y_val - val_preds_log
    std_log = np.std(residuals_log)

    preds_log = np.array(preds_log)
    lower_log = preds_log - 1.96 * std_log
    upper_log = preds_log + 1.96 * std_log

    # ── 7. Back to price scale ─────────────────────────────────────────────
    forecast = np.exp(preds_log)
    lower    = np.exp(lower_log)
    upper    = np.exp(upper_log)

    # ── 8. Diagnostics ─────────────────────────────────────────────────────
    # Residuals on full training set for Ljung-Box
    all_preds_log = model.predict(X_all_s)
    full_residuals = y - all_preds_log

    residual_mean = round(float(np.mean(full_residuals)), 4)

    lb_test   = acorr_ljungbox(full_residuals, lags=[10], return_df=True)
    lb_pvalue = round(float(lb_test["lb_pvalue"].values[0]), 4)

    # Pseudo-AIC:  n*log(MSE) + 2*k   (k = number of features + 1)
    mse = np.mean(full_residuals ** 2)
    k   = X.shape[1] + 1
    aic = round(float(len(y) * np.log(mse + 1e-9) + 2 * k), 2)

    # ── 9. Safe rounding ───────────────────────────────────────────────────
    forecast = np.round(forecast.astype(float), 2)
    lower    = np.round(lower.astype(float), 2)
    upper    = np.round(upper.astype(float), 2)

    return forecast, lower, upper, aic, residual_mean, lb_pvalue
