import numpy as np
import pandas as pd
import math

from model.strategy.strategy import simulate_strategy
from model.backtest.backtest import rolling_backtest
from model.forecasting.arima_model import run_arima


# -----------------------------
# Helper: Safe JSON conversion
# -----------------------------
def sanitize(value):
    if value is None:
        return 0.0
    if isinstance(value, (float, np.floating)):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return float(value)
    return float(value)


# -----------------------------
# MAIN MULTI-ASSET ENGINE
# -----------------------------
def run_portfolio_from_csv(file):

    df = pd.read_csv(file)

    if df.shape[1] < 2:
        return {"error": "CSV must contain multiple asset columns."}

    total_capital = 100000
    asset_results = {}
    portfolio_equity = None

    asset_columns = df.columns[1:]

    asset_vols = {}
    valid_series = {}

    # ---------------------------------
    # 1️⃣ Validate Assets + Volatility
    # ---------------------------------
    for col in asset_columns:

        series = pd.to_numeric(df[col], errors="coerce").dropna()

        if len(series) < 30:
            continue

        returns = series.pct_change().dropna()

        if len(returns) < 10:
            continue

        vol = returns.rolling(10).std().iloc[-1]

        if vol is None or vol == 0 or np.isnan(vol):
            continue

        asset_vols[col] = float(vol)
        valid_series[col] = series.reset_index(drop=True)

    if not asset_vols:
        return {"error": "No valid assets processed."}

    # ---------------------------------
    # 2️⃣ Inverse Volatility Weights
    # ---------------------------------
    inv_vol_sum = sum(1 / v for v in asset_vols.values())

    weights = {
        col: (1 / asset_vols[col]) / inv_vol_sum
        for col in asset_vols
    }

    # ---------------------------------
    # 3️⃣ Run Strategy Per Asset
    # ---------------------------------
    for col, series in valid_series.items():

        bt_rmse, bt_mape, bt_dir, rolling_preds = rolling_backtest(
            series, run_arima
        )

        if rolling_preds is None or len(rolling_preds) == 0:
            continue

        split = int(len(series) * 0.8)

        # full HOLD signal array
        signals = ["HOLD"] * (len(series) - 1)

        max_len = min(len(rolling_preds), len(series) - split - 1)

        for i in range(max_len):

            current_price = series.iloc[split + i]
            predicted_price = rolling_preds[i]

            if predicted_price > current_price:
                signals[split + i] = "BUY"
            else:
                signals[split + i] = "SELL"

        result = simulate_strategy(series, signals)

        raw_equity = np.array(result["equity_curve"])

        if len(raw_equity) == 0:
            continue

        # Normalize equity to 1
        normalized_equity = raw_equity / raw_equity[0]

        allocated_capital = total_capital * weights[col]

        scaled_equity = normalized_equity * allocated_capital

        asset_results[col] = {
            "weight": round(weights[col], 3),
            "total_return": sanitize(result["total_return"]),
            "sharpe_ratio": sanitize(result["sharpe_ratio"]),
            "max_drawdown": sanitize(result["max_drawdown"])
        }

        if portfolio_equity is None:
            portfolio_equity = scaled_equity
        else:
            min_len = min(len(portfolio_equity), len(scaled_equity))
            portfolio_equity = (
                portfolio_equity[:min_len] +
                scaled_equity[:min_len]
            )

    if portfolio_equity is None:
        return {"error": "No valid assets processed."}

    # ---------------------------------
    # 4️⃣ Portfolio Metrics
    # ---------------------------------
    portfolio_total_return = (
        (portfolio_equity[-1] - total_capital) / total_capital
    ) * 100

    portfolio_returns = np.diff(portfolio_equity) / portfolio_equity[:-1]

    if len(portfolio_returns) > 1 and np.std(portfolio_returns) != 0:
        portfolio_sharpe = (
            np.mean(portfolio_returns) /
            np.std(portfolio_returns)
        ) * np.sqrt(252)
    else:
        portfolio_sharpe = 0

    portfolio_max_drawdown = (
        np.min(
            portfolio_equity /
            np.maximum.accumulate(portfolio_equity) - 1
        ) * 100
    )

    clean_equity = [sanitize(x) for x in portfolio_equity]

    return {
        "assets": asset_results,
        "portfolio_total_return": sanitize(portfolio_total_return),
        "portfolio_sharpe": sanitize(portfolio_sharpe),
        "portfolio_max_drawdown": sanitize(portfolio_max_drawdown),
        "portfolio_equity_curve": clean_equity
    }