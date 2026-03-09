import numpy as np
import pandas as pd
import math
from statsmodels.tsa.arima.model import ARIMA


def sanitize(value):
    if value is None:
        return 0.0
    if isinstance(value, (float, np.floating)):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return float(value)
    return float(value)


def run_portfolio_from_csv(file):

    # ---------------------------------
    # Load Data
    # ---------------------------------
    df = pd.read_csv(file)

    if df.shape[1] < 2:
        return {"error": "CSV must contain multiple asset columns."}

    total_capital = 100000
    asset_columns = df.columns[1:]

    df[asset_columns] = df[asset_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    df = df.dropna()

    if len(df) < 30:
        return {"error": "Not enough data points."}

    # ---------------------------------
    # Compute Returns
    # ---------------------------------
    returns_df = df[asset_columns].pct_change().dropna()

    if returns_df.empty:
        return {"error": "Return matrix empty."}

    # ---------------------------------
    # Inverse Volatility Weights
    # ---------------------------------
    vol = returns_df.std().replace(0, 1e-6)
    inv_vol = 1 / vol
    weights = inv_vol / inv_vol.sum()

    # ---------------------------------
    # Portfolio Historical Returns
    # ---------------------------------
    portfolio_returns = returns_df.dot(weights)

    portfolio_equity = total_capital * (1 + portfolio_returns).cumprod()

    # ---------------------------------
    # RETURN-BASED FORECAST
    # ---------------------------------
    forecast_horizon = 7
    portfolio_future_returns = np.zeros(forecast_horizon)

    for col in asset_columns:

        series = df[col].astype(float)
        asset_returns = series.pct_change().dropna()

        if len(asset_returns) < 20:
            continue

        try:
            model = ARIMA(asset_returns, order=(1, 0, 1))
            result = model.fit()

            forecast_ret = result.forecast(steps=forecast_horizon)

            portfolio_future_returns += forecast_ret.values * weights[col]

        except:
            continue

    # Extend Equity Forward
    initial_equity = float(portfolio_equity.iloc[0])
    last_equity = float(portfolio_equity.iloc[-1])

    future_equity = []

    for r in portfolio_future_returns:
        last_equity = last_equity * (1 + r)
        future_equity.append(last_equity)

    # ---------------------------------
    # Portfolio Metrics
    # ---------------------------------
    portfolio_total_return = (
        (portfolio_equity.iloc[-1] - portfolio_equity.iloc[0])
        / portfolio_equity.iloc[0]
    ) * 100

    portfolio_sharpe = 0
    if portfolio_returns.std() != 0:
        portfolio_sharpe = (
            portfolio_returns.mean() /
            portfolio_returns.std()
        ) * np.sqrt(252)

    portfolio_max_drawdown = (
        (portfolio_equity /
         portfolio_equity.cummax() - 1).min()
    ) * 100

    # ---------------------------------
    # Individual Asset Metrics
    # ---------------------------------
    asset_results = {}

    for col in asset_columns:

        asset_ret = returns_df[col]
        asset_equity = total_capital * (1 + asset_ret).cumprod()

        asset_total_return = (
            (asset_equity.iloc[-1] - asset_equity.iloc[0])
            / asset_equity.iloc[0]
        ) * 100

        asset_sharpe = 0
        if asset_ret.std() != 0:
            asset_sharpe = (
                asset_ret.mean() /
                asset_ret.std()
            ) * np.sqrt(252)

        asset_max_dd = (
            (asset_equity /
             asset_equity.cummax() - 1).min()
        ) * 100

        asset_results[col] = {
            "weight": round(float(weights[col]), 3),
            "total_return": round(float(asset_total_return), 2),
            "sharpe_ratio": round(float(asset_sharpe), 2),
            "max_drawdown": round(float(asset_max_dd), 2)
        }

    # ---------------------------------
    # Normalize Historical Equity
    # ---------------------------------
    normalized_portfolio_equity = (
        portfolio_equity / initial_equity
    ) * 100

    clean_equity = [
        round(float(x), 2)
        for x in normalized_portfolio_equity.values
    ]

    # Normalize Forecast Equity
    normalized_future_equity = [
        round(float((x / initial_equity) * 100), 2)
        for x in future_equity
    ]

    # ---------------------------------
    # Final JSON Return
    # ---------------------------------
    return {
        "assets": asset_results,
        "weights": {
            str(k): float(round(v * 100, 2))
            for k, v in weights.items()
        },
        "portfolio_total_return": float(round(portfolio_total_return, 2)),
        "portfolio_sharpe": float(round(portfolio_sharpe, 2)),
        "portfolio_max_drawdown": float(round(portfolio_max_drawdown, 2)),
        "portfolio_equity_curve": clean_equity,
        "portfolio_forecast_curve": normalized_future_equity
    }