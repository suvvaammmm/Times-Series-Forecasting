import numpy as np


def calculate_volatility(series, window=5):
    returns = series.pct_change()
    vol = returns.rolling(window).std()
    return vol


def volatility_position_size(capital, volatility, base_risk=0.02):
    if volatility is None or volatility == 0:
        return capital * base_risk

    adjusted_risk = base_risk / volatility
    adjusted_risk = min(adjusted_risk, 0.05)  # cap max exposure at 5%

    return capital * adjusted_risk


def check_drawdown_stop(equity_curve, max_drawdown_limit=-0.15):
    equity = np.array(equity_curve)
    drawdowns = equity / np.maximum.accumulate(equity) - 1
    if np.min(drawdowns) < max_drawdown_limit:
        return True
    return False


def check_kill_switch(consecutive_losses, max_losses=5):
    return consecutive_losses >= max_losses