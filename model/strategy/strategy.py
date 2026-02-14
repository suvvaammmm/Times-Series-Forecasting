import numpy as np

def simulate_strategy(series, signals):

    transaction_cost = 0.001
    risk_per_trade = 0.02
    starting_capital = 100000
    capital = starting_capital

    returns = []
    equity = [capital]

    for i in range(len(signals)):

        if i >= len(series) - 1:
            break

        current_price = series.iloc[i]
        next_price = series.iloc[i + 1]
        signal = signals[i]

        if signal == "BUY":
            r = (next_price - current_price) / current_price
        elif signal == "SELL":
            r = (current_price - next_price) / current_price
        else:
            r = 0

        r -= transaction_cost

        # Risk-controlled position sizing
        position_size = capital * risk_per_trade
        pnl = position_size * r

        capital += pnl

        # 🚨 Prevent negative capital
        if capital < starting_capital * 0.1:
            capital = starting_capital * 0.1

        returns.append(r)
        equity.append(capital)

    returns = np.array(returns)
    equity = np.array(equity)

    total_return = ((capital - starting_capital) / starting_capital) * 100

    if len(returns) > 1 and np.std(returns) != 0:
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    win_rate = np.mean(returns > 0) * 100 if len(returns) > 0 else 0

    max_drawdown = (
        np.min(equity / np.maximum.accumulate(equity) - 1) * 100
        if len(equity) > 1 else 0
    )

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "equity_curve": equity.tolist()
    }