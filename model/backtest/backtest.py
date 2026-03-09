import numpy as np
import pandas as pd


def backtest(series, model_func, test_size=0.2):

    n = len(series)

    if n < 20:
        return None, None, None

    split = int(n * (1 - test_size))

    train = series.iloc[:split]
    test = series.iloc[split:]

    forecast, _, _, _ = model_func(train)

    forecast = forecast[:len(test)]

    # Align forecast with test index
    forecast = pd.Series(forecast, index=test.index)

    # -----------------------
    # METRICS
    # -----------------------

    rmse = np.sqrt(np.mean((test - forecast) ** 2))

    # Avoid divide by zero
    mape = np.mean(
        np.abs((test - forecast) / test.replace(0, np.nan))
    ) * 100

    # -----------------------
    # DIRECTION ACCURACY
    # -----------------------

    actual_direction = np.sign(test.diff().dropna())
    forecast_direction = np.sign(forecast.diff().dropna())

    direction_accuracy = (
        (actual_direction == forecast_direction).mean() * 100
    )

    return rmse, mape, direction_accuracy

def rolling_backtest(series, model_func, test_size=0.2):

    import numpy as np

    n = len(series)

    if n < 15:
        return None, None, None, None

    split = int(n * 0.7)

    predictions = []
    actuals = []

    for i in range(split, min(split + 30, n - 1)):
        train = series.iloc[:i]
        current_price = series.iloc[i]
        next_price = series.iloc[i + 1]

        try:
            forecast, _, _, _, _, _ = model_func(train)
            pred_next = forecast[0]

            predictions.append(pred_next)
            actuals.append(next_price)

        except:
            continue

    if len(predictions) == 0:
        return None, None, None, None

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    direction_accuracy = np.mean(
        np.sign(np.diff(actuals)) ==
        np.sign(np.diff(predictions))
    ) * 100

    return rmse, mape, direction_accuracy, predictions
