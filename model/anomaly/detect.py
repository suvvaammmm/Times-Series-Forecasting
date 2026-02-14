import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def detect_anomalies(df, threshold_multiplier):

    values = df["value"].values

    model = ARIMA(values, order=(1, 0, 0))
    model_fit = model.fit()

    predictions = model_fit.predict(start=0, end=len(values)-1)

    residuals = values - predictions

    threshold = float(threshold_multiplier) * np.std(residuals)

    anomalies = np.abs(residuals) > threshold

    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))

    return (
        predictions.tolist(),
        anomalies.tolist(),
        float(mae),
        float(rmse)
    )
