import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------
# Load Dataset
# ---------------------------------------
df = pd.read_csv("../data/sample_data.csv")

df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)
df = df.sort_index()

series = df["value"].ffill().dropna()


# ---------------------------------------
# Train-Test Split (80-20)
# ---------------------------------------
split = int(len(series) * 0.8)

train = series[:split]
test = series[split:]


# ---------------------------------------
# Train ARIMA Model
# ---------------------------------------
model = ARIMA(train, order=(2, 1, 2))
model_fit = model.fit()


# ---------------------------------------
# Forecast Test Period
# ---------------------------------------
forecast = model_fit.forecast(steps=len(test))


# ---------------------------------------
# Evaluation Metrics
# ---------------------------------------
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))

print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))


# ---------------------------------------
# Plot Results
# ---------------------------------------
plt.figure(figsize=(10, 5))

plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, forecast, label="Predicted")

plt.title("ARIMA Forecast vs Actual")
plt.legend()
plt.tight_layout()
plt.show()


# ---------------------------------------
# Future 7-Day Forecast
# ---------------------------------------
future_steps = 7
future_forecast = model_fit.forecast(steps=future_steps)

print("\nNext 7-Day Forecast:")
print(future_forecast)


# ---------------------------------------
# Save Model
# ---------------------------------------
joblib.dump(model_fit, "arima_model.pkl")

print("\nModel trained and saved successfully!")