import pandas as pd
import joblib
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
df = pd.read_csv("../data/sample_data.csv")

# Convert date column
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Train ARIMA model
model = ARIMA(df['value'], order=(2,1,2))
model_fit = model.fit()

# Save trained model
joblib.dump(model_fit, "arima_model.pkl")

print("Model trained and saved successfully!")
