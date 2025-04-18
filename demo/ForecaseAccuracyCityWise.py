import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

district = "Ahmedabad"  # Change this to test other districts

# Load actual and forecast data
df_actual = pd.read_csv("gujarat_cost_of_living_full_dataset.csv")
df_forecast = pd.read_csv(f"forecast_{district}.csv")

# Filter and align data
actual = df_actual[df_actual["District"] == district].copy()
actual['Date'] = pd.to_datetime(actual['Date'])
forecast = df_forecast[['ds', 'yhat']].copy()
forecast['ds'] = pd.to_datetime(forecast['ds'])

# Join and evaluate
merged = pd.merge(actual, forecast, left_on="Date", right_on="ds", how="inner")
y_true = merged["Total_Monthly_Expense"]
y_pred = merged["yhat"]

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"Forecast Accuracy for {district}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
