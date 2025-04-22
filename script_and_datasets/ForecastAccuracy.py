import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import os

# Load dataset
df = pd.read_csv('gujarat_cost_of_living_full_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Forecast accuracy results
results = []

# Loop through each district and evaluate
for district in df['District'].unique():
    print(f"Evaluating forecast accuracy for: {district}")
    district_df = df[df['District'] == district][['Date', 'Total_Monthly_Expense']].copy()
    district_df.rename(columns={'Date': 'ds', 'Total_Monthly_Expense': 'y'}, inplace=True)
    
    # Sort and split into train/test (80/20)
    district_df.sort_values('ds', inplace=True)
    split_index = int(len(district_df) * 0.8)
    train_df = district_df[:split_index]
    test_df = district_df[split_index:]
    
    # Fit model
    model = Prophet()
    model.fit(train_df)
    
    # Create future dataframe for test period
    future = model.make_future_dataframe(periods=len(test_df), freq='MS')
    forecast = model.predict(future)
    
    # Extract predicted values for test dates
    forecast_filtered = forecast[['ds', 'yhat']].set_index('ds').loc[test_df['ds']]
    
    # Merge with actual
    merged = test_df.set_index('ds').join(forecast_filtered)
    y_true = merged['y']
    y_pred = merged['yhat']
    
    # Calculate MAPE
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    results.append({'District': district, 'MAPE (%)': round(mape, 2)})

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('forecast_accuracy_results.csv', index=False)
print("\nForecast accuracy (MAPE) saved to 'forecast_accuracy_results.csv'.")
