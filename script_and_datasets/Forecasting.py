import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv('gujarat_cost_of_living_full_dataset.csv')

# Expected columns
expected_columns = ['District', 'Date', 'Rent', 'Utilities', 'Grocery', 'Transportation', 'Healthcare', 'Education']

# Validate column presence
missing_cols = [col for col in expected_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing expected expense columns in dataset: {missing_cols}")

# Calculate total monthly expense
df['Total_Monthly_Expense'] = df[['Rent', 'Utilities', 'Grocery', 'Transportation', 'Healthcare', 'Education']].sum(axis=1)

# Forecasting function
def forecast_district(district):
    district_df = df[df['District'] == district][['Date', 'Total_Monthly_Expense']].copy()
    district_df.rename(columns={'Date': 'ds', 'Total_Monthly_Expense': 'y'}, inplace=True)

    m = Prophet()
    m.fit(district_df)

    future = m.make_future_dataframe(periods=12, freq='M')
    forecast = m.predict(future)

    # Save forecast to CSV
    forecast_outfile = f'forecast_{district}.csv'
    forecast.to_csv(forecast_outfile, index=False)

    # Plot
    fig = m.plot(forecast)
    plt.title(f"Forecast for {district}")
    plt.show()

# Loop through all districts
districts = df['District'].unique()
for district in districts:
    print(f"Processing forecast for: {district}")
    forecast_district(district)
