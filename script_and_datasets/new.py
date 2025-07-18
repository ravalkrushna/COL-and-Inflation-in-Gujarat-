import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import time
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('gujarat_cost_of_living_full_dataset.csv')

# Ensure expected columns exist
expected_columns = ['District', 'Date', 'Rent', 'Utilities', 'Grocery', 'Transportation', 'Healthcare', 'Education']
missing_cols = [col for col in expected_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing expected expense columns in dataset: {missing_cols}")

# Calculate total expense
df['Total_Monthly_Expense'] = df[['Rent', 'Utilities', 'Grocery', 'Transportation', 'Healthcare', 'Education']].sum(axis=1)

# --- Forecasting Functions with Optimizers and Time Logging ---

def forecast_prophet(district_df):
    start = time.time()
    prophet_df = district_df.rename(columns={'Date': 'ds', 'Total_Monthly_Expense': 'y'})
    m = Prophet()
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=12, freq='M')
    forecast = m.predict(future)
    end = time.time()
    print(f"⏱ Prophet Time Taken: {end - start:.2f}s | Space: O(n) | Time Complexity: ~O(n log n)")
    return forecast['yhat'][-12:].values

def forecast_arima(district_df):
    start = time.time()
    y = district_df['Total_Monthly_Expense'].values
    train = y[:-12]
    test = y[-12:]

    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=12)
    end = time.time()
    print(f"⏱ ARIMA Time Taken: {end - start:.2f}s | Space: O(n) | Time Complexity: ~O(n²)")
    return forecast, test

def forecast_linear(district_df):
    start = time.time()
    df_copy = district_df.copy()
    df_copy['Date_ordinal'] = pd.to_datetime(df_copy['Date']).map(pd.Timestamp.toordinal)

    X = df_copy['Date_ordinal'].values.reshape(-1, 1)
    y = df_copy['Total_Monthly_Expense'].values

    model = LinearRegression()
    model.fit(X, y)

    future_dates = pd.date_range(df_copy['Date'].iloc[-1], periods=13, freq='M')[1:]
    future_ordinals = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    forecast = model.predict(future_ordinals)
    end = time.time()
    print(f"⏱ Linear Regression Time Taken: {end - start:.2f}s | Space: O(n) | Time Complexity: ~O(n)")
    return forecast, y[-12:]

def forecast_lstm(district_df):
    start = time.time()
    data = district_df['Total_Monthly_Expense'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(data_scaled) - 12):
        X.append(data_scaled[i:i+12])
        y.append(data_scaled[i+12])
    X, y = np.array(X), np.array(y)

    # LSTM with Adam optimizer and EarlyStopping
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(12, 1)),
        Dense(1)
    ])
    adam_optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=adam_optimizer, loss='mse')
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    model.fit(X, y, epochs=50, verbose=0, callbacks=[early_stop])

    last_input = data_scaled[-12:].reshape(1, 12, 1)
    forecast = []
    for _ in range(12):
        pred = model.predict(last_input, verbose=0)[0][0]
        forecast.append(pred)
        last_input = np.append(last_input[:, 1:, :], [[[pred]]], axis=1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    end = time.time()
    print(f"⏱ LSTM Time Taken: {end - start:.2f}s | Space: High | Time Complexity: O(n·epochs·timesteps·units)")
    return forecast, data[-12:].flatten()

# --- Evaluation & Comparison with RMSE & DAA Summary ---

def compare_models(district):
    print(f"\n📍 Forecasting for District: {district}")
    district_df = df[df['District'] == district][['Date', 'Total_Monthly_Expense']].copy()

    prophet_forecast = forecast_prophet(district_df)
    actual_prophet = district_df['Total_Monthly_Expense'].values[-12:]

    arima_forecast, arima_actual = forecast_arima(district_df)
    linear_forecast, linear_actual = forecast_linear(district_df)
    lstm_forecast, lstm_actual = forecast_lstm(district_df)

    prophet_rmse = np.sqrt(mean_squared_error(actual_prophet, prophet_forecast))
    arima_rmse = np.sqrt(mean_squared_error(arima_actual, arima_forecast))
    linear_rmse = np.sqrt(mean_squared_error(linear_actual, linear_forecast))
    lstm_rmse = np.sqrt(mean_squared_error(lstm_actual, lstm_forecast))

    print("🔍 RMSE Scores:")
    print(f"Prophet:           {prophet_rmse:.2f}")
    print(f"ARIMA:             {arima_rmse:.2f}")
    print(f"Linear Regression: {linear_rmse:.2f}")
    print(f"LSTM (Adam):       {lstm_rmse:.2f}")

    # Optional: plot comparison
    labels = ['Prophet', 'ARIMA', 'LR', 'LSTM']
    scores = [prophet_rmse, arima_rmse, linear_rmse, lstm_rmse]
    plt.bar(labels, scores, color='skyblue')
    plt.ylabel('RMSE (Lower = Better)')
    plt.title(f'Model Accuracy Comparison for {district}')
    plt.show()

# --- Run Forecasts for All Districts ---
districts = df['District'].unique()
for district in districts:
    compare_models(district)
