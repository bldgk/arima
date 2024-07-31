import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import time

# Load the dataset
file_path = './BTC-USD.csv'  # Update with your file path
btc_data = pd.read_csv(file_path)

# Convert the date column to datetime format
btc_data['date'] = pd.to_datetime(btc_data['date'])


# Set the date as the index
btc_data.set_index('date', inplace=True)

# Sort the data by date
btc_data.sort_index(inplace=True)

# Visualize the closing prices over time
plt.figure(figsize=(14, 7))
plt.plot(btc_data['close'], label='BTC Closing Price')
plt.title('Bitcoin Daily Closing Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Perform first-order differencing
btc_data['close_diff'] = btc_data['close'].diff().dropna()

# Drop the NaN value that results from differencing
btc_data_diff = btc_data.dropna()

# Perform the ADF test on the differenced data
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print(f'Critical Value ({key}): {value}')

adf_test(btc_data_diff['close_diff'])

# Plot the differenced data
plt.figure(figsize=(14, 7))
plt.plot(btc_data_diff['close_diff'], label='Differenced BTC Closing Price')
plt.title('Differenced Bitcoin Daily Closing Prices')
plt.xlabel('Date')
plt.ylabel('Differenced Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

start = time.time()
# Fit the ARIMA model
model = ARIMA(btc_data['close'], order=(200, 1, 0))  # Adjust p, d, q parameters as needed
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())



# Forecasting future prices
# forecast_steps = (2028 - 2024) * 365  # Number of days from now to the end of 2025
forecast_steps = 90  # Number of days from now to the end of 2025
# Use the model to make predictions
forecast = model_fit.forecast(steps=forecast_steps)
print(forecast)
end = time.time()
print(end - start)

# Plot the forecast
plt.figure(figsize=(14, 7))
plt.plot(btc_data['close'], label='Historical Prices')
plt.plot(pd.date_range(start=btc_data.index[-1], periods=forecast_steps, freq='D'), forecast, label='Forecasted Prices')
plt.title('Bitcoin Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
