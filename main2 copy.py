import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

import csv
from itertools import dropwhile, takewhile

# Load the dataset
file_path = './BTC-USD.csv'  # Update with your file path
file_path = './ethereum.csv'  # Update with your file path


df = pd.DataFrame(pd.read_csv(file_path))
print('read df', df)
# Convert the date column to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

print('before group', df)
df = df.groupby(df['timestamp'].dt.date).agg({'price': 'mean'}).reset_index()
df.columns = ['date', 'price']

print('after group', df)


# Set the date as the index
df.set_index('date', inplace=True, drop=False)

# # Sort the data by date
df.sort_index(inplace=True)

# Visualize the closing prices over time
# plt.figure(figsize=(14, 7))
# plt.plot(df['price'], label='BTC Closing Price')
# plt.title('Bitcoin Daily Closing Prices')
# plt.xlabel('Date')
# plt.ylabel('Price (USD)')
# plt.legend()
# plt.grid(True)
# plt.show()
print('after index', df)



# Perform first-order differencing
df['price_diff'] = df['price'].diff()

# Drop the NaN value that results from differencing
df_diff = df.dropna()


# first price = open
# last price = close
# denominate into two dfs 



# today - if close > open - bychka, open > close - medvezhka - add % as avg % between 3 days - ATR 
# tomorrow/week - detect golden of death cross, and if 50 > 200 - bychkam 200 > 50 - medvezhka и + ATR % to days
# Identify Golden Cross and Death Cross events
# df['Golden_Cross'] = ((df['50_MA'] > df['200_MA']) & 
#                             (df['50_MA'].shift(1) <= df['200_MA'].shift(1)))
# df['Death_Cross'] = ((df['50_MA'] < df['200_MA']) & 
#                            (df['50_MA'].shift(1) >= df['200_MA'].shift(1)))
# 
# year - arima??

# Plot the differenced data
# plt.figure(figsize=(14, 7))
# plt.plot(df_diff['price_diff'], label='Differenced BTC Closing Price')
# plt.title('Differenced Bitcoin Daily Closing Prices')
# plt.xlabel('Date')
# plt.ylabel('Differenced Price (USD)')
# plt.legend()
# plt.grid(True)
# plt.show()


# # Perform the ADF test on the differenced data
# def adf_test(series):
#     result = adfuller(series, autolag='AIC')
#     print('ADF Statistic:', result[0])
#     print('p-value:', result[1])
#     for key, value in result[4].items():
#         print(f'Critical Value ({key}): {value}')

# adf_test(df_diff['price_diff'])


df['high'] = df['price']
df['low'] = df['price']
df['close'] = df['price']

df['previous_close'] = df['close'].shift(1)
df['tr1'] = df['high'] - df['low']
df['tr2'] = abs(df['high'] - df['previous_close'])
df['tr3'] = abs(df['low'] - df['previous_close'])

df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

# Calculate ATR over a 3-day period for simplicity
atr_period = 14
df['atr'] = df['true_range'].rolling(window=atr_period, min_periods=1).mean()

# Predict potential price movement using ATR
df['upper_bound'] = df['price'] + df['atr']
df['lower_bound'] = df['price'] - df['atr']


short_window = 50
long_window = 200

# # Calculate the short-term and long-term moving averages
df['short_mavg'] = df['price'].rolling(window=50).mean()
df['long_mavg'] = df['price'].rolling(window=200).mean()
# Visualize the moving averages along with the closing prices
plt.figure(figsize=(14, 7))
plt.plot(df['price'], label='BTC Closing Price')
plt.plot(df['short_mavg'], label='50-Day Moving Average', linestyle='--')
plt.plot(df['long_mavg'], label='200-Day Moving Average', linestyle='--')
plt.title('Bitcoin Price with 50-Day and 200-Day Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
# Identify Golden Cross and Death Cross
df['signal'] = 0
df['signal'][short_window:] = np.where(df['short_mavg'][short_window:] > df['long_mavg'][short_window:], 1, -1)
df['position'] = df['signal'].diff()

last_average_price = df['price'].iloc[-1]
last_atr = df['atr'].iloc[-1]
trend = df['signal'].iloc[-1]

predicted_prices = []
optimistic_prices = []
pessimistic_prices = []
last_atrs = []


for day in range(1, 11):
    if trend == 1:  # Bullish trend
        predicted_price = last_average_price + (day * last_atr * 0.1)  # Incremental increase
    else:  # Bearish trend
        predicted_price = last_average_price - (day * last_atr * 0.1)  # Incremental decrease
    
    optimistic_price = predicted_price + last_atr  # Adding ATR for optimistic scenario
    pessimistic_price = predicted_price - last_atr  # Subtracting ATR for pessimistic scenario
    
    predicted_prices.append(predicted_price)
    optimistic_prices.append(optimistic_price)
    pessimistic_prices.append(pessimistic_price)
    
    last_average_price = predicted_price
    last_atr = (last_atr * (atr_period - 1) + abs(predicted_price - last_average_price)) / atr_period
    last_atrs.append(last_atr)


prediction_df = pd.DataFrame({
    'date': pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=10, freq='D').to_series().dt.date,
    'predicted_price': predicted_prices,
    'optimistic_price': optimistic_prices,
    'pessimistic_price': pessimistic_prices,
    'atr': last_atrs
})
prediction_df.set_index('date', inplace=True, drop=False)
prediction_df.sort_index(inplace=True)
print(prediction_df)

df['optimistic_price'] = df['price']
df['predicted_price'] = df['price']
df['pessimistic_price'] = df['price']

res_df = pd.concat([df, prediction_df]).tail(90)
print(res_df)
# Set the date as the index
res_df.set_index('date', inplace=True, drop=False)

# # Sort the data by date
res_df.sort_index(inplace=True)
# Visualize the Golden Cross and Death Cross events
plt.figure(figsize=(14, 7))
plt.plot(res_df['predicted_price'], label='Most likely Price')
plt.plot(res_df['optimistic_price'], label='Optimistic Price', linestyle='--')
plt.plot(res_df['pessimistic_price'], label='Pessimistic Price', linestyle='-.')
plt.plot(res_df['short_mavg'], label='50-Day Moving Average', linestyle=':')
plt.plot(res_df['long_mavg'], label='200-Day Moving Average', linestyle='-')
# plt.plot(res_df['atr'], label='ATR', linestyle='dotted')
plt.title('Ethereum Price predicted price for 10 days')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

exit()
# Identify Golden Cross and Death Cross events
df['Golden_Cross'] = ((df['50_MA'] > df['200_MA']) & 
                            (df['50_MA'].shift(1) <= df['200_MA'].shift(1)))
df['Death_Cross'] = ((df['50_MA'] < df['200_MA']) & 
                           (df['50_MA'].shift(1) >= df['200_MA'].shift(1)))

# Visualize the Golden Cross and Death Cross events
plt.figure(figsize=(14, 7))
plt.plot(df['price'], label='BTC Closing Price')
plt.plot(df['50_MA'], label='50-Day Moving Average', linestyle='--')
plt.plot(df['200_MA'], label='200-Day Moving Average', linestyle='--')
plt.plot(df[df['Golden_Cross']].index, 
         df['50_MA'][df['Golden_Cross']], 
         '^', markersize=10, color='g', label='Golden Cross')
plt.plot(df[df['Death_Cross']].index, 
         df['50_MA'][df['Death_Cross']], 
         'v', markersize=10, color='r', label='Death Cross')
plt.title('Bitcoin Price with Golden Cross and Death Cross')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()


# Create a simple trading strategy based on Golden Cross and Death Cross
df['Signal'] = 0
df.loc[df['Golden_Cross'], 'Signal'] = 1
df.loc[df['Death_Cross'], 'Signal'] = -1

# Calculate strategy returns
df['Strategy_Returns'] = df['close'].pct_change() * df['Signal'].shift(1)

# Plot cumulative returns
df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()

plt.figure(figsize=(14, 7))
plt.plot(df['Cumulative_Strategy_Returns'], label='Strategy Returns')
plt.title('Cumulative Returns of the Golden Cross Strategy')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()
exit()

# Fit the ARIMA model
model = ARIMA(df['close'], order=(200, 1, 0))  # Adjust p, d, q parameters as needed
model_fit = model.fit()


# Summary of the model
print(model_fit.summary())



# Forecasting future prices
# forecast_steps = (2028 - 2024) * 365  # Number of days from now to the end of 2025
forecast_steps = 365  # Number of days from now to the end of 2025
# Use the model to make predictions
forecast = model_fit.forecast(steps=forecast_steps)
print(forecast)

# Plot the forecast
plt.figure(figsize=(14, 7))
plt.plot(df['close'], label='Historical Prices')
plt.plot(pd.date_range(start=df.index[-1], periods=forecast_steps, freq='D'), forecast, label='Forecasted Prices')
plt.title('Bitcoin Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
