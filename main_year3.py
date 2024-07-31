import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import numpy as np


# first price = open
# last price = close
# denominate into two dfs 

def debugIloc(x):
    print(x)
    print(x.iloc[-1])
    print(x.iloc[0])
    print((x.iloc[-1] / x.iloc[0]))
    return (x.iloc[-1] / x.iloc[0]) - 1


# today - if close > open - bychka, open > close - medvezhka - add % as avg % between 3 days - ATR 
# tomorrow/week - detect golden of death cross, and if 50 > 200 - bychkam 200 > 50 - medvezhka и + ATR % to days
# Identify Golden Cross and Death Cross events
# df['Golden_Cross'] = ((df['50_MA'] > df['200_MA']) & 
#                             (df['50_MA'].shift(1) <= df['200_MA'].shift(1)))
# df['Death_Cross'] = ((df['50_MA'] < df['200_MA']) & 
#                            (df['50_MA'].shift(1) >= df['200_MA'].shift(1)))
# 
# year - arima??

import csv
from itertools import dropwhile, takewhile

# Load the dataset
file_path = './ethereum.csv'  # Update with your file path


df = pd.DataFrame(pd.read_csv(file_path))
# Convert the date column to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

df = df.groupby(df['timestamp'].dt.date).agg({'price': 'mean'}).reset_index()
df.columns = ['date', 'price']

df['date'] = pd.to_datetime(df['date'])

comparison_date = pd.to_datetime('2018-01-01')
df = df[df['date'] >= comparison_date]

# Set the date as the index
df.set_index('date', inplace=True, drop=False)
# # Sort the data by date
df.sort_index(inplace=True)
df['year'] = pd.DatetimeIndex(df['date']).year

print(df.groupby('year').agg({'price': 'mean'}))
# year_df = df.groupby('year')
# for key, item in year_df:
#     print(year_df.get_group(key)['price'], "\n\n")

annual_growth_rates = df.groupby('year')['price'].apply(debugIloc)
print(annual_growth_rates)
# Average annual growth rates
average_annual_growth_rate = 0.07
growth_rate_realistic = average_annual_growth_rate  # Adjust based on analysis
# growth_rate_optimistic = growth_rate_realistic * 1.2  # 5% higher for optimistic scenario
# growth_rate_pessimistic = growth_rate_realistic / 1.4 # 3% lower for pessimistic scenario

# print(growth_rate_realistic, growth_rate_optimistic, growth_rate_pessimistic)
# Annual growth rates for different scenarios
# growth_rate_realistic = 0.05  # 5% annual growth
# growth_rate_optimistic = 0.10  # 10% annual growth
# growth_rate_pessimistic = 0.02  # 2% annual growth

# Calculate compounded monthly growth rates from annual growth rates
monthly_growth_realistic = (1 + growth_rate_realistic) ** (1 / 12) - 1
# monthly_growth_optimistic = (1 + growth_rate_optimistic) ** (1 / 12) - 1
# monthly_growth_pessimistic = (1 + growth_rate_pessimistic) ** (1 / 12) - 1

dates_until_2050 = pd.date_range(start='2024-08-01', end='2050-12-31', freq='M')
print(monthly_growth_realistic)
# Initialize variables from the last known values in df
last_average_price = df['price'].iloc[-1]

# Initialize lists to store the predicted prices
predicted_prices = []
optimistic_prices = []
pessimistic_prices = []

for date in dates_until_2050:
    predicted_price = last_average_price * (1 + monthly_growth_realistic)
    # optimistic_price = predicted_price * (1 + monthly_growth_optimistic)
    # pessimistic_price = predicted_price * (1 + monthly_growth_pessimistic)

    # print(date, predicted_price)
    
    predicted_prices.append(predicted_price)
    # optimistic_prices.append(optimistic_price)
    # pessimistic_prices.append(pessimistic_price)
    
    # Update last_average_price for the next iteration
    last_average_price = predicted_price

# Create a DataFrame to store the results
prediction_df = pd.DataFrame({
    'date': dates_until_2050.to_series().dt.date,
    'predicted_price': predicted_prices,
    # 'optimistic_price': optimistic_prices,
    # 'pessimistic_price': pessimistic_prices
})


# print(df)
prediction_df.set_index('date', inplace=True, drop=False)
prediction_df.sort_index(inplace=True)
# print(prediction_df)

# df['optimistic_price'] = df['price']
df['predicted_price'] = df['price']
# df['pessimistic_price'] = df['price']

res_df = pd.concat([df, prediction_df])
# Set the date as the index
res_df['date'] = pd.to_datetime(res_df['date'])
res_df.set_index('date', inplace=True, drop=False)

print(res_df)

# # Sort the data by date
res_df.sort_index(inplace=True)
# Visualize the Golden Cross and Death Cross events
plt.figure(figsize=(14, 7))
plt.plot(res_df['predicted_price'], label='Most likely Price')
# plt.plot(res_df['optimistic_price'], label='Optimistic Price', linestyle='--')
# plt.plot(res_df['pessimistic_price'], label='Pessimistic Price', linestyle='-.')
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
