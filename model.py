
# Fit the ARIMA model
model = ARIMA(btc_data['close'], order=(p, d, q))
model_fit = model.fit(disp=0)

# Summary of the model
print(model_fit.summary())

# Forecasting future prices
forecast_steps = 365  # Number of days to forecast
forecast = model_fit.forecast(steps=forecast_steps)[0]

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
