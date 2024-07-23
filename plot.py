import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Load the dataset
file_path = './BTC-Daily.csv'
btc_data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(btc_data.head())


# Convert the date column to datetime format
# btc_data['date'] = pd.to_datetime(btc_data['date'])

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


# Function to perform the Augmented Dickey-Fuller test
def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print(f'Critical Value ({key}): {value}')

# Perform the ADF test on the closing prices
adf_test(btc_data['close'])

