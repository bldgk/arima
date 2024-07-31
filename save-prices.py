import pandas as pd

import csv
from itertools import dropwhile, takewhile

# Load the dataset
file_path = './prices.csv'  # Update with your file path

for chunk in pd.read_csv(file_path, chunksize=500):
    df = pd.DataFrame(chunk, columns=['timestamp', 'price', 'volume_24h', 'coingecko'])
    df = df.query('coingecko == "ethereum"')
    print(df)
    df.to_csv('./ethereum.csv', mode='a', header=False)

