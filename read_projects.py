import pandas as pd

file_path = './prices.csv'  # Update with your file path

projects = set()

for chunk in pd.read_csv(file_path, chunksize=500):
    df = pd.DataFrame(chunk, columns=['timestamp', 'price', 'volume_24h', 'coingecko'])
    print(df)
    projects.update(df['coingecko'])


df = pd.DataFrame(projects)
df.to_csv('./projects.csv', mode='a', header=False)