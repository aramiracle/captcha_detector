import pandas as pd

df = pd.read_csv('dataset.csv')

print(df['image']['bytes'])
print(df.iloc[0, 0]['text'])