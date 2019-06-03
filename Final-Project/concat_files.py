import glob
import pandas as pd


df = pd.concat([pd.read_csv(file) for file in glob.glob('data/*.csv')], ignore_index=True)

# specify the sequence of columns
# df. df.reindex(columns = )

print(df.info())

df.to_csv('data/aggregated_data.csv', index=None, header=True)