import glob
import pandas as pd

# Import files into dataframe
df = pd.concat([pd.read_csv(file) for file in glob.glob('data/*.csv')], ignore_index=True)

# Look for columns to drop
print(df.info())

# Export to aggregated file
df.to_csv('data/aggregated_data.csv', index=True, header=True)