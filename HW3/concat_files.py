import glob
import pandas as pd

df = pd.concat([pd.read_csv(file) for file in glob.glob('data/train_with_missing/*.csv')], ignore_index=True)
df = pd.concat([pd.read_csv(file) for file in glob.glob('data/train_groundtruth/*.csv')], ignore_index=True)

csv_file = df.to_csv(r'./training_with_missing.csv', index=None, header=True)
csv_file = df.to_csv(r'./training_groundtruth.csv', index=None, header=True)


