import glob
import pandas as pd

df = pd.concat([pd.read_csv(file) for file in glob.glob('data/train_groundtruth/*.csv')], ignore_index=True)

csv_file = df.to_csv(r'./training_groundtruth.csv', index=None, header=True)


