import pandas as pd

# Import social spambot files
s1 = pd.read_csv('data/social1.csv')
s2 = pd.read_csv('data/social2.csv')
s3 = pd.read_csv('data/social3.csv')

# Import traditional spambot files
t1 = pd.read_csv('data/traditional1.csv')
t2 = pd.read_csv('data/traditional2.csv')
t3 = pd.read_csv('data/traditional3.csv')
t4 = pd.read_csv('data/traditional4.csv')

# Import fake followers file
ff = pd.read_csv('data/fake_followers.csv')

# Import genuine user file
gu = pd.read_csv('data/genuine.csv')


def fillNA(df, col, val):
    df[col] = df[col].fillna(val)
    return df
print(s1.info())

def addColumn(df, col, val):
    df[col] = val
    return df

# Add column describing dataset
s1 = addColumn(s1, 'dataset','social_1')
s2 = addColumn(s2, 'dataset','social_2')
s3 = addColumn(s3, 'dataset','social_3')
t1 = addColumn(t1, 'dataset','traditional_1')
t2 = addColumn(t2, 'dataset','traditional_2')
t3 = addColumn(t3, 'dataset','traditional_3')
t4 = addColumn(t4, 'dataset','traditional_4')
ff = addColumn(ff, 'dataset','fake_followers')
gu = addColumn(gu, 'dataset','genuine_users')

data = [s1, s2, s3, t1, t2, t3, t4, ff, gu]

s1.to_csv('social_1.csv')
s2.to_csv('social_2.csv')
s3.to_csv('social_3.csv')
t1.to_csv('traditional_1.csv')
t2.to_csv('traditional_2.csv')
t3.to_csv('traditional_3.csv')
t4.to_csv('traditional_4.csv')
ff.to_csv('fake_followers.csv')
gu.to_csv('genuine.csv')

s1 = fillNA(s1, 'geo_enabled', 0)
s1 = fillNA(s1, 'default_profile', 0)
# print(s1.info())