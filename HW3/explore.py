import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc("font", size=14)
import seaborn as sns

sns.set()

df = pd.read_csv('training_with_missing.csv')
df2 = pd.read_csv('training_groundtruth.csv')

print(df.info())
print(df.isnull().sum())
print(df2.isnull().sum())



# df.dropna(axis=0, how='any', inplace=True)




# df_log = pd.DataFrame(df, columns=['PNA'])
# df_log['PNA'] = np.log2(df_log['PNA'])
#
# sns.distplot(df_log.PNA, hist=True)
# plt.title('PBUN')
# plt.show()


# sns.distplot(df.PCL, hist=True)
# plt.title('PCL')
# plt.show()

# sns.distplot(df.PK, hist=True)
# plt.title('PK')
# plt.show()

# sns.distplot(df.PLCO2, hist=True)
# plt.title('PLCO2')
# plt.show()

# sns.distplot(df.PNA, hist=True)
# plt.title('PNA')
# plt.show()

# sns.distplot(df.HCT, hist=True)
# plt.title('HCT')
# plt.show()

# sns.distplot(df.HCT, hist=True)
# plt.title('HCT')
# plt.show()

# sns.distplot(df.HGB, hist=True)
# plt.title('HGB')
# plt.show()

# sns.distplot(df.MCV, hist=True)
# plt.title('MCV')
# plt.show()

# sns.distplot(df.WBC, hist=True)
# plt.title('WBC')
# plt.show()


# sns.distplot(df.RDW, hist=True)
# plt.title('RDW')
# plt.show()

# sns.distplot(df_log10.PBUN, hist=True)
# plt.title('PBUN')
# plt.show()

# sns.distplot(df.PCRE, hist=True)
# plt.title('PCRE')
# plt.show()
#
# sns.distplot(df.PGLU, hist=True)
# plt.title('PGLU')
# plt.show()

