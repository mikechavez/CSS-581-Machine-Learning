import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/aggregated_users.csv')

print(df.info())

# df = df.groupby('dataset').mean().reset_index()

print(df.info())

# user_dist.to_csv('user_dist.csv', header=True)



df.rename(columns={'Unnamed: 0' : 'user_count'})

k = ['traditional_1', 'traditional_2', 'traditional_3', 'traditional_4', 'social_1', 'social_2', 'social_3', 'fake_followers', 'genuine_users']

# Plot distribution of users for each dataset
sns.countplot(data=df, x=df.dataset, order=k)
plt.xticks(rotation=45)
plt.show()

#
# # Plot mean status count for each dataset
# sns.barplot(x=df.dataset, y=df.statuses_count, data=df, order=k).set_title('Mean Status Count')
# plt.xticks(rotation=45)
# plt.show()
#
# # Plot mean follower count for each dataset
# sns.barplot(x=df.dataset, y=df.followers_count, data=df, order=k).set_title('Mean Follower Count')
# plt.xticks(rotation=45)
# plt.show()
#
# # Plot mean follower count for each dataset
# sns.barplot(x=df.dataset, y=df.friends_count, data=df, order=k).set_title('Mean Friend Count')
# plt.xticks(rotation=45)
# plt.show()
#
# # Plot mean follower count for each dataset
# sns.barplot(x=df.dataset, y=df.favourites_count, data=df, order=k).set_title('Mean Favorite Count')
# plt.xticks(rotation=45)
# plt.show()