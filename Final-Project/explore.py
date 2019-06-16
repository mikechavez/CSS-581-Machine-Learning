import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/aggregated_data.csv')

print(df.info())

k = ['traditional_1', 'traditional_2', 'traditional_3', 'traditional_4', 'social_1', 'social_2', 'social_3', 'fake_followers', 'genuine']

# Plot distribution of users for each dataset
sns.countplot(data=df, x=df.dataset, order=k)
plt.xticks(rotation=45)
plt.show()

df = df.groupby('dataset').mean().reset_index()

# Plot mean status count for each dataset
sns.barplot(x=df.dataset, y=df.statuses_count, data=df, order=k).set_title('Mean Status Count')
plt.xticks(rotation=45)
plt.show()

# Plot mean listed count for each dataset
sns.barplot(x=df.dataset, y=df.listed_count, data=df, order=k).set_title('Mean Listed Count')
plt.xticks(rotation=45)
plt.show()

# Plot mean follower count for each dataset
sns.barplot(x=df.dataset, y=df.followers_count, data=df, order=k).set_title('Mean Follower Count')
plt.xticks(rotation=45)
plt.show()

# Plot mean friend count for each dataset
sns.barplot(x=df.dataset, y=df.friends_count, data=df, order=k).set_title('Mean Friend Count')
plt.xticks(rotation=45)
plt.show()

# Plot mean favorite count for each dataset
sns.barplot(x=df.dataset, y=df.favourites_count, data=df, order=k).set_title('Mean Favorite Count')
plt.xticks(rotation=45)
plt.show()

# Plot mean has_background_image count for each dataset
sns.barplot(x=df.dataset, y=df.profile_use_background_image, data=df, order=k).set_title('Mean profile_use_background_image Count')
plt.xticks(rotation=45)
plt.show()