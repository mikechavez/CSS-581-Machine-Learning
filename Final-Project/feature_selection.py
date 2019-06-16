import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/aggregated_data.csv')

print(df.info())


# 20X Friends-to-followers ratio
df['friends_followers_ratio'] = df['friends_count']/df['followers_count']
df.replace([np.Inf, -np.Inf], 0, inplace=True)
df['20x_friends_followers'] = df['friends_followers_ratio'] >= 20
df['20x_friends_followers'] = df['20x_friends_followers'].astype(int)

# 100X Friends-to-followers ratio
df['100x_friends_followers'] = df['friends_followers_ratio'] >= 100
df['100x_friends_followers'] = df['100x_friends_followers'].astype(int)

# 2X Followers-to-listed ratio
df['followers_listed_ratio'] = df['followers_count']/df['listed_count']
df.replace([np.NaN, np.Inf], 0, inplace=True)
df['100x_followers_listed'] = df['followers_listed_ratio'] >= 100
df['100x_followers_listed'] = df['100x_followers_listed'].astype(int)

# Has been in a list
df['is_in_a_list'] = df['listed_count'] >= 1
df['is_in_a_list'] = df['is_in_a_list'].astype(int)

# Selected Language is 'En'
df['lang_is_En'] = df['lang'] == 'en'
df['lang_is_En'] = df['lang_is_En'].astype(int)

# Has never favorited a tweet
df['has_never_favorited'] = df['favourites_count'] == 0
df['has_never_favorited'] = df['has_never_favorited'].astype(int)

# 5X Followers-to-tweets ratio
df['followers_tweets_ratio'] = df['followers_count']/df['statuses_count']
df.replace([np.NaN, np.Inf], 0, inplace=True)
df['5x_followers_tweets'] = df['followers_tweets_ratio'] >= 5
df['5x_followers_tweets'] = df['5x_followers_tweets'].astype(int)

# Has less than 20 followers
df['less_than_20_followers'] = df['followers_count'] <= 20
df['less_than_20_followers'] = df['less_than_20_followers'].astype(int)

# Has description
df['description'] = df['description'].astype(str)
df['has_description'] = df['description'] != '0'
df['has_description'] = df['has_description'].astype(int)


# Has profile background image
df['has_profile_background_image'] = df['profile_use_background_image'] == 1.0
df['has_profile_background_image'] = df['has_profile_background_image'].astype(int)

# Has geolocation enabled
df['has_geo_enabled'] = df['geo_enabled'] == 1.0
df['has_geo_enabled'] = df['has_geo_enabled'].astype(int)

# Create feature space
df = df[[
    '20x_friends_followers',
    '100x_friends_followers',
    '100x_followers_listed',
    'is_in_a_list',
    'lang_is_En',
    'has_never_favorited',
    '5x_followers_tweets',
    'less_than_20_followers',
    'has_description',
    'has_profile_background_image',
    'has_geo_enabled',
    'is_bot'
]]

# Export features dataframe to csv
df.to_csv('data/selected_features.csv')

# Get correlations
df = df.round(2)
corrmat = df.corr().round(2)

# Plot Heatmap
sns.heatmap(corrmat, cmap='BuPu', annot=True)
plt.show()