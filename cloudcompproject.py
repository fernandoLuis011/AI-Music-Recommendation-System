# -*- coding: utf-8 -*-
"""

Load, Clean, and Preprocess the dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.decomposition import PCA
import joblib

# Read the dataset
df = pd.read_csv('dataset.csv')
df.info()

#Drop rows that have the same song in different albums or singles(duplicates songs)
df.drop_duplicates(subset=['track_id'], inplace=True)

# Drop track_id from dataset
df = df.drop(columns=['track_id'])

# Drop rows with missing values
df = df.dropna()

# Drop songs with the same name,
# when we search for a song where there exists more songs with the same name, it causes some issues
df = df.drop_duplicates(subset=['track_name'])


#Select the features
features = ['valence', 'energy', 'tempo', 'danceability', 'acousticness', 'loudness', 'liveness']



X = df[features]

# Drop rows with missing values
df = df.dropna()



# Create mood labels based on thresholds
# Function to classify songs into 30 mood categories
def classify_mood(row):
    if row['valence'] > 0.8 and row['energy'] > 0.8:
        return 'Happy'
    elif row['valence'] < 0.3 and row['energy'] < 0.3:
        return 'Sad'
    elif row['acousticness'] > 0.8 and row['tempo'] < 90:
        return 'Calm'
    elif row['energy'] > 0.8 and row['tempo'] > 130:
        return 'Energetic'
    elif row['valence'] > 0.7 and row['danceability'] > 0.6:
        return 'Romantic'
    elif row['acousticness'] > 0.7 and row['liveness'] > 0.5:
        return 'Chill'
    elif row['danceability'] > 0.8 and row['energy'] > 0.7:
        return 'Party'
    elif row['valence'] > 0.7 and row['tempo'] > 100 and row['tempo'] < 130:
        return 'Uplifting'
    elif row['valence'] < 0.4 and row['acousticness'] > 0.5:
        return 'Melancholy'
    elif row['energy'] < 0.4 and row['acousticness'] > 0.7:
        return 'Relaxed'
    elif row['energy'] > 0.9 and row['valence'] < 0.3:
        return 'Aggressive'
    elif row['danceability'] > 0.7 and row['tempo'] > 110 and row['tempo'] < 140:
        return 'Funky'
    elif row['loudness'] > -5 and row['energy'] > 0.8 and row['tempo'] > 140:
        return 'Epic'
    elif row['acousticness'] > 0.8 and row['energy'] < 0.4:
        return 'Dreamy'
    elif row['danceability'] > 0.5 and row['tempo'] > 100 and row['tempo'] < 120 and row['energy'] > 0.5:
        return 'Groovy'
    elif row['valence'] < 0.4 and row['energy'] > 0.8:
        return 'Dark'
    elif row['acousticness'] > 0.6 and row['energy'] < 0.4:
        return 'Ambient'
    elif row['energy'] > 0.7 and row['tempo'] > 120:
        return 'Excited'
    elif row['acousticness'] > 0.8 and row['tempo'] < 70:
        return 'Serene'
    elif row['valence'] < 0.4 and row['acousticness'] > 0.5 and row['tempo'] < 100:
        return 'Reflective'
    elif row['danceability'] > 0.6 and row['energy'] > 0.8 and row['liveness'] > 0.5:
        return 'Wild'
    elif row['valence'] < 0.3 and row['danceability'] > 0.5:
        return 'Nostalgic'
    elif row['speechiness'] > 0.4 and row['acousticness'] > 0.7:
        return 'Hypnotic'
    elif row['valence'] > 0.6 and row['acousticness'] > 0.6:
        return 'Emotional'
    elif row['danceability'] > 0.7 and row['tempo'] > 110 and row['energy'] > 0.7:
        return 'Festive'
    elif row['loudness'] > -3 and row['tempo'] > 100:
        return 'Cinematic'
    elif row['energy'] < 0.4 and row['acousticness'] > 0.6:
        return 'Soft'
    elif row['loudness'] > -6 and row['tempo'] > 140:
        return 'Heavy'
    elif row['tempo'] > 120 and row['valence'] > 0.5:
        return 'Dynamic'
    elif row['speechiness'] > 0.4 and row['valence'] < 0.3:
        return 'Eerie'
    else:
        return 'Neutral'

# Apply the function to the DataFrame
df['mood'] = df.apply(classify_mood, axis=1)
y = df['mood']

# Display the distribution of moods
print(df['mood'].value_counts())

"""Visualizations to understand the dataset"""

# Mood Distribution Graph
df['mood'].value_counts().plot(kind='bar', figsize=(10, 6))
plt.title("Mood Distribution")
plt.xlabel("Mood")
plt.ylabel("Count")
plt.show()

# Correlation heatmap of features
correlation_matrix = X.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Heatmap")
plt.show()

# Pairwise scatterplots for key features
sns.pairplot(df, vars=['valence', 'energy', 'danceability'], hue='mood', palette='viridis')
plt.suptitle("Pairwise Scatterplots of Key Features", y=1.02)
plt.show()

# Feature distributions
for feature in features:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[feature], kde=True, bins=30, color='skyblue')
    plt.title(f"Distribution of {feature.capitalize()}")
    plt.xlabel(feature.capitalize())
    plt.ylabel("Frequency")
    plt.show()

"""#Model Training

# Using Random Forest Classifier
"""

# Train a Random Forest Classifier
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum depth of each tree
    min_samples_split=5,   # Minimum samples to split an internal node
    min_samples_leaf=2,    # Minimum samples at a leaf node
    random_state=42,       # For reproducibility
    n_jobs=-1              # Use all CPU cores for parallel processing
)
rf.fit(X_train, y_train)

# Predict moods for the entire dataset
df['predicted_mood'] = rf.predict(df[features])

# Evaluate the model
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))

# Get feature importances from Random Forest
importances = rf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("Feature Importances:\n", feature_importance)
print()

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# Convert confusion matrix to a DataFrame
cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
cm_df = pd.DataFrame(cm, index=rf.classes_, columns=rf.classes_)

# Print the DataFrame
print("Confusion Matrix:")
print(cm_df)

"""#Clustering Genres with K-Means

This will allow us to explore patterns in the data and assign moods based on clusters, independent of the predefined mood labels.
"""

# Determine the optimal number of clusters using the Elbow Method
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
inertia = []
cluster_range = range(1, 15)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, inertia, marker='o')
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# Prepare numerical features for clustering
X = df[features]

# Create a K-Means clustering pipeline
cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=5, random_state=42))
])

# Fit the pipeline and predict cluster labels
cluster_pipeline.fit(X)
df['cluster'] = cluster_pipeline.predict(X)

# Display the number of samples per cluster
print(df['cluster'].value_counts())

# Create a PCA pipeline
pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2))
])

# Generate PCA embedding
pca_embedding = pca_pipeline.fit_transform(X)

# Create a DataFrame for visualization
pca_projection = pd.DataFrame(columns=['x', 'y'], data=pca_embedding)
pca_projection['track_name'] = df['track_genre']
pca_projection['cluster'] = df['cluster']

# Visualize using Plotly
fig = px.scatter(
    pca_projection, x='x', y='y', color='cluster',
    hover_data=['x', 'y', 'track_name'], title="PCA Clustering of Songs"
)
fig.show()

# Summarize feature means by cluster
cluster_summary = df.groupby('cluster')[features].mean()

# Print the cluster summary
print(cluster_summary)

# Plot feature distributions by cluster
for feature in features:
    cluster_summary[feature].plot(kind='bar', figsize=(10, 5), title=f"{feature.capitalize()} by Cluster")
    plt.ylabel(feature.capitalize())
    plt.xlabel("Cluster")
    plt.show()

# Define the mapping of clusters to mood categories based on analysis
cluster_to_mood = {
    0: 'Energetic and Upbeat',
    1: 'Calm and Acoustic',
    2: 'Lively and Experimental',
    3: 'Relaxed and Reflective',
    4: 'Intense and Driving'
}

# Map the clusters to mood categories
df['mood_kmeans'] = df['cluster'].map(cluster_to_mood)

# Display the first few rows with mood labels
print(df[['cluster', 'mood_kmeans']].head())

# Display the distribution of moods
print("\nMood Distribution:")
print(df['mood_kmeans'].value_counts())

#The distribution of mood categories
plt.figure(figsize=(10, 6))
sns.countplot(y='mood_kmeans', data=df, order=df['mood_kmeans'].value_counts().index, palette='viridis')
plt.title("Distribution of Songs Across Moods (K-Means Clusters)")
plt.xlabel("Count")
plt.ylabel("Mood Category")
plt.show()

"""Based on the visualizations, we can see that similar genres tend to have data points that are located close to each other. We can use this to build a recommendation system by taking the data points of the songs a user has listened to and recommending songs corresponding to nearby data points.

How the Recommendation System Works
Input Songs:

The user provides a list of songs (with track_name).
These songs are either:
Matched in the dataset (spotify_data), or
Searched using the Spotify API (if not found in the dataset).

Feature Extraction:

For each input song, the system retrieves its corresponding audio features. These include:
Valence: Emotional positivity of the song.

Energy: Intensity and activity level.

Danceability: How suitable the track is for dancing.

Tempo: Speed or beats per minute.

Etc.

Mean Feature Vector:

The system calculates the mean feature vector of all input songs, representing the "average sound profile" of the user's selection.
For example, if the input songs are mostly high-energy and danceable, the mean vector will reflect these characteristics.

Cosine Similarity:

The system computes the cosine similarity between the mean vector and the feature vectors of all songs in the dataset (spotifyDataset).
Songs with smaller cosine distances are considered more similar to the input songs.

Recommendation Output:

The top n closest songs (based on cosine similarity) are recommended.
These songs exclude the input songs to ensure uniqueness.

The recommendation system uses the cluster pipeline, which combines feature scaling (StandardScaler) and K-Means clustering, to group songs with similar characteristics. First, the pipeline scales the dataset's audio features (like valence, energy, and tempo) for consistent clustering. When a user provides input songs, their features are scaled using the same pipeline, ensuring they match the dataset's format.

The system computes cosine distances between the input songs and the dataset to find the most similar songs, prioritizing those in the same cluster as the input songs. By grouping songs into clusters, the system ensures recommendations are relevant, focused, and consistent with the user's preferences.

The recommend_songs_by_mood_rf function provides mood-based song recommendations. It filters a dataset of songs to include only those that match a specified predicted mood (Happy, Uplifting, etc). After filtering, it uses the recommend_songs function to find the top n_songs most similar songs to a given list of input songs.
"""

!pip install spotipy --upgrade

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

# Spotify API credentials
client_id = "23256f1399574539ada16366afb2abd2"
client_secret = "53d5393f6471482aa88e58cac57433d6"

# Define the necessary scopes
scopes = "user-library-read user-top-read user-read-recently-played user-follow-read playlist-read-private user-read-email user-read-private"

# Create a SpotifyOAuth object with the required scopes
auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)

# Spotify API client
sp = spotipy.Spotify(auth_manager=auth_manager, requests_timeout=10, retries=3)

# List of numerical features to consider for recommendations
number_cols = ['valence', 'acousticness', 'danceability', 'duration_ms', 'energy',
               'explicit', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
               'popularity', 'speechiness', 'tempo']


# Function to find a song on Spotify based on its name
def find_song(name):
    song_data = defaultdict()
    results = sp.search(q=f'track: {name}', limit=1)
    if not results['tracks']['items']:
        return None

    track_info = results['tracks']['items'][0]
    track_id = track_info['id']
     # Check if audio features are available
    try:
        audio_features = sp.audio_features(track_id)[0]
    except spotipy.exceptions.SpotifyException as e:
        print(f"Error fetching audio features for {name}: {e}")  # Print the error
        return None  # Return None if features are not available

    # Extract basic metadata and audio features
    song_data['track_name'] = name
    song_data['explicit'] = int(track_info['explicit'])
    song_data['duration_ms'] = track_info['duration_ms']
    song_data['popularity'] = track_info['popularity']

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame([song_data])


# Function to retrieve song data from the dataset or Spotify API
def get_song_data(song, spotify_data):
    try:
        # Check if the song exists in the dataset
        song_data = spotify_data[spotify_data['track_name'] == song['track_name']].iloc[0]
        return song_data
    except IndexError:
        # Search for the song on Spotify if not found in the dataset
        print(f"Song '{song['track_name']}' not found in dataset, searching on Spotify...")
        spotify_result = find_song(song['track_name'])
        if spotify_result is not None:
            return spotify_result
        else:
            print(f"Warning: Song '{song['track_name']}' not found on Spotify.")
            return None

# Function to compute the mean vector for a list of songs
def get_mean_vector(song_list, spotify_data):
    song_vectors = []

    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print(f"Warning: {song['track_name']} does not exist in Spotify or in the dataset.")
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)

    # Compute the mean vector
    if song_vectors:
        song_matrix = np.array(song_vectors)
        return np.mean(song_matrix, axis=0)
    else:
        return None


# Flatten a list of dictionaries into a single dictionary
def flatten_dict_list(dict_list):
    flattened_dict = defaultdict(list)
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
    return flattened_dict


# Function to recommend similar songs
def recommend_songs(song_list, spotify_data, mood = None, cluster_pipeline=None, n_songs=10):
    metadata_cols = ['track_name', 'artists']
    song_dict = flatten_dict_list(song_list)

    # Compute the mean vector for the input songs
    song_center = get_mean_vector(song_list, spotify_data)
    if song_center is None:
        print("No valid songs found in the input list.")
        return []

    # Scale the dataset and the input mean vector
    if cluster_pipeline:
        feature_names_fit = cluster_pipeline.steps[0][1].feature_names_in_  # Get feature names from scaler

        # Ensure song_center and spotify_data have the same features and column names
        song_center = song_center.reshape(1, -1)  # Reshape song_center to 2D
        song_center_df = pd.DataFrame(song_center, columns=number_cols)  # Create a DataFrame with column names
        song_center_df = song_center_df[feature_names_fit]  # Select only the features used during fitting

        # Select only the features used during fitting from spotify_data
        scaled_data = spotify_data[feature_names_fit]

        scaler = cluster_pipeline.steps[0][1]  # Assumes the first step in the pipeline is the scaler
        scaled_data = scaler.transform(scaled_data)  # Transform the data

        # Transform song_center_df (DataFrame with column names)
        scaled_song_center = scaler.transform(song_center_df)
    else:
        scaled_data = spotify_data[number_cols].values
        scaled_song_center = song_center.reshape(1, -1)

    # Compute cosine distances between the mean vector and all songs in the dataset
    distances = cdist(scaled_song_center, scaled_data, metric='cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    # Retrieve the recommended songs
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['track_name'].isin(song_dict['track_name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

def recommend_songs_by_mood_rf(song_list, spotify_data, mood, n_songs=10):
    # Filter the dataset by the predicted mood
    mood_data = spotify_data[spotify_data['predicted_mood'] == mood]

    # If no songs match the mood, return a message
    if mood_data.empty:
        return f"No songs found for the mood: {mood}"

    # Use the existing recommendation logic with the filtered dataset
    return recommend_songs(song_list, mood_data, n_songs=n_songs)

# usage of recommendation system using the cluster pipeline

spotify_data = df.copy()

spotify_data.to_csv('spotify_data.csv', index=False)

# input songs
recommendations = recommend_songs(
    [{'track_name': 'Safe and Sound'},
     {'track_name': 'Life, Life'},
     {'track_name': 'Animals'},
     {'track_name': 'Akta dig för polisen'},
     {'track_name': 'Rooster (2022 Remaster)'},
     {'track_name': 'Smells Like Teen Spirit'}],
    spotify_data,
    cluster_pipeline=cluster_pipeline,
    n_songs=10
)

# Display recommendations
for idx, song in enumerate(recommendations, 1):
    print(f"{idx}. {song['track_name']} by {song['artists']}")

# Usage of the recommendation system using the random forest classifier

recommendations = recommend_songs_by_mood_rf(
    [{'track_name': 'Smells Like Teen Spirit'}],
    spotify_data,
    mood='Uplifting',
    n_songs=10
)

# Display recommendations
for idx, song in enumerate(recommendations, 1):
    print(f"{idx}. {song['track_name']} by {song['artists']}")

# Use joblib to save the random forest model
import joblib
joblib.dump(rf, 'random_forest_model.pkl')

# use joblib to save the clustering pipeline
joblib.dump(cluster_pipeline, 'kmeans_pipeline.pkl')

# Load the models
rf_loaded = joblib.load('random_forest_model.pkl')
cluster_pipeline_loaded = joblib.load('kmeans_pipeline.pkl')

# Test the loaded models
print(rf_loaded.predict(X_test[:5]))
print(cluster_pipeline_loaded.predict(X[:5]))