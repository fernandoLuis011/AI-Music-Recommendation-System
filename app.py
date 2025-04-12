"""
A Flask app thaht serves as the backend for the system.
Users provide song inputs and select an algorithm (random_forest or kmeans) via the web form.
The app processes the input and generates recommendations using the selected algorithm.
The recommendations are returned in JSON format and displayed on the frontend.
User inputs and recommendations are saved to Firestore for later retrieval.
Users can view past recommendations by entering their user ID.
"""

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from google.cloud import firestore

# Initialize Firestore client
db = firestore.Client()

def save_recommendations(user_id, input_data, recommendations):
    """
    Save user input and recommendations to Firestore.
    
    Parameters:
        user_id (str): Unique identifier for the user.
        input_data (dict): User input data (songs, mood, algorithm, etc.).
        recommendations (list): List of recommended songs.
    """
    db.collection("recommendations").document(user_id).set({
        "input_data": input_data,
        "recommendations": recommendations
    })

def get_recommendations(user_id):
    """
    Retrieve past recommendations for a user from Firestore.
    
    Parameters:
        user_id (str): Unique identifier for the user.

    Returns:
        dict: User input data and recommendations.
    """
    doc = db.collection("recommendations").document(user_id).get()
    if doc.exists:
        return doc.to_dict()
    return None


# Spotify API credentials
client_id = "23256f1399574539ada16366afb2abd2"
client_secret = ""
auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

# Flask app initialization
app = Flask(__name__)

# Load pre-trained models
rf_model = joblib.load("random_forest_model.pkl")
kmeans_model = joblib.load("kmeans_pipeline.pkl")
features = ['valence', 'acousticness', 'danceability', 'duration_ms', 'energy',
               'explicit', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
               'popularity', 'speechiness', 'tempo']

# Load Spotify dataset
spotify_data = pd.read_csv("spotify_data.csv")  


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


"""
The get_mean_vector function is responsible for calculating a mean vector from the audio features of a list of songs. 
This mean vector represents the "average" characteristics of the provided songs, 
which can then be used as the input to a recommendation algorithm to find similar songs.
Example:
Let's say the user inputs three songs:

Song A: valence=0.8, energy=0.9, tempo=120, etc
Song B: valence=0.6, energy=0.7, tempo=115, etc
Song C: valence=0.7, energy=0.8, tempo=125, etc
The function will compute the mean vector:

valence = (0.8 + 0.6 + 0.7) / 3 = 0.7
energy = (0.9 + 0.7 + 0.8) / 3 = 0.8
tempo = (120 + 115 + 125) / 3 = 120

Resulting mean vector:
[valence=0.7, energy=0.8, tempo=120]
"""
# Function to compute the mean vector for a list of songs
def get_mean_vector(song_list, spotify_data):
    song_vectors = []

    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print(f"Warning: {song['track_name']} does not exist in Spotify or in the dataset.")
            continue
        song_vector = song_data[features].values
        song_vectors.append(song_vector)

    # Compute the mean vector
    if song_vectors:
        song_matrix = np.array(song_vectors)
        return np.mean(song_matrix, axis=0)
    else:
        return None


"""
The flatten_dict_list function takes a list of dictionaries and combines their data into a single dictionary,
 where the keys are preserved, and the values from all dictionaries are collected into lists.

Input: A list of dictionaries.
Each dictionary represents a set of key-value pairs.

{"track_name": "Song A", "artist": "Artist A"},
{"track_name": "Song B", "artist": "Artist B"}

Output: A single dictionary, where:

Keys from the original dictionaries become keys in the flattened dictionary.
Values for each key are combined into a list.

Example:
    "track_name": ["Song A", "Song B"],
    "artist": ["Artist A", "Artist B"]
"""
# Flatten a list of dictionaries into a single dictionary
def flatten_dict_list(dict_list):
    flattened_dict = defaultdict(list)
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
    return flattened_dict


"""
The recommend_songs function is responsible for generating song recommendations based on a list of input songs. 
It uses the features of the input songs, computes a mean vector, 
and finds the closest songs from a dataset using a clustering pipeline.
"""

# Function to recommend similar songs using K-Means pipeline
def recommend_songs(song_list, spotify_data, mood = None, cluster_pipeline=None, n_songs=10):
    metadata_cols = ['track_name', 'artists']
    song_dict = flatten_dict_list(song_list)

    # Compute the mean vector for the input songs
    song_center = get_mean_vector(song_list, spotify_data)
    if song_center is None:
        print("No valid songs found in the input list.")
        return []

    # Scale the dataset and the input mean vector
    """
    This part of the code the clustering model assigns songs to clusters based on their similarity.
    """
    if cluster_pipeline:
        feature_names_fit = cluster_pipeline.steps[0][1].feature_names_in_  # Get feature names from scaler

        # Ensure song_center and spotify_data have the same features and column names
        song_center = song_center.reshape(1, -1)  # Reshape song_center to 2D
        song_center_df = pd.DataFrame(song_center, columns=features)  # Create a DataFrame with column names
        song_center_df = song_center_df[feature_names_fit]  # Select only the features used during fitting

        # Select only the features used during fitting from spotify_data
        scaled_data = spotify_data[feature_names_fit]

        scaler = cluster_pipeline.steps[0][1]  # Assumes the first step in the pipeline is the scaler
        scaled_data = scaler.transform(scaled_data)  # Transform the data

        # Transform song_center_df (DataFrame with column names)
        scaled_song_center = scaler.transform(song_center_df)
    else:
        scaled_data = spotify_data[features].values
        scaled_song_center = song_center.reshape(1, -1)

    """
    This part of the code calculates the cosine similarity between the
    scaled mean vector (song_center) and all songs in the dataset.
    Finds the n_songs closest matches (smallest cosine distance).

    Cosine Similarity
    Cosine similarity measures the similarity between two vectors (data points) by calculating the cosine of the angle between them. 
    It is commonly used to find how "close" two data points are in a high-dimensional space. 
    The value of cosine similarity ranges between -1 and 1:

    1: The vectors are perfectly similar (pointing in the same direction).
    0: The vectors are orthogonal (no similarity at all).
    -1: The vectors are completely opposite (pointing in opposite directions).
    The formula for cosine similarity is:

    Cosine Similarity = A x B / ||A|| x ||B||

    Where: 
    A and B are the feature vectors of two data points (songs) 
    A x B is the product of the two vectors. 
    ||A|| and ||B|| are the magnitudes (norms) of the vectors.

    Example: Suppose we have two songs represented by their features: 
    Song A: [ 0.8 , 0.7 , 120 , 0.9 , 0.2 , - 5.0 , 0.1 ] 
    Song B: [ 0.75 , 0.65 , 125 , 0.88 , 0.25 , - 5.2 , 0.12 ] 

    The cosine similarity between these two vectors is calculated to determine how similar their features are.
    A high cosine similarity (close to 1) indicates that Song B is a good recommendation if the user likes Song A. 
    
    Conversely, cosine distance would measure how "different" these two songs are.
    Cosine distance is derived from cosine similarity and measures the dissimilarity between two vectors. 
    It is computed as: 
    Cosine Distance = 1 - Cosine Similarity 

    0: The vectors are identical (maximum similarity).
    1: The vectors are completely dissimilar (no similarity).
    Values between 0 and 1 represent varying levels of dissimilarity.
    """
    # Compute cosine distances between the mean vector and all songs in the dataset
    distances = cdist(scaled_song_center, scaled_data, metric='cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    # Retrieve the recommended songs
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['track_name'].isin(song_dict['track_name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')


"""
This function, recommend_songs_by_mood_rf, is responsible for recommending songs based on a specific mood provided by the user. 
The function leverages the Random Forest classifier's predictions (stored in the predicted_mood column of the dataset) to filter 
the songs and provide recommendations.
"""
# Function to recommend similar songs using the random forest classifier model
def recommend_songs_by_mood_rf(song_list, spotify_data, mood, n_songs=10):
    # Filter the dataset by the predicted mood
    mood_data = spotify_data[spotify_data['predicted_mood'] == mood]

    # If no songs match the mood, return a message
    if mood_data.empty:
        return f"No songs found for the mood: {mood}"

    # Use the existing recommendation logic with the filtered dataset
    return recommend_songs(song_list, mood_data, n_songs=n_songs)


# Flask route for the homepage
@app.route("/")
def index():
    # Render the "index.html" template for the homepage
    return render_template("index.html")


# Flask route to handle song recommendations
@app.route("/recommend", methods=["POST"])
def recommend_endpoint():
    # Parse the JSON data sent in the POST request
    data = request.json
    
    # Extract the list of songs from the input and create a dictionary with "track_name"
    # Split the input songs by commas and strip any extra spaces
    song_list = [{"track_name": song.strip()} for song in data.get("song_list", "").split(",")]
    
    # Extract the chosen algorithm (random forest or k-means) from the input
    algorithm = data.get("algorithm")
    
    # Extract the optional mood parameter (for random forest) from the input
    mood = data.get("mood")
    
    # Extract the number of song recommendations to return; default is 10 if not provided
    n_songs = int(data.get("n_songs", 10))

    user_id = data.get("user_id", "default_user") 


    # If the random forest algorithm is selected
    if algorithm == "random_forest":
        # Use the random forest-based recommendation system and filter by the provided mood
        recommendations = recommend_songs_by_mood_rf(song_list, spotify_data, mood, n_songs=n_songs)
    # If the k-means algorithm is selected
    elif algorithm == "kmeans":
        # Use the k-means-based recommendation system
        recommendations = recommend_songs(song_list, spotify_data, cluster_pipeline=kmeans_model, n_songs=n_songs)
    # If the algorithm is invalid or not specified correctly
    else:
        # Return an error message in JSON format
        return jsonify({"error": "Invalid algorithm specified."})

    # Save to Firestore
    input_data = {
        "song_list": data.get("song_list"),
        "algorithm": algorithm,
        "mood": mood,
        "n_songs": n_songs
    }
    save_recommendations(user_id, input_data, recommendations)

    # Return the recommendations in JSON format
    return jsonify({"recommendations": recommendations})

# Flask route to handle past recommendations
@app.route("/past-recommendations", methods=["GET"])
def past_recommendations():
    """
    Endpoint to retrieve past song recommendations for a user.

    Query Parameters:
        - user_id: (optional) The ID of the user for whom past recommendations are to be retrieved.
          Defaults to "default_user" if no user_id is provided.

    Returns:
        - JSON response containing past recommendations if they exist.
        - JSON response with an error message if no recommendations are found.
    """
    user_id = request.args.get("user_id", "default_user") 
    past_data = get_recommendations(user_id)

    if past_data:
        return jsonify(past_data)
    return jsonify({"error": "No past recommendations found."})


# Main entry point of the Flask app
if __name__ == "__main__":
    # Run the Flask app on all network interfaces (host="0.0.0.0") and port 8080
    app.run(host="0.0.0.0", port=8080)

