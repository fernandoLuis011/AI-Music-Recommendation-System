# The Dockerfile allows us to package the app (Python code, dependencies, and environment) into a single container image.
# Google Cloud Run uses this Dockerfile to build a container image.
# Once the image is built, it is deployed to a fully managed, serverless platform.
# Cloud Run automatically scales the container and makes it publicly accessible.
FROM python:3.9-slim

# Install dependencies
RUN pip install --no-cache-dir flask pandas joblib scikit-learn google-cloud-bigquery db-dtypes spotipy google-cloud-firestore

# Copy app files
COPY . /app
WORKDIR /app

# Expose Flask's default port
EXPOSE 8080

# Run the Flask app
CMD ["python", "app.py"]