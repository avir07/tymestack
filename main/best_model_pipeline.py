import json
from google.cloud import storage
import train_model
import xgboost as xgb


df = train_model.df

def download_best_params_from_gcs(bucket_name):
    """Downloads JSON files from GCS and finds the best hyperparameters."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs()

    results = []

    # Iterate through all blobs in the bucket
    for blob in blobs:
        if blob.name.endswith('.json'):
            # Download the blob content as a string
            content = blob.download_as_text()
            data = json.loads(content)
            results.append((data['params'], data['mse']))

    # Find the best parameters (lowest MSE)
    best_result = min(results, key=lambda x: x[1])
    return best_result


# Usage
bucket_name = 'tymestack-bucket'  # Replace with your GCS bucket name
best_params, best_mse = download_best_params_from_gcs(bucket_name)
print(f"Best Parameters: {best_params}, Best MSE: {best_mse}")


# Train the final model with the best hyperparameters
final_model = xgb.XGBRegressor(**best_params)
y = df['MEDV']
X = df.drop('MEDV', axis=1)
final_model.fit(X, y)

# Serialize and save the trained model to GCS
import joblib

# Save the model locally first
model_file_path = 'model.joblib'
joblib.dump(final_model, model_file_path)

# Upload the model to GCS
def upload_model_to_gcs(bucket_name, model_file_path):
    """Uploads the trained model to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob('models/final_model.joblib')  # Change the path as needed
    blob.upload_from_filename(model_file_path)
    print(f"Uploaded model to GCS: models/final_model.joblib")

# Usage
upload_model_to_gcs(bucket_name, model_file_path)
