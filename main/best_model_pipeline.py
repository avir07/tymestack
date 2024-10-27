import json
import joblib
import xgboost as xgb
from google.cloud import storage
from sklearn.metrics import mean_squared_error
import train_model
from sklearn.model_selection import train_test_split
import io
from google.cloud import aiplatform


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


def download_model_from_gcs(bucket_name):
    """Downloads the existing model from GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs()

    for blob in blobs:
        print(blob)
        if blob.name.startswith('models/') and blob.name.endswith('.joblib'):
            try:
                content = blob.download_as_bytes()

                # Load the joblib model from the bytes using BytesIO
                existing_model = joblib.load(io.BytesIO(content))
                return existing_model  # Return the model if found
            except Exception as e:
                print(f"No model exists in {blob.name}: {e}")

    return None  # Return None if no model found


def upload_model_to_gcs(bucket_name, model_file_path):
    """Uploads the trained model to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob('models/model.joblib')  # Change the path as needed
    blob.upload_from_filename(model_file_path)
    print(f"Uploaded model to GCS: models/model.joblib")

# loading dataset & bucket, alongside training and validation sets
df = train_model.df
y = df['MEDV']
X = df.drop('MEDV', axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.9, random_state=42)
bucket_name = 'tymestack-bucket'

# Train the final model with the best hyperparameters and evaluating MSE
best_params, best_mse = download_best_params_from_gcs(bucket_name)
print(f"Best Parameters: {best_params}, Best MSE: {best_mse}")
final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X_train, y_train)
new_model_predictions = final_model.predict(X_valid)
new_model_mse = mean_squared_error(y_valid, new_model_predictions)
print(f"New Model MSE: {new_model_mse}")

# Load the existing model if it exists and compute its MSE
existing_model = download_model_from_gcs(bucket_name)
if existing_model is not None:
    existing_model_predictions = existing_model.predict(X_valid)
    existing_model_mse = mean_squared_error(y_valid, existing_model_predictions)
    print(f"Existing Model MSE: {existing_model_mse}")
else:
    existing_model_mse = float('inf')
    print("No existing model found. New model will be used as the initial model.")

# Compare and update the model only if the new model has a lower MSE
if new_model_mse < existing_model_mse:
    print("New model outperforms the existing model. Updating...")
    existing_model_path = 'model.joblib'
    final_model.set_params(gpu_id=-1)
    joblib.dump(final_model, existing_model_path)
    upload_model_to_gcs(bucket_name, existing_model_path)

    # Setting up Google Cloud project and region
    PROJECT_ID = 'tymestack-439409'
    REGION = 'us-central1'

    # model deployment on VertexAI
    aiplatform.init(project=PROJECT_ID, location=REGION)
    model_display_name = 'model'
    model_path = 'gs://tymestack-bucket/models/'
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_path,
        serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest',
        serving_container_environment_variables={
            'AIP_MODEL_DIR': model_path,
        },
    )

    # model endpoint on VertexAI
    endpoint = aiplatform.Endpoint.create(display_name='model-endpoint')
    deployed_model = endpoint.deploy(
        model=model,
        traffic_split={"0": 100},
    )

    # Optional cleanup
    # endpoint.undeploy_all()
    # endpoint.delete()
else:
    print("Existing model performs better or equally well. No update made.")
