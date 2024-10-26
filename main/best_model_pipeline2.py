import json
import joblib
import xgboost as xgb
from google.cloud import storage
from sklearn.metrics import mean_squared_error
import train_model
from sklearn.model_selection import train_test_split
import io


# Assuming 'df' is already defined in train_model
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

# Usage
bucket_name = 'tymestack-bucket'  # Replace with your GCS bucket name
best_params, best_mse = download_best_params_from_gcs(bucket_name)
print(f"Best Parameters: {best_params}, Best MSE: {best_mse}")

# Train the final model with the best hyperparameters
final_model = xgb.XGBRegressor(**best_params)
y = df['MEDV']
X = df.drop('MEDV', axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

final_model.fit(X_train, y_train)

# Evaluate the new model's MSE
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
    # If there's no existing model, set a high mse to ensure the new model replaces it
    existing_model_mse = float('inf')
    print("No existing model found. New model will be used as the initial model.")

# Compare and update the model only if the new model has a lower MSE
if new_model_mse < existing_model_mse:
    print("New model outperforms the existing model. Updating...")
    existing_model_path = 'model.joblib'  # Define the model path for saving
    joblib.dump(final_model, existing_model_path)

    # Upload the model to GCS
    def upload_model_to_gcs(bucket_name, model_file_path):
        """Uploads the trained model to GCS."""
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob('models/model.joblib')  # Change the path as needed
        blob.upload_from_filename(model_file_path)
        print(f"Uploaded model to GCS: models/model.joblib")

    upload_model_to_gcs(bucket_name, existing_model_path)

    # model deployment
    from google.cloud import aiplatform

    # Set up your Google Cloud project and region
    PROJECT_ID = 'tymestack-439409'
    REGION = 'us-central1'

    # Initialize Vertex AI
    aiplatform.init(project=PROJECT_ID, location=REGION)

    # Import the model
    model_display_name = 'model'
    model_path = 'gs://tymestack-bucket/models/'  # Ensure this path is correct

    # Upload the model
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_path,
        serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest',
        # Updated image URI
        serving_container_environment_variables={
            'AIP_MODEL_DIR': model_path,
        },
    )

    # Create an endpoint
    endpoint = aiplatform.Endpoint.create(display_name='model-endpoint')

    # Deploy the model to the endpoint
    deployed_model = endpoint.deploy(
        model=model,
        traffic_split={"0": 100},
    )

    sample_features = [
        [  # This outer list represents the 2D structure
            0.00632,  # CRIM
            18.0,  # ZN
            2.31,  # INDUS
            0,  # CHAS
            0.538,  # NOX
            6.575,  # RM
            65.2,  # AGE
            4.09,  # DIS
            1,  # RAD
            296,  # TAX
            15.3,  # PTRATIO
            396.9,  # B
            4.98  # LSTAT
        ]
    ]

    # Make a prediction request
    response = endpoint.predict(instances=sample_features)

    # Print the prediction response
    print("Prediction response:", response.predictions)

    # Optional cleanup
    # endpoint.undeploy_all()  # Uncomment if you want to undeploy the model
    # endpoint.delete()  # Uncomment if you want to delete the endpoint


else:
    print("Existing model performs better or equally well. No update made.")
