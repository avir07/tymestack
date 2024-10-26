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
    serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/xgboost-gpu.1-7:latest',  # Updated image URI
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
        18.0,     # ZN
        2.31,     # INDUS
        0,        # CHAS
        0.538,    # NOX
        6.575,    # RM
        65.2,     # AGE
        4.09,     # DIS
        1,        # RAD
        296,      # TAX
        15.3,     # PTRATIO
        396.9,    # B
        4.98      # LSTAT
    ]
]

# Make a prediction request
response = endpoint.predict(instances=sample_features)

# Print the prediction response
print("Prediction response:", response.predictions)

# Optional cleanup
# endpoint.undeploy_all()  # Uncomment if you want to undeploy the model
# endpoint.delete()  # Uncomment if you want to delete the endpoint
