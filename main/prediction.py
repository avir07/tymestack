from google.cloud import aiplatform

# Set up your Google Cloud project and region
PROJECT_ID = 'tymestack-439409'
REGION = 'us-central1'
ENDPOINT_ID = '4449364017307189248'  # Replace with your actual endpoint ID

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)

# Load the existing endpoint
endpoint = aiplatform.Endpoint(endpoint_name=f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}")

# Sample feature data for prediction, wrapped in a list to create a 2D array
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
