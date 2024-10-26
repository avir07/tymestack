import pandas as pd
import numpy as np
import ssl
from google.cloud import storage


def convert_to_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

# Initialize Google Cloud Storage client
def upload_to_gcs(bucket_name, blob_name, content):
    """Uploads content to a Google Cloud Storage bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(content)
    print(f"Uploaded {blob_name} to bucket {bucket_name}.")


# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]


full_data = np.column_stack((data, target))

# Create a DataFrame with the desired column names
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.DataFrame(full_data, columns=columns)

# Display the DataFrame
#print(df.head())

y = df['MEDV']
X = df.drop('MEDV', axis=1)

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import json

# Load your data (replace with your dataset)

# Split into training and validation set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid for tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1]
}


# Random search for hyperparameters
def random_search(params, n_iters=5):
    results = []
    for i in range(n_iters):
        params['n_estimators'] = np.random.choice(param_grid['n_estimators'])
        params['max_depth'] = np.random.choice(param_grid['max_depth'])
        params['learning_rate'] = np.random.choice(param_grid['learning_rate'])
        params['subsample'] = np.random.choice(param_grid['subsample'])

        # Train XGBoost model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        # Evaluate model performance
        preds = model.predict(X_valid)
        mse = mean_squared_error(y_valid, preds)
        results.append((params, mse))

        # Store results (for Kubernetes job artifacts)
        result_data = {'params': params, 'mse': convert_to_serializable(mse)}
        local_file_content = json.dumps(result_data, default=convert_to_serializable)
        gcs_blob_name = f'trial_{i}.json'  # You can use a different naming scheme
        upload_to_gcs(bucket_name, gcs_blob_name, local_file_content)


    return sorted(results, key=lambda x: x[1])  # Return best result

bucket_name = 'tymestack-bucket'  # Replace with your GCS bucket name

# Run the random search
best_params = random_search({'objective': 'reg:squarederror'}, n_iters=10)
print(f"Best Parameters: {best_params[0]}")
##############################################################################

