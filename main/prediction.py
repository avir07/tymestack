import joblib
import xgboost as xgb
from google.cloud import storage
import io
import numpy as np
import ssl
import pandas as pd
from sklearn.model_selection import train_test_split


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


ssl._create_default_https_context = ssl._create_unverified_context
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
full_data = np.column_stack((data, target))
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.DataFrame(full_data, columns=columns)
y = df['MEDV']
X = df.drop('MEDV', axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


bucket_name = 'tymestack-bucket'
existing_model = download_model_from_gcs(bucket_name)
sampled_rows = X_valid.sample(n=3, random_state=42)
prediction_rows = existing_model.get_booster().predict(xgb.DMatrix(sampled_rows))

print(prediction_rows)
