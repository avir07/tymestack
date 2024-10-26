import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from deployed_model_prediction import load_model, predict  # Assuming these exist in your project

def test_model_accuracy():
    # Load the current and new model
    old_model = joblib.load('gs://your-bucket/model.joblib')  # Replace with actual GCS path
    new_model = load_model()  # Assuming load_model loads the new model

    # Example data (replace with actual test data)
    X_test = np.array([[1, 2, 3], [4, 5, 6]])  # Replace with real test data
    y_test = np.array([0, 1])

    # Get predictions
    old_predictions = old_model.predict(X_test)
    new_predictions = predict(X_test)

    # Compare accuracy
    old_accuracy = accuracy_score(y_test, old_predictions)
    new_accuracy = accuracy_score(y_test, new_predictions)

    assert new_accuracy > old_accuracy, "New model performance is worse!"
