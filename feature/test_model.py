import pytest
import xgboost as xgb
from sklearn.metrics import mean_squared_error

def test_model_accuracy():
    X, y = load_your_data()
    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X, y)

    preds = model.predict(X)
    mse = mean_squared_error(y, preds)

    # Ensure the Mean Squared Error is within an acceptable range
    assert mse < 1000, "Model MSE is too high!"
