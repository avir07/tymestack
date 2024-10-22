FROM python:3.8-slim

# Install dependencies
RUN pip install xgboost scikit-learn pytest pandas numpy argparse

# Copy the training and testing script
COPY train_model.py /app/train_model.py
COPY test_model.py /app/test_model.py

# Set the working directory
WORKDIR /app

# Run tests
CMD ["pytest", "test_model.py"]
