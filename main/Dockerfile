# Base image with Python
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install the required packages
RUN pip install --no-cache-dir \
    xgboost \
    scikit-learn \
    pandas \
    argparse \
    numpy \
    joblib \
    google-cloud-storage \
    google-cloud-aiplatform

# Copy the script into the container
COPY best_model_pipeline2.py best_model_pipeline2.py
COPY train_model.py train_model.py

# Define the command to run your script
CMD ["python", "best_model_pipeline2.py"]
