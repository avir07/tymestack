# Base image with Python
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements (if any) and install them
COPY main/requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the script into the container
COPY main/best_model_pipeline2.py best_model_pipeline2.py
COPY main/train_model.py train_model.py

# Define the command to run your script
CMD ["python", "best_model_pipeline2.py"]