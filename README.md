# Tymestack Project: Architecture and Tools

This project utilizes Google Kubernetes Engine (GKE) for parallelized job processing, with model training and deployment orchestrated across several Google Cloud Platform (GCP) services, including Cloud Storage, Vertex AI, and Cloud Build. Below is a detailed overview of the architecture, tools used, and challenges encountered during implementation.

## Architecture Overview

### Task 1: Model Training and Deployment

#### 1. **Google Kubernetes Engine (GKE)**
   - **Cluster**: A GKE cluster (found under Kubernetes Engine > Clusters) is configured for parallel job execution, allowing efficient resource usage and scalability.
   - **Parallelization**: Jobs are executed in parallel to maximize performance, managed under Kubernetes Engine > Workloads.
   - **Job Execution**:
     - Jobs are defined via YAML files that specify Docker images and configurations.
     - The jobs run as pods on GKE, where each pod utilizes the Python script `best_model_pipeline.py` (which also references `train_model.py` for end-to-end pipeline execution).
     - Docker images are pushed to Google Container Registry (Artifact Registry on GCP), making them accessible to GKE.

   - **GCP Project Details**:
     - **Project Name**: `tymestack`
     - **Project ID**: `tymestack-439409`
     - **GitHub Repository**: `tymestack-github`
     - **Bucket**: `tymestack-bucket`

   - **Docker and Image Building**:
     - The Docker image is built using a `Dockerfile` that includes the required dependencies listed in `requirements.txt`.
     - This image is pushed to the Artifact Registry for use in GKE jobs.

#### 2. **Cloud Storage**
   - Used for storing artifacts such as model files and JSON configurations.
   - Buckets are accessed by the `train_model.py` and `best_deployment.py` scripts to handle artifacts and upload the trained model.
   - **Bucket Access**:
     - Training results and hyperparameters are uploaded to the bucket from the `train_model.py`.
     - `best_deployment.py` accesses the bucket to load model data for optimal hyperparameter settings.

#### 3. **Vertex AI**
   - Vertex AI is used for managing trained models, including deployment and online predictions.
   - Models are stored and registered within the Vertex AI model registry, where they can be accessed and deployed for predictions.

### Task 2: Continuous Integration and Deployment

#### 1. **Git Repository and Cloud Build Integration**
   - **Repository**: `tymestack-github` contains all project files, managed via Git commands (`add`, `commit`, `push`).
   - **Cloud Build**:
     - **Linking GitHub**: Cloud Build is linked to the GitHub repository to automate build triggers.
     - **Pipeline Trigger**: Cloud Build triggers are set up to initiate the CI/CD pipeline on code push events.
     - **Pipeline Stages**:
       - First, the Docker image is generated from the repository code.
       - The image is pushed to the repository (`tymestack-github`) in Artifact Registry.
       - Environment variables are set for GKE cluster configuration.
       - Finally, the cluster is launched using the `xgboost-job-ci.yaml`, executing the XGBoost model.

## Tools and Services Used

| Component           | Tool/Service         | Purpose                                                 |
|---------------------|----------------------|---------------------------------------------------------|
| **Cluster**         | GKE                  | Parallel job execution with containerized workloads     |
| **Image Registry**  | Artifact Registry    | Stores Docker images for Kubernetes jobs                |
| **Storage**         | Cloud Storage        | Stores model artifacts and JSON files                   |
| **Model Management**| Vertex AI            | Manages and deploys trained models                      |
| **CI/CD**           | Cloud Build          | Automates Docker builds, links GitHub repo, and deploys |
| **Permissions**     | IAM                  | Sets up service account permissions for secure access   |

## Trade-Offs

1. **Data Loading Constraints**:
   - SSL usage was required for security but introduced issues with data loading from external sources.
   
2. **Model Prediction Format**:
   - XGBoost model requires input data in `DMatrix` format, limiting browser-based prediction inputs without conversion or custom handling.

3. **XGBoost Framework Compatibility**:
   - The highest available XGBoost version on GCP is 1.7, so we had to downgrade from a more recent version. This led to compatibility issues and unexpected errors until the versions were fully aligned across services.

## Challenges

1. **Service Account and IAM Permissions**:
   - A dedicated Kubernetes service account was created and annotated to interact with the Google service account for controlled access across GKE, Cloud Storage, and Vertex AI.
   - Configuring IAM permissions involved granting precise access levels, which required troubleshooting to ensure interoperability between services and security compliance.

2. **YAML Configurations**:
   - Defining environment variables and configuring job specifications in YAML for the GKE cluster required careful attention to syntax and version compatibility with GCP services.

This README provides a snapshot of the architecture, tools, trade-offs, and challenges encountered in the Tymestack project. This setup enables efficient model training and deployment with high scalability and automation across GCP services.
