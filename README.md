# Invoice-Validation-System---IEEE-ODC

# Invoice validation API


This Invoice Validation System API is designed for the IEEE-ODC internship project and uses deep learning models to validate various components of invoices. The API is built with FastAPI and leverages a pre-trained models.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Before you start, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/akramLh005/Invoice-Validation-System---IEEE-ODC.git
   cd Invoice-Validation-System---IEEE-ODC/API
   ```

## I- OBJECT DETECTION : 

### 1. Running the Application with Docker Compose

This guide will walk you through setting up and running the application using Docker Compose, which simplifies the process of managing multi-container Docker applications. The setup includes containers for both the FastAPI server and TorchServe.

#### Prerequisites

- **Docker**: You need Docker installed on your machine. [Download Docker](https://www.docker.com/products/docker-desktop).
- **Docker Compose**: Ensure Docker Compose is installed. It usually comes with the Docker Desktop installation.

#### Setup

1. **Object detection folder**:
   If you haven't already, clone the repository to your local machine:
   ```bash
   cd object detection
   ```
2. **Build and Run the Containers:**
Use Docker Compose to build and start the services defined in your **docker-compose.yml:**
   ```bash

   docker-compose up --build
   ```
This command builds the images if they don't exist and starts the containers. The **--build** flag ensures that the images are re-built if there are changes.

    
 

### 2. Running the Server manually
**1. Preparing the Model**

Before starting the server, ensure the **checkpoints directory is appropriately set up**:

Checkpoints Directory: Verify that the checkpoints folder is not empty and contains the model files . The API relies on these pre-trained models to function correctly.
 
2. **Set Up a Virtual Environment (Optional but recommended)**
 
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements_ALL.txt
    ```
4. **To start the FastAPI server:**

```bash
uvicorn app:app --reload
```
This will launch the server on **http://127.0.0.1:8000**, where you can access the API documentation and test endpoints via Swagger UI or Postman.

-------------------------------------------------------------------------------------------------------------------------------
## Interacting with the API using Postman

Postman is a popular tool for testing APIs. It allows you to easily configure the requests, inspect the responses, and interact with APIs in a user-friendly interface. Here's how to send an invoice file to the FastAPI application using Postman:

### Setting Up Postman

1. **Download and Install Postman**:
   - If you haven't already installed Postman, download it from [Postman's official website](https://www.postman.com/downloads/) and install it on your machine.

2. **Create a New Request**:
   - Open Postman.
   - Click on the "New" button or the "+" tab to create a new request.

### Configuring the Request

3. **Set Up the Request Details**:
   - **Method**: Select `POST` from the dropdown menu.
   - **Request URL**: Enter `http://localhost:8000/detect` as the request URL.

4. **Configure the Request Body**:
   - Click on the 'Body' tab below the URL field.
   - Choose 'form-data' from the options available.
   - In the 'KEY' field, type `file`.
   - On the right side of the key field, change




## Part : ML Experiments with MLflow and DagsHub



### Prerequisites

- Python 3.7 or higher
- MLflow

### Installation
1. Clone the repository
```bash
git clone https://github.com/akramLh005/Invoice-Validation-System---IEEE-ODC.git
```

2. install dependencies :
```bash
pip install -r requirements.txt
```

 
### Configuration Steps : 
This guide will walk you through the steps to configure your model scripts to send metrics to our centralized MLflow server hosted on DagsHub.
##### 1. Set Environment Variables 

Before running your script, set the following environment variables with the provided credentials:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/akramLh005/Invoice-Validation-System---IEEE-ODC.mlflow
export MLFLOW_TRACKING_USERNAME=akramLh005
export MLFLOW_TRACKING_PASSWORD=<password>
```
Replace \<password\> with the actual password provided.

#### 2. Update Your Model Script 

In your model script, add the following lines to configure MLflow to use the DagsHub server:
```bash
import mlflow

remote_server_uri = "https://dagshub.com/akramLh005/Invoice-Validation-System---IEEE-ODC.mlflow"
mlflow.set_tracking_uri(remote_server_uri)
```

#### 3. Log Parameters and Metrics 
Within your script, use the mlflow.log_param and mlflow.log_metric functions to log parameters and metrics:
```bash
# Example parameters
mlflow.log_param("alpha", alpha)
mlflow.log_param("l1_ratio", l1_ratio)

# Example metrics
mlflow.log_metric("rmse", rmse)
mlflow.log_metric("r2", r2)
mlflow.log_metric("mae", mae)
```
Ensure you replace alpha, l1_ratio, rmse, r2, and mae with the actual variables from your script.

#### 4. Running Your Script
Run your script as usual. The metrics and parameters will automatically be logged to the DagsHub MLflow server.

#### 5. Verifying Metrics on DagsHub
After running your script, log into DagsHub and navigate to your project's MLflow page to verify that the metrics and parameters have been logged correctly.
