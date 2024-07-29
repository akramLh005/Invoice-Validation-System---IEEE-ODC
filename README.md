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
   
 2. **Set Up a Virtual Environment (Optional but recommended)**
 
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```
    
### Preparing the Model

Before starting the server, ensure the **checkpoints directory is appropriately set up**:

Checkpoints Directory: Verify that the checkpoints folder is not empty and contains the model files . The API relies on these pre-trained models to function correctly.

### Running the Server

To start the FastAPI server:

```bash
uvicorn app:main --reload
```
This will launch the server on **http://127.0.0.1:8000**, where you can access the API documentation and test endpoints via Swagger UI or Postman.





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
