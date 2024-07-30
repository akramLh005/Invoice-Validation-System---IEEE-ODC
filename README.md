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

Postman is a popular tool for testing APIs. It provides a user-friendly interface for sending requests to API endpoints and viewing responses. Here’s how you can use Postman to send invoices to the FastAPI service running at `http://localhost:8000/detect`.

### Setting Up Postman

1. **Download and Install Postman:**
   - If you haven't already installed Postman, you can download it from [Postman's official website](https://www.postman.com/downloads/).

2. **Open Postman:**
   - Launch the Postman application.

### Sending an Invoice to the API

1. **Create a New Request:**
   - Click the "New" button in the upper left corner and select "Request".
   - This opens a new tab where you can configure your request.

2. **Configure the Request:**
   - **HTTP Method**: Set the method to `POST` by selecting it from the dropdown menu next to the URL input field.
   - **Request URL**: Enter `http://localhost:8000/detect` as the URL.

3. **Add Multipart/Form-Data:**
   - Navigate to the 'Body' tab below the URL field.
   - Select 'form-data' from the list of options.
   - In the key field, type `file`. Select 'File' from the dropdown on the right (it defaults to 'Text').
   - Click the 'Select Files' button next to the newly created 'file' key and choose the invoice image you wish to upload from your file system.

4. **Send the Request:**
   - Click the 'Send' button.
   - Postman will process the request and display the response in the lower section of the interface.

### Reviewing the Response

After sending the request, you can view the API response in the 'Response' section at the bottom. The API should return a JSON object containing details about the detected elements in the invoice, such as the presence of signatures or stamps.

```json
[{
  "filename": "invoice.png",
  "detected_classes": {
    "signature": {
      "present": true,
      "detections": [
        {"box": [50, 50, 200, 200], "score": 0.95}
      ]
    },
    "stamp": {
      "present": false,
      "detections": []
    }
  }
}]

### Handling PDFs with Multiple Invoices

When sending a PDF file that contains multiple invoices, the API is capable of processing each page as a separate invoice. The response will include a JSON array where each element corresponds to an individual invoice's detection results. Here’s an example of how to send a PDF and what response you might expect:

#### Sending a PDF

To send a PDF containing multiple invoices:

1. **Follow the Same Steps as for Image Files**:
   - Open Postman.
   - Create a new POST request to `http://localhost:8000/detect`.
   - Set the body to 'form-data' and add a file field.
   - Upload your PDF file containing multiple invoices.

#### Example JSON Response for Multiple Invoices

The response for a PDF with multiple invoices will include an array of objects, each representing the detection results for each invoice:

```json
[
  {
    "filename": "invoice1_page1.png",
    "detected_classes": {
      "signature": {
        "present": true,
        "detections": [
          {"box": [50, 50, 200, 200], "score": 0.95}
        ]
      },
      "stamp": {
        "present": false,
        "detections": []
      }
    }
  },
  {
    "filename": "invoice2_page2.png",
    "detected_classes": {
      "signature": {
        "present": true,
        "detections": [
          {"box": [150, 150, 250, 250], "score": 0.92}
        ]
      },
      "stamp": {
        "present": true,
        "detections": [
          {"box": [100, 100, 160, 160], "score": 0.88}
        ]
      }
    }
  }
]






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
