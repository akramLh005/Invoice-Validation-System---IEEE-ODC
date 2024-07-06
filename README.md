# Invoice-Validation-System---IEEE-ODC

# Invoice validation 






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
