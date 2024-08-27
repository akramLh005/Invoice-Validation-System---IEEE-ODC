# Use an official PyTorch image as a parent image
FROM pytorch/pytorch:latest

# Set the working directory
WORKDIR /home/model-server/

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    openjdk-11-jdk \
    wget \
    curl \
    bash

# Upgrade pip
RUN pip3 install --upgrade pip

# Install TorchServe and torch-model-archiver
RUN pip install torchserve torch-model-archiver onnxruntime nvgpu

# Install additional dependencies
COPY requirements_torchserve.txt .
RUN pip install -r requirements_torchserve.txt

# Copy the raw model files and any other necessary files for archiving
COPY checkpoints/ /home/model-server/model-store/

# Set an environment variable for the model type (ONNX or PYTORCH)
ARG MODEL_TYPE=PYTORCH
ENV MODEL_TYPE=${MODEL_TYPE}

# Copy the handler based on the model type
COPY docker_handler.py /home/model-server/model-store/docker_handler.py
COPY docker_onnx_handler.py /home/model-server/model-store/docker_onnx_handler.py

# Archive the model based on the provided model type
RUN if [ "$MODEL_TYPE" = "onnx" ]; then \
        echo "Archiving ONNX model..."; \
        torch-model-archiver --model-name retinanet --version 1.0 --serialized-file /home/model-server/model-store/modelPrunned.onnx \
        --handler /home/model-server/model-store/docker_onnx_handler.py \
        --export-path /home/model-server/model-store/ -f; \
    else \
        echo "Archiving PyTorch model..."; \
        torch-model-archiver --model-name retinanet --version 1.0 --serialized-file /home/model-server/model-store/best_model.pth \
        --handler /home/model-server/model-store/docker_handler.py \
        --extra-files /home/model-server/model-store/retinanet_resnet50_fpn_v2_coco-5905b1c5.pth \
        --export-path /home/model-server/model-store/ -f; \
    fi

# Copy the config.properties file to the container
COPY config.properties /home/model-server/config.properties

# Copy the startup script to the container
COPY start-torchserve.sh /start-torchserve.sh

# Make the script executable
RUN chmod +x /start-torchserve.sh

# Set the script as the entrypoint
ENTRYPOINT ["/start-torchserve.sh"]
CMD ["serve"]
