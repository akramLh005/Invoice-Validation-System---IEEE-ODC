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
    curl

# Upgrade pip
RUN pip3 install --upgrade pip

# Install TorchServe and torch-model-archiver
RUN pip install torchserve torch-model-archiver

# Install additional dependencies
COPY requirements_torchserve.txt .
RUN pip install -r requirements_torchserve.txt

# Copy the raw model files and any other necessary files for archiving
COPY checkpoints/ /home/model-server/model-store/

# Copy the handler
COPY docker_handler.py /home/model-server/model-store/

# Ensure the path is corrected for the extra files if they are stored in a subdirectory
# If not, adjust the path accordingly. Below assumes they are directly under model-store.
RUN torch-model-archiver --model-name retinanet --version 1.0 --serialized-file /home/model-server/model-store/best_model.pth \
    --handler /home/model-server/model-store/docker_handler.py \
    --extra-files /home/model-server/model-store/retinanet_resnet50_fpn_v2_coco-5905b1c5.pth \
    --export-path /home/model-server/model-store/ -f

# Copy the config.properties file to the container
COPY config.properties /home/model-server/config.properties

# Copy the startup script to the container
COPY start-torchserve.sh /start-torchserve.sh

# Make the script executable
RUN chmod +x /start-torchserve.sh

# Set the script as the entrypoint
ENTRYPOINT ["/start-torchserve.sh"]
CMD ["serve"]
