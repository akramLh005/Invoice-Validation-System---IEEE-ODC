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
    libpng-dev libjpeg-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install TorchServe and torch-model-archiver
RUN pip install torchserve torch-model-archiver onnxruntime nvgpu

# Copy the requirements file and install Python dependencies
COPY requirements_torchserve.txt .
RUN pip install -r requirements_torchserve.txt

# Create model-store directory
RUN mkdir -p /home/model-server/model-store


# Copy the raw model files and any other necessary files for archiving
COPY ./checkpoints/layoutlmv3_machine_written.onnx /home/model-server/model-store/layoutlmv3_machine_written.onnx
COPY ./checkpoints/layoutlmv3_hand_written.onnx /home/model-server/model-store/layoutlmv3_hand_written.onnx


# Copy handlers
COPY docker_onnx_handler_machine_written.py /home/model-server/model-store/docker_onnx_handler_machine_written.py
COPY docker_onnx_handler_hand_written.py /home/model-server/model-store/docker_onnx_handler_hand_written.py


# Archive LMv3_machinewritten model
RUN torch-model-archiver --model-name LMv3_machinewritten --version 1.0 \
    --serialized-file /home/model-server/model-store/layoutlmv3_machine_written.onnx \
    --handler /home/model-server/model-store/docker_onnx_handler_machine_written.py \
    --export-path /home/model-server/model-store/ -f

# Archive LMv3_handwritten model
RUN torch-model-archiver --model-name LMv3_handwritten --version 1.0 \
    --serialized-file /home/model-server/model-store/layoutlmv3_hand_written.onnx \
    --handler /home/model-server/model-store/docker_onnx_handler_hand_written.py \
    --export-path /home/model-server/model-store/ -f

# Copy the config.properties and start script
COPY config.properties start-torchserve.sh /home/model-server/

# Make the start script executable
RUN chmod +x /home/model-server/start-torchserve.sh

# Set the script as the entrypoint
ENTRYPOINT ["/home/model-server/start-torchserve.sh"]
CMD ["serve"]
