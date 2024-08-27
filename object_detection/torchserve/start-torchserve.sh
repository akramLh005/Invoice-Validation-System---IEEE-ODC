#!/bin/bash
set -e

# Default model store directory
MODEL_STORE="/home/model-server/model-store"

# Determine the model type from an environment variable
MODEL_TYPE="${MODEL_TYPE:-PYTORCH}"  # Default to PYTORCH if not set

# Define the model name as 'retinanet' for both types
MODEL_NAME="retinanet"

# Start TorchServe command with consistent model naming
if [[ "$1" == "serve" ]]; then
    shift 1
    # Start TorchServe with the specified model store and the model loaded by default
    torchserve --start --ncs --model-store ${MODEL_STORE} --models ${MODEL_NAME}=${MODEL_NAME}.mar --ts-config /home/model-server/config.properties --disable-token-auth

    # Wait for TorchServe to fully start
    sleep 30
    # Initialize workers for the model (Default = 2)
    curl -X PUT "http://localhost:8081/models/${MODEL_NAME}?min_worker=2" || {
        echo "Failed to set workers"
        exit 1
    }
else
    eval "$@"
fi

# Prevent docker exit
tail -f /dev/null
