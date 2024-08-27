#!/bin/bash
set -e

# Default model store directory
MODEL_STORE="/home/model-server/model-store"

# Start TorchServe command with specific models
if [[ "$1" == "serve" ]]; then
    shift 1
    # Start TorchServe with the specified model store and disable auth for ease of testing
    torchserve --start --ncs --model-store ${MODEL_STORE} --ts-config /home/model-server/config.properties --disable-token-auth

    # Wait for TorchServe to fully start
    sleep 10

    # Register LMv3_machinewritten model with initial workers
    curl -X POST "http://localhost:8081/models?url=LMv3_machinewritten.mar&initial_workers=1&model_name=LMv3_machinewritten" || {
        echo "Failed to register LMv3_machinewritten"
        exit 1
    }

    # Register LMv3_handwritten model with initial workers
    curl -X POST "http://localhost:8081/models?url=LMv3_handwritten.mar&initial_workers=1&model_name=LMv3_handwritten" || {
        echo "Failed to register LMv3_handwritten"
        exit 1
    }
    
else
    # If not 'serve' command, pass all parameters to TorchServe for processing
    eval "$@"
fi

# Prevent docker exit
tail -f /dev/null

