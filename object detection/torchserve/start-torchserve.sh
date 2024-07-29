#!/bin/bash
set -e

# Default model store directory
MODEL_STORE="/home/model-server/model-store"

if [[ "$1" = "serve" ]]; then
    shift 1
    # Start TorchServe with the specified model store and the retinanet model loaded by default
    torchserve --start --ncs --model-store ${MODEL_STORE} --models retinanet=retinanet.mar --ts-config /home/model-server/config.properties --disable-token-auth

    # Wait for TorchServe to fully start
    sleep 10
    # Initialize workers for the model ( Default = 2 )
    curl -X PUT "http://localhost:8081/models/retinanet?min_worker=2" || {
        echo "Failed to set workers"
        exit 1
    }
else
    eval "$@"
fi

# Prevent docker exit
tail -f /dev/null
