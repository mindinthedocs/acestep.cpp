#!/bin/bash

set -eu

# If you have both, uncomment one to use it
#GGML_BACKEND=CUDA0
#GGML_BACKEND=Vulkan0

./build/ace-server \
    --host 0.0.0.0 \
    --port 8085 \
    --models ./models \
    --loras ./loras \
    --max-batch 1
