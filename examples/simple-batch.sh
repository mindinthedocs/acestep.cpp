#!/bin/bash
# Generate 2 LM variations, then 2 DiT variations per LM output = 4 total MP3s
#
# LM phase (batch_size=2 in simple-batch.json):
# simple-batch.json -> simple-batch0.json, simple-batch1.json
#
# DiT phase (batch_size=2 set via sed):
# simple-batch0.json -> simple-batch00.mp3, simple-batch01.mp3
# simple-batch1.json -> simple-batch10.mp3, simple-batch11.mp3

set -eu

# Phase 1: LM generates 2 variations (different lyrics/codes/metas)
../build/ace-lm \
    --request simple-batch.json \
    --lm ../models/acestep-5Hz-lm-4B-Q8_0.gguf

# Set batch_size=2 for DiT noise variations (LM consumed it and reset to 1)
sed -i 's/"batch_size": *[0-9]*/"batch_size": 2/' simple-batch0.json simple-batch1.json

# Phase 2: DiT+VAE renders 2 noise variations per LM output
../build/ace-synth \
    --request simple-batch0.json simple-batch1.json \
    --embedding ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-turbo-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf
