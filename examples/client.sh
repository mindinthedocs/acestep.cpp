#!/bin/bash
# Test ace-server: LM enriches caption, synth renders to MP3.
# Start the server first (./server.sh), then run this.

set -eu

curl -sf http://127.0.0.1:8085/lm \
    -H "Content-Type: application/json" \
    -d @simple.json \
| jq '.[0]' > server-lm0.json

curl -sf http://127.0.0.1:8085/synth \
    -H "Content-Type: application/json" \
    -d @server-lm0.json \
    -o server0.mp3
