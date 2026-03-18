#!/bin/bash
# Roundtrip: audio -> understand -> SFT DiT -> 2 MP3 variations
#
# Usage: ./understand-roundtrip.sh input.wav (or input.mp3)
#
# understand:
# input -> ace-understand.json (audio codes + metadata)
#
# ace-synth:
# ace-understand.json -> output0.mp3 .. output1.mp3

set -eu

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input.wav|input.mp3>"
    exit 1
fi

input="$1"

../build/ace-understand \
    --src-audio "$input" \
    --dit ../models/acestep-v15-sft-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf \
    --lm ../models/acestep-5Hz-lm-4B-Q8_0.gguf \
    -o ace-understand.json

sed -i \
    -e 's/"audio_cover_strength": *[0-9.]*/"audio_cover_strength": 0.04/' \
    -e 's/"batch_size": *[0-9]*/"batch_size": 2/' \
    ace-understand.json

../build/ace-synth \
    --src-audio "$input" \
    --request ace-understand.json \
    --embedding ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-sft-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf
