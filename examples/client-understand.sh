#!/bin/bash
# Roundtrip via ace-server: audio -> understand -> synth -> MP3
#
# Usage: ./client-understand.sh input.wav (or input.mp3)
#
# POST /understand (multipart):
# input -> server-understand.json (audio codes + metadata)
#
# POST /synth (multipart):
# server-understand.json + input -> server-understand.mp3
#
# Start the server first (./server.sh).

set -eu

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input.wav|input.mp3>"
    exit 1
fi

input="$1"

curl -sf http://127.0.0.1:8085/understand \
    -F "audio=@${input}" \
    -o server-understand.json

sed -i \
    -e 's/"audio_cover_strength": *[0-9.]*/"audio_cover_strength": 0.04/' \
    server-understand.json

curl -sf http://127.0.0.1:8085/synth \
    -F "request=@server-understand.json" \
    -F "audio=@${input}" \
    -o server-understand.mp3
