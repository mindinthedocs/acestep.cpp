#!/usr/bin/env python3
# client-batch.py: test batching via ace-server
#
# 2 LM variations x 2 DiT variations = 4 MP3s
#
# POST /lm (batch_size=2 in JSON):
#   simple-batch.json -> 2 enriched requests
#
# POST /synth (batch_size=2 per request, multipart/mixed response):
#   request 0 -> server-batch00.mp3, server-batch01.mp3
#   request 1 -> server-batch10.mp3, server-batch11.mp3
#
# Start the server first: ./server.sh

import json
import sys
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8085"


def post_json(endpoint, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        URL + endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return resp.read(), resp.headers


def parse_multipart_mixed(data, content_type):
    """Parse multipart/mixed response into list of (headers_dict, body_bytes)."""
    # extract boundary from content-type header
    boundary = None
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundary = part[len("boundary="):].strip().encode()
            break
    if not boundary:
        raise ValueError("no boundary in content-type: " + content_type)

    delimiter = b"--" + boundary
    terminator = delimiter + b"--"
    parts = []

    for chunk in data.split(delimiter):
        if not chunk or chunk.startswith(b"--"):
            continue
        chunk = chunk.strip(b"\r\n")
        if not chunk:
            continue

        # split headers from body at first blank line
        sep = chunk.find(b"\r\n\r\n")
        if sep < 0:
            continue
        header_block = chunk[:sep].decode()
        body = chunk[sep + 4:]
        if body.endswith(b"\r\n"):
            body = body[:-2]

        headers = {}
        for line in header_block.split("\r\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip()] = v.strip()
        parts.append((headers, body))

    return parts


# Phase 1: LM generates 2 variations
try:
    with open("simple-batch.json") as f:
        request_json = json.load(f)
except FileNotFoundError:
    print("ERROR: simple-batch.json not found (run from the examples/ directory)")
    sys.exit(1)

try:
    print("POST /lm (batch_size=%d)..." % request_json.get("batch_size", 1))
    lm_data, _ = post_json("/lm", request_json)
except urllib.error.URLError as e:
    print("ERROR: cannot connect to %s (%s)" % (URL, e.reason))
    print("Start the server first: ./server.sh")
    sys.exit(1)
lm_results = json.loads(lm_data)
print("  -> %d variations" % len(lm_results))

# Phase 2: synth with batch_size=2 for each LM output
total = 0
for i, lm_req in enumerate(lm_results):
    lm_req["batch_size"] = 2
    print("POST /synth (LM variation %d, batch_size=2, seed=%d)..." % (i, lm_req["seed"]))

    body = json.dumps(lm_req).encode()
    req = urllib.request.Request(
        URL + "/synth",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        resp_data = resp.read()
        content_type = resp.headers.get("Content-Type", "")

    if "multipart/mixed" in content_type:
        # batch response: parse multipart/mixed
        parts = parse_multipart_mixed(resp_data, content_type)
        for j, (headers, mp3_data) in enumerate(parts):
            path = "server-batch%d%d.mp3" % (i, j)
            with open(path, "wb") as f:
                f.write(mp3_data)
            seed = headers.get("X-Seed", "?")
            dur = headers.get("X-Duration", "?")
            print("  -> %s (%s bytes, seed=%s, dur=%ss)" % (path, len(mp3_data), seed, dur))
            total += 1
    else:
        # single track response (batch_size was clamped to 1)
        path = "server-batch%d0.mp3" % i
        with open(path, "wb") as f:
            f.write(resp_data)
        print("  -> %s (%d bytes)" % (path, len(resp_data)))
        total += 1

print("Done: %d MP3s" % total)
