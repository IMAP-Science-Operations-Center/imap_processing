#!/usr/bin/env bash
set -e
echo "Building docker container"
docker build -t algorithm-example:latest .
echo "Finished."
echo
echo "Running container example."
docker run --rm -it \
  -e PROCESSING_DROPBOX=/opt/data/dropbox \
  --volume="$(pwd)/container_example_data:/opt/data" \
  algorithm-example:latest /opt/data/dropbox/input_manifest_20220923t000000.json
echo "Algorithm complete"