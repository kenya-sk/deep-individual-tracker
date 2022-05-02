#!/bin/bash
set -e

IMAGE_NAME="density-annotator"
BUILD_STATUS=false

if [ "$(docker image ls -q "${IMAGE_NAME}")" ]; then
    echo "Image ${NAME_IMAGE} already exist."
    BUILD_STATUS=true
fi

if "${BUILD_STATUS}" && [ "$1" = "ignore" ]; then
  echo "Container is not build. Use an image that already exists."
  docker run "${IMAGE_NAME}"
else
  echo "Build a new image."
  docker build -t "${IMAGE_NAME}" .
  docker run "${IMAGE_NAME}"
fi

