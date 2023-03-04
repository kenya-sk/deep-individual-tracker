#!/bin/bash
set -ex

echo "Running Density Annotator..."
poetry run python annotator/run_annotator.py

