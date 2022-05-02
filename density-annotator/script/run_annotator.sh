#!/bin/bash
set -ex

echo "Running Density Annotator..."
poetry run python src/run_annotator.py

