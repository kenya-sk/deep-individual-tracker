#!/bin/bash
set -ex

rm -rf ./docs/build/*
poetry run sphinx-apidoc -e -f -o ./docs/source ./annotator
poetry run sphinx-build ./docs/source/ ./docs/build
