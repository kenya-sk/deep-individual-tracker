#!/bin/bash
set -ex

rm -rf ./docs/build/*
poetry run sphinx-apidoc -e -f -o ./docs/source ./monitoring
poetry run sphinx-build ./docs/source/ ./docs/build
