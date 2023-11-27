#!/bin/bash
set -ex

rm -rf ./docs/build/*
sphinx-apidoc -e -f -o ./docs/source ./detector
sphinx-build ./docs/source/ ./docs/build
