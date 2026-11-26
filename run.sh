#!/usr/bin/env bash
# Regenerate all Milestone 1 figures inside the dolfinx container (dolfinx 0.11, gmsh 4.15).
cd "$(dirname "$0")"
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp \
  -v "$PWD":/work -w /work dolfinx/dolfinx:stable python3 main.py "$@"
