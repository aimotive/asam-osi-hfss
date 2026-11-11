#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."

git config --global --add safe.directory '*'
git submodule update --init

uv sync --locked
uv run python scripts/build_proto.py

bin/radar-dsp --help > /dev/null && echo "the modules in bin/ run: the notebooks can run them"

uv run python -c "from osi_hfss import pipeline; print(len(pipeline.build_all()), 'data files built')"
uv run pytest -q
