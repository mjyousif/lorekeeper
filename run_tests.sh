#!/usr/bin/env bash
# Run tests with proper virtual environment activation

set -e

cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Install test dependencies if not already installed
pip install -q -r tests/requirements-test.txt 2>/dev/null || true

# Run pytest
pytest "$@"