#!/bin/sh
# Entry point for node containers. Cleans up any cached Python bytecode that
# might have been created on the host before starting the requested command.

set -e

cleanup_pycache() {
    find "$1" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
    find "$1" -name '*.pyc' -delete 2>/dev/null || true
}

cleanup_pycache /app/src
cleanup_pycache /app/scripts

exec "$@"
