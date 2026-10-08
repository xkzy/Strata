#!/bin/sh
# Strata native runner: runs standalone Go binary if available, else builds it or falls back to setup.sh
cd "$(dirname "$0")" || exit 1
if [ -x bin/strata ]; then
  exec bin/strata "$@"
elif command -v go >/dev/null 2>&1; then
  echo "Building native Strata binary..."
  mkdir -p bin
  go build -o bin/strata ./cmd/strata
  exec bin/strata "$@"
else
  exec ./setup.sh "$@"
fi
