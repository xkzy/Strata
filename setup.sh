#!/bin/sh
# Strata: Native AI Engine, Mathematical Runtime & Web Server in Go & C++
cd "$(dirname "$0")" || exit 1

if [ -x bin/strata ]; then
  exec bin/strata "$@"
fi

if command -v go >/dev/null 2>&1; then
  echo "Building native Strata binary..."
  mkdir -p bin
  go build -o bin/strata ./cmd/strata
  go build -o bin/strata-server ./cmd/strata-server
  exec bin/strata "$@"
fi

echo "Please install Go (golang) to build bin/strata, or run the pre-built binary."
exit 1
