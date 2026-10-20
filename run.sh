#!/bin/sh
# Safe local launcher for Strata.
#
# Serving defaults to the native sm_89/sm_120 engine and the RTX 5060 Ti only.
# Override STRATA_ENGINE or CUDA_VISIBLE_DEVICES when deliberately testing a
# different setup. Never point STRATA_ENGINE at a PTX-only sm_89 build on the
# RTX 5060 Ti: that configuration has reproduced Xid 79 in this workspace.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"

if [ ! -x bin/strata ]; then
  if command -v go >/dev/null 2>&1; then
    echo "Building native Strata binary..." >&2
    mkdir -p bin
    go build -o bin/strata ./cmd/strata
  else
    exec ./setup.sh "$@"
  fi
fi

if [ "${1-}" = serve ]; then
  ENGINE=${STRATA_ENGINE:-$ROOT/build-fat/strata}
  GPU=${CUDA_VISIBLE_DEVICES:-GPU-da4ac99c-634b-cb80-8ac6-499ec76fff39}

  if [ ! -x "$ENGINE" ]; then
    echo "error: native engine not found or not executable: $ENGINE" >&2
    echo "Set STRATA_ENGINE to a verified native engine binary." >&2
    exit 1
  fi

  case "$ENGINE" in
    *strata-sm89-ptx*)
      echo "error: refusing to run the known sm_89/PTX-only engine on the RTX 5060 Ti: $ENGINE" >&2
      exit 1
      ;;
  esac

  if command -v nvidia-smi >/dev/null 2>&1 && ! nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader >/dev/null 2>&1; then
    echo "error: nvidia-smi cannot reach GPU $GPU (driver lost device handle, or it's absent)." >&2
    echo "Serving now would just crash-loop. Recover the card first: sudo bash reset-gpu-soft.sh" >&2
    exit 1
  fi

  export CUDA_VISIBLE_DEVICES=$GPU
  shift
  exec bin/strata serve -binary-path "$ENGINE" "$@"
fi

exec bin/strata "$@"
