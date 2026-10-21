#!/bin/sh
# Local launcher for Strata: two-GPU A/B test mode.
#
# Serving runs the native sm_89/sm_120 engine on both cards (RTX 5060 Ti =
# device 0, RTX 4060 = device 1, PCI bus order). STRATA_AB picks one test
# variable per run, to find which two-GPU path loses the card:
#   STRATA_AB=none    (default) the normal two-GPU setup
#   STRATA_AB=ring    the 4060's prompt ring off       (STRATA_PF_PEER_STREAM=0)
#   STRATA_AB=direct  no zero-copy peer results in the reply (STRATA_PEER_DIRECT=0)
# The model config's "env" is applied after this script's, so a key set there
# wins over the same key set here.
# Never point STRATA_ENGINE at a PTX-only sm_89 build on the RTX 5060 Ti: that
# configuration has reproduced Xid 79 in this workspace.
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

  case "${STRATA_AB:-none}" in
    none) ;;
    ring) export STRATA_PF_PEER_STREAM=0 ;;
    direct) export STRATA_PEER_DIRECT=0 ;;
    *)
      echo "error: STRATA_AB must be none, ring or direct (got: $STRATA_AB)" >&2
      exit 1
      ;;
  esac

  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
  echo "run.sh: engine $ENGINE, GPUs $CUDA_VISIBLE_DEVICES, A/B ${STRATA_AB:-none}" >&2
  shift
  exec bin/strata serve -binary-path "$ENGINE" "$@"
fi

exec bin/strata "$@"
