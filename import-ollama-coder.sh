#!/bin/sh
# Import the local Qwen3.8-Flash-Next Coder GGUF into Ollama.
#
# Ollama reads a split GGUF from the first shard; all shards must remain in the
# same directory.  Override CODER_GGUF with the first shard when the model is
# stored somewhere else.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MODEL=${CODER_GGUF:-${ROOT}/../Strata-data/models/coder-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf}
NAME=${OLLAMA_MODEL_NAME:-qwen-coder-next}
CONTEXT=${OLLAMA_CONTEXT_LENGTH:-32768}

case "$CONTEXT" in
  ''|*[!0-9]*) echo "error: OLLAMA_CONTEXT_LENGTH must be an integer" >&2; exit 2 ;;
esac

if [ ! -f "$MODEL" ]; then
  echo "error: first Coder GGUF shard not found: $MODEL" >&2
  echo "Set CODER_GGUF to the path of ...-00001-of-00002.gguf" >&2
  exit 1
fi

DIR=$(dirname -- "$MODEL")
BASE=$(basename -- "$MODEL")
case "$BASE" in
  *-00001-of-00002.gguf) SECOND=${BASE%-00001-of-00002.gguf}-00002-of-00002.gguf ;;
  *)
    echo "error: CODER_GGUF must be the first -00001-of-00002.gguf shard" >&2
    exit 2
    ;;
esac

if [ ! -f "$DIR/$SECOND" ]; then
  echo "error: second Coder GGUF shard not found: $DIR/$SECOND" >&2
  exit 1
fi

# The Strata Coder is intentionally not a generic GGUF: its expert down
# matrices use private GGUF type 42 (Strata's 64-value/18-byte Q2_0 format).
# Ollama 0.34.x does not know that type and reports the misleading
# "size overflows" error while parsing it.  Do not upload the first shard
# before reporting the incompatibility.
case "$BASE" in
  *GSQ-RCO-IQ1_M*)
  echo "error: this Strata Coder GGUF uses private tensor type 42" >&2
  echo "Ollama cannot import it directly; the file is not corrupt." >&2
  echo "Use a standard GGUF quantization, or convert type 42 expert tensors" >&2
  echo "to a format supported by Ollama before importing." >&2
  exit 1
  ;;
esac

OLLAMA=${OLLAMA_BIN:-ollama}
if ! command -v "$OLLAMA" >/dev/null 2>&1; then
  echo "error: Ollama was not found; set OLLAMA_BIN to its executable" >&2
  exit 1
fi

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT HUP INT TERM
cat >"$TMP" <<EOF
FROM $MODEL
PARAMETER num_ctx $CONTEXT
PARAMETER temperature 0.2
EOF

echo "Importing $BASE and its adjacent split shard as Ollama model $NAME..." >&2
exec "$OLLAMA" create "$NAME" --file "$TMP"
