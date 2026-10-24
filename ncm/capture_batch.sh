#!/bin/bash
# One teacher capture per prompt file (unmodified teacher). Stops at the first failure; never retries.
# usage: BIN=build-ncm/strata PROMPTS=dir OUT=dir [GPU=1 SLOTS=256 CHUNK=64 NEW=96] bash ncm/capture_batch.sh
set -u
R="/home/khing/Downloads/Strata copy"
D=/home/khing/Downloads/Strata-data
M=$D/models/coder-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M
: "${BIN:?}" "${PROMPTS:?}" "${OUT:?}"
mkdir -p "$OUT"
for f in "$PROMPTS"/*.txt; do
  n=$(basename "$f" .txt)
  [ -s "$OUT/final_r_$n.bin" ] && continue  # resumable
  timeout 900 "$BIN" \
    --pack $D/packs/coder-iq1_m --native $M-00001-of-00002.gguf --ple-gguf $M-00002-of-00002.gguf \
    --expert-profile "$R/data/expert-profile-coder.bin" --expert-cache ${SLOTS:-256} --mmap-experts --gpu ${GPU:-1} \
    --spec 2 --prefill ${CHUNK:-64} --tokens-file "$f" --max-new ${NEW:-96} --max-context 4096 \
    --dump-final-r "$OUT/final_r_$n.bin" --dump-routing "$OUT/routing_$n.bin" \
    > "$OUT/run_$n.out" 2> "$OUT/run_$n.err" || { echo "prompt $n failed (exit $?)"; exit 1; }
  echo "prompt $n done"
done
