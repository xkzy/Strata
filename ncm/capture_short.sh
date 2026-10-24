#!/bin/bash
# Short teacher capture: unmodified teacher (no control vectors / abliteration / MTP), 4-token prompt, 2 new tokens.
R="/home/khing/Downloads/Strata copy"
D=/home/khing/Downloads/Strata-data
O=/home/khing/.claude/jobs/a719c9cc/tmp
M=$D/models/coder-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M
exec timeout 1500 "$R/build-fat/strata" \
  --pack $D/packs/coder-iq1_m --native $M-00001-of-00002.gguf --ple-gguf $M-00002-of-00002.gguf \
  --expert-profile "$R/data/expert-profile-coder.bin" --expert-cache ${SLOTS:-256} --mmap-experts --gpu ${GPU:-1} --spec 2 --prefill ${CHUNK:-16} \
  --tokens 9707,11,1879,0 --max-new 2 --max-context 4096 \
  --dump-residual $O/residual.bin --dump-logits $O/logits.bin --dump-routing $O/routing.bin \
  > $O/cap.out 2> $O/cap.err
