#!/bin/bash
# Run the teacher's LM head on a float32 [N, hc, n_embd] row file (ncm/head_agreement.py prepare). env: BIN, GPU (default 1), WORK (dir with rows.f32)
R="/home/khing/Downloads/Strata copy"
D=/home/khing/Downloads/Strata-data
M=$D/models/coder-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M
exec timeout 1500 "${BIN:-$R/build-ncm/strata}" \
  --pack $D/packs/coder-iq1_m --native $M-00001-of-00002.gguf --ple-gguf $M-00002-of-00002.gguf \
  --expert-profile "$R/data/expert-profile-coder.bin" --expert-cache ${SLOTS:-256} --mmap-experts --gpu ${GPU:-1} --spec 2 --prefill ${CHUNK:-16} \
  --tokens 9707,11,1879,0 --max-new 1 --max-context 4096 \
  --head-eval $WORK/rows.f32 $WORK/argmax.i32 \
  > $WORK/eval.out 2> $WORK/eval.err
