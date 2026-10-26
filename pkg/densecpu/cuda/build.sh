#!/bin/sh
# Builds libdensegpu.so for the RTX 40-series (sm_89) and 50-series (sm_120) cards. CUDA 13.2 does not accept gcc 16,
# so the host compiler is gcc 15. Then: go test -tags cuda ./pkg/densecpu
set -e
cd "$(dirname "$0")"
CUDA=${CUDA_HOME:-/usr/local/cuda-13.2}
"$CUDA/bin/nvcc" -O3 -ccbin "${HOST_CXX:-g++-15}" -shared -Xcompiler -fPIC,-B/usr/lib64 \
  -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 \
  densegpu.cu -o libdensegpu.so
