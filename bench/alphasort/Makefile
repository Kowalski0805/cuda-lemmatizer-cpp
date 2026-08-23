# alphasort-bench
#
#   make                 -> bench_synth  (self-contained stand-in trie)
#   make real            -> bench_real   (production trie from gpu_*.bin)
#   make ARCH=sm_86 ...  -> other GPUs (sm_89 = RTX 40xx, default here)
ARCH ?= sm_89
NVCC ?= /usr/local/cuda-12.8/bin/nvcc
NVCCFLAGS = -O3 -std=c++17 -arch=$(ARCH) --extended-lambda -lineinfo -Iinclude

SRC = src/bench.cu
HDR = include/common.cuh include/trie.cuh include/trie_real.cuh

all: bench_synth
real: bench_real

bench_synth: $(SRC) $(HDR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRC)

bench_real: $(SRC) $(HDR)
	$(NVCC) $(NVCCFLAGS) -DUSE_REAL_TRIE -o $@ $(SRC)

clean:
	rm -f bench_synth bench_real

.PHONY: all real clean
