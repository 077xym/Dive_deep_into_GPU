# Paths
TORCH_INCLUDE1 = /home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/include
TORCH_INCLUDE2 = /home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/include/torch/csrc/api/include
PYTHON_INCLUDE = /home/zeus/miniconda3/envs/cloudspace/include/python3.10
TORCH_LIB = /home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/torch/lib
CUDA_HOME = /usr/local/cuda

NVCC = $(CUDA_HOME)/bin/nvcc
NVCCFLAGS = -std=c++17 -O3

INCLUDES = -I$(TORCH_INCLUDE1) -I$(TORCH_INCLUDE2) -I$(PYTHON_INCLUDE) -I$(CUDA_HOME)/include # header files
LIBS = -L$(TORCH_LIB) -ltorch -ltorch_cpu -ltorch_cuda -lc10 # libraries, the actual implementation to link
RPATH = -Xlinker -rpath -Xlinker $(TORCH_LIB) # find .so to embed during runtime

NAIVE_TARGET = better_attention
NAIVE_SRC = better_attention.cu

naive:
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(NAIVE_SRC) -o $(NAIVE_TARGET) $(LIBS) $(RPATH)

clean:
	rm -f $(TARGET) *.o
