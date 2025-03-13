# Makefile for parallel Sobel filter implementations

# Compiler options
CC = mpicc
NVCC = nvcc
CFLAGS = -O3 -Wall -fopenmp 
LDFLAGS = -lgif -lm -fopenmp
NVCCFLAGS = -O3
CUDA_LIBS = -lcudart

# Directories
SRC_DIR = src
OBJ_DIR = obj
INCLUDE_DIR = include
CUDA_INCLUDE = /usr/local/cuda/include
CUDA_LIB = /usr/local/cuda/lib64

# Source files
SRC_FILES = $(wildcard $(SRC_DIR)/*.c)
CUDA_SRC = $(SRC_DIR)/sobel_cuda.cu

# Object files
OBJ_FILES = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_FILES))
CUDA_OBJ = $(OBJ_DIR)/sobel_cuda.o

# Executables
SOBELF = sobelf
SOBELF_OMP = sobelf_omp
SOBELF_CUDA = sobelf_cuda
SOBELF_HYBRID = sobelf_hybrid

# Default target
all: $(SOBELF) $(SOBELF_OMP) $(SOBELF_CUDA) $(SOBELF_HYBRID)

# Create object directory if it doesn't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Compile C files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Compile CUDA files
$(CUDA_OBJ): $(CUDA_SRC) | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(INCLUDE_DIR) -I$(CUDA_INCLUDE) -c $< -o $@

# Link sequential version
$(SOBELF): $(OBJ_DIR)/main.o
	$(CC) $^ -o $@ $(LDFLAGS)

# Link OpenMP version
$(SOBELF_OMP): $(OBJ_DIR)/main_omp.o
	$(CC) $^ -o $@ $(LDFLAGS)

# Link CUDA version
$(SOBELF_CUDA): $(OBJ_DIR)/main_cuda.o $(CUDA_OBJ)
	$(CC) $^ -o $@ $(LDFLAGS) -L$(CUDA_LIB) $(CUDA_LIBS)

# Link hybrid MPI+OpenMP+CUDA version
$(SOBELF_HYBRID): $(OBJ_DIR)/main_hybrid.o $(CUDA_OBJ)
	$(CC) $^ -o $@ $(LDFLAGS) -L$(CUDA_LIB) $(CUDA_LIBS)

# Clean target
clean:
	rm -rf $(OBJ_DIR) $(SOBELF) $(SOBELF_OMP) $(SOBELF_CUDA) $(SOBELF_HYBRID)

# Run tests
test: all
	./run_test.sh

.PHONY: all clean test
