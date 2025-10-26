# Directories
SRC_DIR = src
HEADER_DIR = include
OBJ_DIR = obj

# Compilers
CC = gcc
MPICC = mpicc
NVCC = nvcc

# Flags (add -fopenmp as needed)
CFLAGS    = -O3 -I$(HEADER_DIR) -fopenmp
LDFLAGS   = -lm -fopenmp

MPI_CFLAGS    = -O3 -I$(HEADER_DIR) -fopenmp
MPI_LDFLAGS   = -lm -fopenmp

# For CUDA, add diag-suppress for warning 541.
CUDA_CFLAGS = -O3 -I$(HEADER_DIR) -diag-suppress=541
CUDA_LDFLAGS = -lm

# OpenMP flags
OMP_CFLAGS = -O3 -I$(HEADER_DIR) -fopenmp
OMP_LDFLAGS = -lm -fopenmp

# Common I/O sources and objects
GIF_IO_SRC = gif_io.c
GIF_IO_OBJ = $(OBJ_DIR)/gif_io.o


# Sequential version sources and objects
SRC = dgif_lib.c \
      egif_lib.c \
      gif_err.c \
      gif_font.c \
      gif_hash.c \
      gifalloc.c \
      sequential.c \
      openbsd-reallocarray.c \
      quantize.c

OBJ = $(OBJ_DIR)/dgif_lib.o \
      $(OBJ_DIR)/egif_lib.o \
      $(OBJ_DIR)/gif_err.o \
      $(OBJ_DIR)/gif_font.o \
      $(OBJ_DIR)/gif_hash.o \
      $(OBJ_DIR)/gifalloc.o \
      $(OBJ_DIR)/sequential.o \
      $(OBJ_DIR)/openbsd-reallocarray.o \
      $(OBJ_DIR)/quantize.o

# MPI version sources and objects (original MPI version)
MPI_SRC = dgif_lib.c \
          egif_lib.c \
          gif_err.c \
          gif_font.c \
          gif_hash.c \
          gifalloc.c \
          sobelf_mpi.c \
          openbsd-reallocarray.c \
          quantize.c

MPI_OBJ = $(OBJ_DIR)/dgif_lib.o \
          $(OBJ_DIR)/egif_lib.o \
          $(OBJ_DIR)/gif_err.o \
          $(OBJ_DIR)/gif_font.o \
          $(OBJ_DIR)/gif_hash.o \
          $(OBJ_DIR)/gifalloc.o \
          $(OBJ_DIR)/sobelf_mpi.o \
          $(OBJ_DIR)/openbsd-reallocarray.o \
          $(OBJ_DIR)/quantize.o

# Domain-decomposed MPI version sources and objects
MPI_DOMAIN_SRC = dgif_lib.c \
                 egif_lib.c \
                 gif_err.c \
                 gif_font.c \
                 gif_hash.c \
                 gifalloc.c \
                 sobelf_mpi_domain.c \
                 openbsd-reallocarray.c \
                 quantize.c

MPI_DOMAIN_OBJ = $(OBJ_DIR)/dgif_lib.o \
                 $(OBJ_DIR)/egif_lib.o \
                 $(OBJ_DIR)/gif_err.o \
                 $(OBJ_DIR)/gif_font.o \
                 $(OBJ_DIR)/gif_hash.o \
                 $(OBJ_DIR)/gifalloc.o \
                 $(OBJ_DIR)/sobelf_mpi_domain.o \
                 $(OBJ_DIR)/openbsd-reallocarray.o \
                 $(OBJ_DIR)/quantize.o

# CUDA version sources and objects.
CUDA_SRC = dgif_lib.c \
           egif_lib.c \
           gif_err.c \
           gif_font.c \
           gif_hash.c \
           gifalloc.c \
           sobelf_cuda.cu \
           openbsd-reallocarray.c \
           quantize.c

CUDA_OBJ = $(OBJ_DIR)/dgif_lib.o \
           $(OBJ_DIR)/egif_lib.o \
           $(OBJ_DIR)/gif_err.o \
           $(OBJ_DIR)/gif_font.o \
           $(OBJ_DIR)/gif_hash.o \
           $(OBJ_DIR)/gifalloc.o \
           $(OBJ_DIR)/sobelf_cuda.o \
           $(OBJ_DIR)/openbsd-reallocarray.o \
           $(OBJ_DIR)/quantize.o

# OpenMP version sources and objects
OMP_SRC = dgif_lib.c \
          egif_lib.c \
          gif_err.c \
          gif_font.c \
          gif_hash.c \
          gifalloc.c \
          sobelf_omp.c \
          openbsd-reallocarray.c \
          quantize.c

OMP_OBJ = $(OBJ_DIR)/dgif_lib.o \
          $(OBJ_DIR)/egif_lib.o \
          $(OBJ_DIR)/gif_err.o \
          $(OBJ_DIR)/gif_font.o \
          $(OBJ_DIR)/gif_hash.o \
          $(OBJ_DIR)/gifalloc.o \
          $(OBJ_DIR)/sobelf_omp.o \
          $(OBJ_DIR)/openbsd-reallocarray.o \
          $(OBJ_DIR)/quantize.o

# Adaptive filter sources and objects
ADAPT_SRC = adaptive_filter.c \
            sequential_filter.c \
            mpi_domain_filter.c \
            cuda_common.cu \
            omp_filter.c \
            cuda_filter.cu \
            cuda_mpi_filter.cu \
            cuda_omp_mpi_filter.cu \
            hybrid_openmp_mpi_filter.c \
            gif_io.c \
            dgif_lib.c \
            egif_lib.c \
            gif_err.c \
            gif_font.c \
            gif_hash.c \
            gifalloc.c \
            openbsd-reallocarray.c \
            quantize.c

ADAPT_OBJ = $(addprefix $(OBJ_DIR)/, gif_io.o adaptive_filter.o sequential_filter.o mpi_domain_filter.o cuda_omp_mpi_filter.o cuda_mpi_filter.o omp_filter.o cuda_filter.o hybrid_openmp_mpi_filter.o dgif_lib.o cuda_common.o egif_lib.o gif_err.o gif_font.o gif_hash.o gifalloc.o openbsd-reallocarray.o quantize.o)

# Default target: build the sequential version
all: sobelf

sobelf: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

sobelf_mpi: $(MPI_OBJ)
	$(MPICC) $(MPI_CFLAGS) -o $@ $^ $(MPI_LDFLAGS)

sobelf_mpi_domain: $(MPI_DOMAIN_OBJ)
	$(MPICC) $(MPI_CFLAGS) -o $@ $^ $(MPI_LDFLAGS)

sobelf_cuda: $(CUDA_OBJ)
	$(NVCC) $(CUDA_CFLAGS) -o $@ $^ $(CUDA_LDFLAGS)

sobelf_omp: $(OMP_OBJ)
	$(CC) $(OMP_CFLAGS) -o $@ $^ $(OMP_LDFLAGS)

adaptive_filter: $(ADAPT_OBJ)
	$(MPICC) $(MPI_CFLAGS) -o $@ $^ $(MPI_LDFLAGS) -L$(CUDA_ROOT)/lib64 -lcudart -lstdc++

# Create the object directory if needed
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Generic rule for compiling C files (for sequential/MPI targets)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# Specific rule for MPI source files
$(OBJ_DIR)/sobelf_mpi.o: $(SRC_DIR)/sobelf_mpi.c | $(OBJ_DIR)
	$(MPICC) $(MPI_CFLAGS) -c -o $@ $<

$(OBJ_DIR)/sobelf_mpi_domain.o: $(SRC_DIR)/sobelf_mpi_domain.c | $(OBJ_DIR)
	$(MPICC) $(MPI_CFLAGS) -c -o $@ $<

# Specific rule for mpi_domain_filter.c
$(OBJ_DIR)/mpi_domain_filter.o: $(SRC_DIR)/mpi_domain_filter.c | $(OBJ_DIR)
	$(MPICC) $(MPI_CFLAGS) -c -o $@ $<

# Specific rule for OpenMP source file
$(OBJ_DIR)/sobelf_omp.o: $(SRC_DIR)/sobelf_omp.c | $(OBJ_DIR)
	$(CC) $(OMP_CFLAGS) -c -o $@ $<

$(OBJ_DIR)/gif_io.o: $(SRC_DIR)/gif_io.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# Specific rule for CUDA filter source file
$(OBJ_DIR)/cuda_filter.o: $(SRC_DIR)/cuda_filter.cu | $(OBJ_DIR)
	$(NVCC) $(CUDA_CFLAGS) -c -o $@ $<

# Specific rule for hybrid filter source file
$(OBJ_DIR)/hybrid_openmp_mpi_filter.o: $(SRC_DIR)/hybrid_openmp_mpi_filter.c | $(OBJ_DIR)
	$(MPICC) $(MPI_CFLAGS) -c -o $@ $<

$(OBJ_DIR)/cuda_common.o: $(SRC_DIR)/cuda_common.cu | $(OBJ_DIR)
	$(NVCC) $(CUDA_CFLAGS) -I$(MPI_ROOT)/include -c -o $@ $< -Xcompiler "-fexceptions"

# Specific rule for adaptive filter source file (include CUDA headers)
$(OBJ_DIR)/adaptive_filter.o: $(SRC_DIR)/adaptive_filter.c | $(OBJ_DIR)
	$(MPICC) $(MPI_CFLAGS) -I$(CUDA_ROOT)/include -c -o $@ $<

# Specific rule for CUDA+MPI filter source file
$(OBJ_DIR)/cuda_mpi_filter.o: $(SRC_DIR)/cuda_mpi_filter.cu | $(OBJ_DIR)
	$(NVCC) $(CUDA_CFLAGS) -I$(MPI_ROOT)/include -c -o $@ $< -Xcompiler "-fexceptions"

$(OBJ_DIR)/cuda_omp_mpi_filter.o: $(SRC_DIR)/cuda_omp_mpi_filter.cu | $(OBJ_DIR)
	$(NVCC) $(CUDA_CFLAGS) -I$(MPI_ROOT)/include -c -o $@ $< -Xcompiler "-fexceptions"


# Rule for compiling CUDA source files (for sobelf_cuda.cu and others)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(CUDA_CFLAGS) -c -o $@ $<

clean:
	rm -f sobelf sobelf_mpi sobelf_mpi_domain sobelf_cuda sobelf_omp adaptive_filter $(OBJ_DIR)/*.o