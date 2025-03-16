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

# Sequential version sources and objects
SRC = dgif_lib.c \
      egif_lib.c \
      gif_err.c \
      gif_font.c \
      gif_hash.c \
      gifalloc.c \
      main.c \
      openbsd-reallocarray.c \
      quantize.c

OBJ = $(OBJ_DIR)/dgif_lib.o \
      $(OBJ_DIR)/egif_lib.o \
      $(OBJ_DIR)/gif_err.o \
      $(OBJ_DIR)/gif_font.o \
      $(OBJ_DIR)/gif_hash.o \
      $(OBJ_DIR)/gifalloc.o \
      $(OBJ_DIR)/main.o \
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
# Here we compile all the same GIF-related C sources along with the CUDA source.
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

# Specific rule for OpenMP source file
$(OBJ_DIR)/sobelf_omp.o: $(SRC_DIR)/sobelf_omp.c | $(OBJ_DIR)
	$(CC) $(OMP_CFLAGS) -c -o $@ $<

# Rule for compiling CUDA source files (for sobelf_cuda.cu and others)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(CUDA_CFLAGS) -c -o $@ $<

clean:
	rm -f sobelf sobelf_mpi sobelf_mpi_domain sobelf_cuda sobelf_omp $(OBJ_DIR)/*.o
