
SRC_DIR = src
HEADER_DIR = include
OBJ_DIR = obj


CC = gcc
MPICC = mpicc
NVCC = nvcc


CFLAGS    = -O3 -I$(HEADER_DIR) -fopenmp
LDFLAGS   = -lm -fopenmp

MPI_CFLAGS    = -O3 -I$(HEADER_DIR) -fopenmp
MPI_LDFLAGS   = -lm -fopenmp

CUDA_CFLAGS = -O3 -I$(HEADER_DIR) -diag-suppress=541
CUDA_LDFLAGS = -lm


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

ADAPT_OBJ = $(addprefix $(OBJ_DIR)/, \
            gif_io.o adaptive_filter.o sequential_filter.o mpi_domain_filter.o \
            cuda_omp_mpi_filter.o cuda_mpi_filter.o omp_filter.o cuda_filter.o \
            hybrid_openmp_mpi_filter.o dgif_lib.o cuda_common.o egif_lib.o \
            gif_err.o gif_font.o gif_hash.o gifalloc.o openbsd-reallocarray.o quantize.o)


adaptive_filter: $(ADAPT_OBJ)
	$(MPICC) $(MPI_CFLAGS) -o $@ $^ $(MPI_LDFLAGS) -L$(CUDA_ROOT)/lib64 -lcudart -lstdc++


$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)


$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<


$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(CUDA_CFLAGS) -c -o $@ $< -Xcompiler "-fexceptions"

clean:
	rm -f adaptive_filter $(OBJ_DIR)/*.o
