# Directories
SRC_DIR = src
HEADER_DIR = include
OBJ_DIR = obj

# Compilers
CC = gcc
MPICC = mpicc

# Flags (add -fopenmp as needed)
CFLAGS    = -O3 -I$(HEADER_DIR) -fopenmp
LDFLAGS   = -lm -fopenmp

MPI_CFLAGS    = -O3 -I$(HEADER_DIR) -fopenmp
MPI_LDFLAGS   = -lm -fopenmp

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

# Default target: build the sequential version
all: sobelf

sobelf: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)


sobelf_mpi: $(MPI_OBJ)
	$(MPICC) $(MPI_CFLAGS) -o $@ $^ $(MPI_LDFLAGS)

# Create the object directory if needed
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Generic rule for compiling C files (used for all files except sobelf_mpi.c)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# Specific rule for the MPI source file
$(OBJ_DIR)/sobelf_mpi.o: $(SRC_DIR)/sobelf_mpi.c | $(OBJ_DIR)
	$(MPICC) $(MPI_CFLAGS) -c -o $@ $<

clean:
	rm -f sobelf sobelf_mpi $(OBJ_DIR)/*.o
