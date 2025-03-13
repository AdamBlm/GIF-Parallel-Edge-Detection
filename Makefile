SRC_DIR=src
HEADER_DIR=include
OBJ_DIR=obj

CC=mpicc
NVCC = nvcc
CFLAGS = -O3 -Wall -fopenmp -I$(HEADER_DIR)
LDFLAGS = -lm -fopenmp
NVCCFLAGS = -O3
CUDA_LIBS = -lcudart

# Add -fopenmp to both CFLAGS (for compilation) and LDFLAGS (for linking)
#CFLAGS = -O3 -I$(HEADER_DIR) -fopenmp
#LDFLAGS = -lm -fopenmp

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

all: sobelf

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $^

sobelf: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f sobelf $(OBJ)
