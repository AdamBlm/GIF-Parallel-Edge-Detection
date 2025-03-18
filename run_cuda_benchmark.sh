#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parameters
OPENMP_THREADS=${1:-4}
MPI_PROCESSES=${2:-4}
IMAGE_NAME=${3:-"giphy-3"}  # Default to a smaller test image

echo -e "${CYAN}========== CUDA+MPI Benchmark ===========${NC}"
echo -e "${CYAN}Setting up CUDA environment...${NC}"

# Make sure CUDA is available
if ! command -v nvcc >/dev/null 2>&1; then
    echo -e "${RED}CUDA compiler not found!${NC}"
    exit 1
fi

# Setup CUDA paths - needed to find shared libraries at runtime
CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/lib:$LD_LIBRARY_PATH

# CUDA-aware MPI settings
export CUDA_VISIBLE_DEVICES=0,1  # Use first two GPUs if available
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMPI_MCA_btl_openib_allow_ib=1
export MPICH_GPU_SUPPORT_ENABLED=1

# Test CUDA setup
echo -e "${CYAN}Testing CUDA library paths...${NC}"
if [ -f "$CUDA_PATH/lib64/libcudart.so" ]; then
    echo -e "${GREEN}Found CUDA runtime library at $CUDA_PATH/lib64/libcudart.so${NC}"
else
    echo -e "${YELLOW}Warning: Could not find CUDA runtime library at $CUDA_PATH/lib64/libcudart.so${NC}"
    # Try to find it on the system
    CUDA_SO=$(find /usr -name "libcudart.so*" 2>/dev/null | head -1)
    if [ -n "$CUDA_SO" ]; then
        CUDA_LIB_PATH=$(dirname "$CUDA_SO")
        echo -e "${GREEN}Found CUDA runtime at $CUDA_SO${NC}"
        echo -e "${CYAN}Adding $CUDA_LIB_PATH to LD_LIBRARY_PATH${NC}"
        export LD_LIBRARY_PATH=$CUDA_LIB_PATH:$LD_LIBRARY_PATH
    else
        echo -e "${RED}Could not find CUDA runtime library on the system.${NC}"
        echo -e "${RED}Please check your CUDA installation.${NC}"
        exit 1
    fi
fi

# Check if the CUDA version is running properly
echo -e "${CYAN}Testing CUDA runtime...${NC}"
INPUT_IMAGE="images/original/${IMAGE_NAME}.gif"
TEST_OUTPUT="images/processed/${IMAGE_NAME}_cuda_test.gif"

if [ ! -f "$INPUT_IMAGE" ]; then
    echo -e "${RED}Test image $INPUT_IMAGE does not exist!${NC}"
    exit 1
fi

echo -e "${CYAN}Running single-process CUDA test...${NC}"
./sobelf_cuda "$INPUT_IMAGE" "$TEST_OUTPUT"

if [ $? -ne 0 ]; then
    echo -e "${RED}Single-process CUDA test failed!${NC}"
    echo -e "${RED}Fix basic CUDA execution before trying MPI+CUDA${NC}"
    exit 1
else
    echo -e "${GREEN}Single-process CUDA test passed!${NC}"
    if [ -f "$TEST_OUTPUT" ]; then
        echo -e "${GREEN}Output file created successfully: $TEST_OUTPUT${NC}"
    else
        echo -e "${YELLOW}Warning: Output file not created even though process exited successfully${NC}"
    fi
fi

# Now run the MPI+CUDA test
echo -e "${CYAN}Running MPI+CUDA test with $MPI_PROCESSES processes...${NC}"
MPI_TEST_OUTPUT="images/processed/${IMAGE_NAME}_mpi_cuda_test.gif"

mpirun -np $MPI_PROCESSES ./sobelf_cuda "$INPUT_IMAGE" "$MPI_TEST_OUTPUT"

if [ $? -ne 0 ]; then
    echo -e "${RED}MPI+CUDA test failed!${NC}"
else
    echo -e "${GREEN}MPI+CUDA test passed!${NC}"
    if [ -f "$MPI_TEST_OUTPUT" ]; then
        echo -e "${GREEN}Output file created successfully: $MPI_TEST_OUTPUT${NC}"
    else
        echo -e "${YELLOW}Warning: Output file not created even though process exited successfully${NC}"
    fi
fi

# Run the full benchmark with proper environment variables
echo -e "${CYAN}Running full benchmark with CUDA environment set...${NC}"
echo -e "${CYAN}OpenMP Threads: $OPENMP_THREADS, MPI Processes: $MPI_PROCESSES${NC}"

DEBUG=2 ./benchmark_all.sh $OPENMP_THREADS $MPI_PROCESSES

echo -e "${CYAN}Benchmark complete.${NC}"
echo -e "${CYAN}=========================================${NC}" 
