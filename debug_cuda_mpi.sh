#!/bin/bash

# Source environment variables
source set_env.sh

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Set parameters
MPI_PROCESSES=${1:-2}  # Default to 2 processes
IMAGE=${2:-"images/original/Campusplan-Hausnr.gif"}  # Default test image
OUTPUT="images/processed/debug_output.gif"

echo -e "${CYAN}================== CUDA+MPI Debug Mode ==================${NC}"
echo -e "${CYAN}Using ${MPI_PROCESSES} MPI processes${NC}"
echo -e "${CYAN}Input image: ${IMAGE}${NC}"
echo -e "${CYAN}Output file: ${OUTPUT}${NC}"

# Create logs directory
mkdir -p debug_logs

# First verify if CUDA is available
if command -v nvcc >/dev/null 2>&1; then
    echo -e "${GREEN}CUDA compiler (nvcc) is available${NC}"
else
    echo -e "${RED}CUDA compiler (nvcc) not found!${NC}"
    exit 1
fi

# Check for CUDA devices
if nvidia-smi >/dev/null 2>&1; then
    echo -e "${GREEN}NVIDIA GPUs detected:${NC}"
    nvidia-smi -L
else
    echo -e "${RED}No NVIDIA GPUs detected!${NC}"
    exit 1
fi

# Compile with debug flags if needed
echo -e "${CYAN}Compiling CUDA+MPI version with debug flags...${NC}"
make clean -f Makefile.cuda
make -f Makefile.cuda CFLAGS="-g -DSOBELF_DEBUG=1" NVCC_FLAGS="-g -G -DSOBELF_DEBUG=1"

if [ ! -f "./sobelf_cuda" ]; then
    echo -e "${RED}Failed to compile CUDA version${NC}"
    exit 1
fi

# Environment variables that might help
export CUDA_VISIBLE_DEVICES=0,1  # Use first two GPUs if available
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMPI_MCA_btl_openib_allow_ib=1
export MPICH_GPU_SUPPORT_ENABLED=1

# Run with detailed MPI debugging output
echo -e "${CYAN}Running CUDA+MPI version with enhanced diagnostics...${NC}"
echo -e "${CYAN}Command: mpirun -np ${MPI_PROCESSES} --mca btl_base_verbose 10 ./sobelf_cuda \"${IMAGE}\" \"${OUTPUT}\"${NC}"

# Save full output to log file and display
LOG_FILE="debug_logs/cuda_mpi_debug.log"
mpirun -np ${MPI_PROCESSES} --mca btl_base_verbose 10 ./sobelf_cuda "${IMAGE}" "${OUTPUT}" |& tee ${LOG_FILE}

# Check result
if [ -f "${OUTPUT}" ]; then
    echo -e "${GREEN}Output file created successfully${NC}"
    ls -la "${OUTPUT}"
else
    echo -e "${RED}Output file was not created!${NC}"
fi

# Check for common error messages in the log
echo -e "\n${CYAN}Checking log for common CUDA+MPI errors:${NC}"
grep -i "error" ${LOG_FILE}
grep -i "fail" ${LOG_FILE}
grep -i "cuda" ${LOG_FILE} | grep -i "device"

echo -e "\n${CYAN}=======================================================${NC}"
echo -e "${YELLOW}Full debug log saved to: ${LOG_FILE}${NC}"
echo -e "${YELLOW}To run with different parameters: $0 [mpi_processes] [input_image]${NC}" 
