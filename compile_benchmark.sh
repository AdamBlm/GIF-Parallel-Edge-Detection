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

# Function to compile and check result
compile_and_check() {
    local version=$1
    local make_command=$2
    local binary=$3
    
    echo -e "${CYAN}Compiling $version version...${NC}"
    # Ensure previous build artifact is removed
    if [ -f "./$binary" ]; then
        rm -f "./$binary"
    fi
    
    eval $make_command
    
    if [ ! -f "./$binary" ]; then
        echo -e "${RED}Failed to compile $version version${NC}"
        return 1
    else
        echo -e "${GREEN}Successfully compiled $version version${NC}"
        return 0
    fi
}

# Clean all binaries first
echo -e "${YELLOW}Cleaning all previous builds...${NC}"
# Sequential and OpenMP clean
make clean

# MPI clean
make -f Makefile.mpi clean

# CUDA specific clean
echo -e "${YELLOW}Cleaning CUDA-specific objects...${NC}"
rm -f sobelf_cuda obj/sobelf_cuda.o obj/gif_utils.o
rm -f *.cubin *.fatbin *.gpu *.ptx *.o *.cudafe* *.module_id *.cpp*.ii

# Hybrid clean
make -f Makefile.hybrid clean
make -f Makefile.hybrid_cuda clean

# Remove all binary files to ensure fresh compilation
rm -f sobelf sobelf_omp sobelf_mpi sobelf_omp_mpi sobelf_cuda sobelf_mpi_omp_cuda

errors=0

# Compile sequential version
compile_and_check "sequential" "make sobelf" "sobelf" || ((errors++))

# Compile OpenMP version
compile_and_check "OpenMP" "make sobelf_omp" "sobelf_omp" || ((errors++))

# Compile MPI-only version
compile_and_check "MPI-only" "make Makefile.mpi sobelf_mpi" "sobelf_mpi" || ((errors++))

# Compile MPI+OpenMP version
compile_and_check "MPI+OpenMP" "make -f Makefile.mpi sobelf_omp_mpi" "sobelf_omp_mpi" || ((errors++))

# Check if CUDA is available on this system
if command -v nvcc >/dev/null 2>&1; then
    # Compile CUDA version with more verbose output
    echo -e "${YELLOW}Found CUDA compiler (nvcc). Compiling CUDA versions...${NC}"
    
    # Compile CUDA version
    compile_and_check "CUDA" "make -f Makefile.cuda sobelf_cuda" "sobelf_cuda" || ((errors++))
    
    # Compile MPI+OpenMP+CUDA hybrid version
    compile_and_check "MPI+OpenMP+CUDA hybrid" "make -f Makefile.hybrid_cuda sobelf_mpi_omp_cuda" "sobelf_mpi_omp_cuda" || ((errors++))
else
    echo -e "${YELLOW}CUDA compiler (nvcc) not found. Skipping CUDA versions.${NC}"
    echo -e "${YELLOW}To compile CUDA versions, please ensure CUDA is installed and nvcc is in your PATH.${NC}"
fi

# Summary
if [ $errors -eq 0 ]; then
    echo -e "\n${GREEN}All versions compiled successfully!${NC}"
    echo -e "${BLUE}You can now run the benchmark with:${NC}"
    echo -e "  ./benchmark_all.sh [openmp_threads] [mpi_processes]"
else
    echo -e "\n${RED}Compilation completed with $errors error(s).${NC}"
    echo -e "${YELLOW}Please fix the issues above before running the benchmark.${NC}"
    exit 1
fi 