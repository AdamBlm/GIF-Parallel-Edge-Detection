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

# Debug level
DEBUG_MODE=${DEBUG:-0}

# Function to run a command with error handling
run_with_timeout() {
    local cmd="$1"
    local output_file="$2"
    local timeout=${3:-180}  # Default 3 minutes timeout
    local name="$4"
    
    if [ $DEBUG_MODE -ge 1 ]; then
        echo -e "${CYAN}Running: $cmd${NC}"
    fi
    
    # Create a temporary file for output
    local temp_output=$(mktemp)
    
    # Run the command with timeout
    timeout $timeout bash -c "$cmd" > "$temp_output" 2>&1
    local status=$?
    
    # Check execution status
    if [ $status -eq 124 ]; then
        echo -e "${RED}Command timed out after ${timeout}s: $name${NC}"
        echo "TIMEOUT" > "$output_file"
        if [ $DEBUG_MODE -ge 1 ]; then
            echo "Last output before timeout:"
            tail -n 20 "$temp_output"
        fi
        rm "$temp_output"
        return 1
    elif [ $status -ne 0 ]; then
        echo -e "${RED}Command failed with status $status: $name${NC}"
        echo "FAILED" > "$output_file"
        if [ $DEBUG_MODE -ge 1 ]; then
            echo "Error output:"
            cat "$temp_output"
        fi
        rm "$temp_output"
        return 1
    else
        # Command succeeded, copy output
        cat "$temp_output" > "$output_file"
        rm "$temp_output"
        return 0
    fi
}

# Process arguments
if [ $# -ge 2 ]; then
    OPENMP_THREADS=$1
    MPI_PROCESSES=$2
else
    OPENMP_THREADS=4
    MPI_PROCESSES=4
    echo -e "${YELLOW}Using default values: $OPENMP_THREADS OpenMP threads, $MPI_PROCESSES MPI processes${NC}"
    echo -e "${YELLOW}Usage: $0 <openmp_threads> <mpi_processes>${NC}"
fi

# Set debug mode from environment variable (3rd parameter)
if [ $# -ge 3 ]; then
    DEBUG_MODE=$3
    echo -e "${YELLOW}Debug mode set to: $DEBUG_MODE${NC}"
fi

# Set up directories
INPUT_DIR=generated/increasing_frames
OUTPUT_DIR=generated/increasing_frames_processed
LOG_DIR=output/logs
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Check if executables exist
missing=0
echo -e "${CYAN}Checking for required executables...${NC}"
for exe in sobelf sobelf_omp sobelf_mpi sobelf_omp_mpi sobelf_cuda sobelf_mpi_omp_cuda; do
    if [ ! -f "./$exe" ]; then
        echo -e "${RED}Missing executable: $exe. Please run ./compile_benchmark.sh first.${NC}"
        missing=1
    else
        echo -e "${GREEN}Found: $exe${NC}"
    fi
done

if [ $missing -eq 1 ]; then
    echo -e "${YELLOW}Some executables are missing. Running compile_benchmark.sh...${NC}"
    ./compile_benchmark.sh
    if [ $? -ne 0 ]; then
        echo -e "${RED}Compilation failed. Please fix compilation errors before running the benchmark.${NC}"
        exit 1
    fi
fi

# Selected images for benchmarking
SELECTED_IMAGES=(
    "noise_frames_1"
    "noise_frames_2"
    "noise_frames_3"
    "noise_frames_4"
    "noise_frames_5"
    "noise_frames_6"
    "noise_frames_7"
    "noise_frames_8"
    "noise_frames_9"
    "noise_frames_10"
)

echo -e "\n${GREEN}Comprehensive Benchmark of All Implementations${NC}"
echo -e "${YELLOW}OpenMP Threads: $OPENMP_THREADS, MPI Processes: $MPI_PROCESSES${NC}\n"

# Check for CUDA availability
CUDA_AVAILABLE=0
if command -v nvcc >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo -e "${GREEN}CUDA is available. GPU implementations will be benchmarked.${NC}"
    CUDA_AVAILABLE=1
    # Print GPU info
    if [ $DEBUG_MODE -ge 1 ]; then
        echo -e "${CYAN}GPU Information:${NC}"
        nvidia-smi
    fi
else
    echo -e "${YELLOW}CUDA is not available. GPU implementations will be skipped.${NC}"
fi

# Print the header
echo -e "${BLUE}Image | Specs | Sequential | OpenMP | MPI | MPI+OpenMP | CUDA | CUDA+MPI | Hybrid MPI+OpenMP+CUDA${NC}"
echo "-------------------------------------------------------------------------------------------------------------------------------"

# Process selected images
for img_name in "${SELECTED_IMAGES[@]}"; do
    img_path="${INPUT_DIR}/${img_name}.gif"
    
    # Skip if image doesn't exist
    if [ ! -f "$img_path" ]; then
        echo -e "${YELLOW}Skipping $img_name (file not found)${NC}"
        continue
    fi
    
    # Get image info
    if command -v identify > /dev/null 2>&1; then
        frame_count=$(identify "$img_path" | wc -l)
        resolution=$(identify -format "%wx%h" "$img_path[0]" 2>/dev/null)
        image_specs="$frame_count frames, $resolution"
    else
        image_specs="Size info N/A"
    fi
    
    # Initialize results row
    row="${img_name} | ${image_specs}"
    
    # Create log directory for this image
    img_log_dir="${LOG_DIR}/${img_name}"
    mkdir -p "${img_log_dir}"
    
    # Run sequential version
    seq_output="${OUTPUT_DIR}/${img_name}_seq.gif"
    seq_log="${img_log_dir}/sequential.log"
    
    run_with_timeout "./sobelf '$img_path' '$seq_output'" "$seq_log" 300 "Sequential"
    seq_time=$(grep "SOBEL done in" "$seq_log" | awk '{print $4}')
    
    if [ ! -z "$seq_time" ]; then
        row="${row} | ${seq_time}s (1.00x)"
        baseline_time=$seq_time
    else
        row="${row} | Failed"
        baseline_time=""
    fi
    
    # Run OpenMP version
    export OMP_NUM_THREADS=$OPENMP_THREADS
    openmp_output="${OUTPUT_DIR}/${img_name}_openmp.gif"
    openmp_log="${img_log_dir}/openmp.log"
    
    run_with_timeout "./sobelf_omp '$img_path' '$openmp_output' $OPENMP_THREADS" "$openmp_log" 300 "OpenMP"
    openmp_time=$(grep "SOBEL done in" "$openmp_log" | awk '{print $4}')
    
    if [ ! -z "$openmp_time" ] && [ ! -z "$baseline_time" ]; then
        speedup=$(echo "scale=2; $baseline_time/$openmp_time" | bc)
        row="${row} | ${openmp_time}s (${speedup}x)"
    else
        row="${row} | Failed"
    fi
    
    # Run MPI-only version
    mpi_only_output="${OUTPUT_DIR}/${img_name}_mpi_only.gif"
    mpi_only_log="${img_log_dir}/mpi_only.log"
    
    run_with_timeout "mpirun -np $MPI_PROCESSES ./sobelf_mpi '$img_path' '$mpi_only_output'" "$mpi_only_log" 300 "MPI-only"
    mpi_only_time=$(grep "SOBEL done in" "$mpi_only_log" | awk '{print $4}')
    
    if [ ! -z "$mpi_only_time" ] && [ ! -z "$baseline_time" ]; then
        speedup=$(echo "scale=2; $baseline_time/$mpi_only_time" | bc)
        row="${row} | ${mpi_only_time}s (${speedup}x)"
    else
        row="${row} | Failed"
    fi
    
    # Run MPI+OpenMP version
    mpi_output="${OUTPUT_DIR}/${img_name}_mpi_omp.gif"
    mpi_log="${img_log_dir}/mpi_omp.log"
    
    run_with_timeout "mpirun -np $MPI_PROCESSES ./sobelf_omp_mpi '$img_path' '$mpi_output' $OPENMP_THREADS" "$mpi_log" 300 "MPI+OpenMP"
    mpi_time=$(grep "SOBEL done in" "$mpi_log" | awk '{print $4}')
    
    if [ ! -z "$mpi_time" ] && [ ! -z "$baseline_time" ]; then
        speedup=$(echo "scale=2; $baseline_time/$mpi_time" | bc)
        row="${row} | ${mpi_time}s (${speedup}x)"
    else
        row="${row} | Failed"
    fi
    
    # Run CUDA version (single process)
    if [ $CUDA_AVAILABLE -eq 1 ]; then
        cuda_output="${OUTPUT_DIR}/${img_name}_cuda.gif"
        cuda_log="${img_log_dir}/cuda.log"
        
        run_with_timeout "./sobelf_cuda '$img_path' '$cuda_output'" "$cuda_log" 300 "CUDA"
        cuda_time=$(grep "SOBEL done in" "$cuda_log" | awk '{print $4}')
        
        if [ ! -z "$cuda_time" ] && [ ! -z "$baseline_time" ]; then
            speedup=$(echo "scale=2; $baseline_time/$cuda_time" | bc)
            row="${row} | ${cuda_time}s (${speedup}x)"
        else
            row="${row} | Failed"
        fi
        
        # Run CUDA+MPI version with enhanced error handling
        cudampi_output="${OUTPUT_DIR}/${img_name}_cudampi.gif"
        cudampi_log="${img_log_dir}/cuda_mpi.log"
        
        echo -e "${CYAN}Running CUDA+MPI version with enhanced diagnostics...${NC}"
        
        # Set environment variables that might help with CUDA+MPI
        export CUDA_VISIBLE_DEVICES="0,1"  # Limit to first two GPUs
        export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
        
        # Run with detailed output
        if [ $DEBUG_MODE -ge 2 ]; then
            # More verbose run for debugging
            run_with_timeout "mpirun -np $MPI_PROCESSES --mca btl_base_verbose 10 ./sobelf_cuda '$img_path' '$cudampi_output'" "$cudampi_log" 300 "CUDA+MPI"
        else
            run_with_timeout "mpirun -np $MPI_PROCESSES ./sobelf_cuda '$img_path' '$cudampi_output'" "$cudampi_log" 300 "CUDA+MPI"
        fi
        
        cudampi_time=$(grep "SOBEL done in" "$cudampi_log" | awk '{print $4}')
        
        if [ ! -z "$cudampi_time" ] && [ ! -z "$baseline_time" ]; then
            speedup=$(echo "scale=2; $baseline_time/$cudampi_time" | bc)
            row="${row} | ${cudampi_time}s (${speedup}x)"
        else
            echo -e "${RED}CUDA+MPI version failed. See log at ${cudampi_log}${NC}"
            if [ $DEBUG_MODE -ge 1 ] && [ -f "$cudampi_log" ]; then
                echo -e "${YELLOW}Last 20 lines of log:${NC}"
                tail -n 20 "$cudampi_log"
            fi
            row="${row} | Failed"
        fi
        
        # Run Hybrid MPI+OpenMP+CUDA version with enhanced error handling
        hybrid_output="${OUTPUT_DIR}/${img_name}_hybrid.gif"
        hybrid_log="${img_log_dir}/hybrid.log"
        
        echo -e "${CYAN}Running Hybrid MPI+OpenMP+CUDA version with enhanced diagnostics...${NC}"
        
        # Run with detailed output
        if [ $DEBUG_MODE -ge 2 ]; then
            # More verbose run for debugging
            run_with_timeout "mpirun -np $MPI_PROCESSES --mca btl_base_verbose 10 ./sobelf_mpi_omp_cuda '$img_path' '$hybrid_output' $OPENMP_THREADS" "$hybrid_log" 300 "Hybrid"
        else
            run_with_timeout "mpirun -np $MPI_PROCESSES ./sobelf_mpi_omp_cuda '$img_path' '$hybrid_output' $OPENMP_THREADS" "$hybrid_log" 300 "Hybrid"
        fi
        
        hybrid_time=$(grep "SOBEL done in" "$hybrid_log" | awk '{print $4}')
        
        if [ ! -z "$hybrid_time" ] && [ ! -z "$baseline_time" ]; then
            speedup=$(echo "scale=2; $baseline_time/$hybrid_time" | bc)
            row="${row} | ${hybrid_time}s (${speedup}x)"
        else
            echo -e "${RED}Hybrid version failed. See log at ${hybrid_log}${NC}"
            if [ $DEBUG_MODE -ge 1 ] && [ -f "$hybrid_log" ]; then
                echo -e "${YELLOW}Last 20 lines of log:${NC}"
                tail -n 20 "$hybrid_log"
            fi
            row="${row} | Failed"
        fi
    else
        # CUDA not available, add placeholders to the results
        row="${row} | N/A | N/A | N/A"
    fi
    
    echo "$row"
done

echo -e "\n${GREEN}Summary:${NC}"
echo "  - Sequential: Base implementation, single thread"
echo "  - OpenMP: Multi-threaded CPU implementation using $OPENMP_THREADS threads"
echo "  - MPI: Distributed CPU implementation using $MPI_PROCESSES processes"
echo "  - MPI+OpenMP: Distributed CPU implementation using $MPI_PROCESSES processes, each with $OPENMP_THREADS threads"

if [ $CUDA_AVAILABLE -eq 1 ]; then
    echo "  - CUDA: GPU-accelerated implementation (single process)"
    echo "  - CUDA+MPI: Distributed GPU implementation using $MPI_PROCESSES processes, each using GPU acceleration"
    echo "  - Hybrid MPI+OpenMP+CUDA: Fully hybrid implementation with $MPI_PROCESSES processes, each with $OPENMP_THREADS threads and GPU acceleration"
else
    echo "  - CUDA implementations: Not available (CUDA not detected)"
fi

echo "  - Speedup values are relative to sequential implementation (higher is better)"
echo "  - Detailed logs are saved in ${LOG_DIR}"
echo "=========================================================================================================================="

echo -e "\n${YELLOW}Note:${NC}"
if [ $DEBUG_MODE -eq 0 ]; then
    echo "  - Run with DEBUG=1 for more diagnostic information: DEBUG=1 $0 [openmp_threads] [mpi_processes]"
    echo "  - Run with DEBUG=2 for verbose MPI and CUDA information: DEBUG=2 $0 [openmp_threads] [mpi_processes]"
fi 