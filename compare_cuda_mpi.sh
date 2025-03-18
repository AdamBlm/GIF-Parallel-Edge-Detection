#!/bin/bash

# Source environment variables
source set_env.sh

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Compile our CUDA version if needed
if [ ! -f "./sobelf_cuda" ]; then
    echo -e "${YELLOW}Compiling CUDA version...${NC}"
    make -f Makefile.cuda
    if [ ! -f "./sobelf_cuda" ]; then
        echo -e "${RED}Failed to compile CUDA version${NC}"
        exit 1
    fi
fi

# Set up directories
INPUT_DIR=images/original
OUTPUT_DIR=images/processed
mkdir -p $OUTPUT_DIR

# Process arguments
if [ $# -ge 1 ]; then
    # Use the specified processes or run with 1, 2, 4 processes by default
    MPI_PROCS_LIST=("$@")
else
    MPI_PROCS_LIST=(1 2 4)
fi

# Selected large images for better testing
SELECTED_IMAGES=(
    "Campusplan-Hausnr"
    "Mandelbrot-large"
    "giphy-3"
    "australian-flag-large"
    "051009.vince"
)

# Format for the title row
FORMAT="%-25s | %-15s"
for proc in "${MPI_PROCS_LIST[@]}"; do
    FORMAT="${FORMAT} | %-15s"
done
FORMAT="${FORMAT}\n"

echo -e "\n${GREEN}Comparing CUDA Performance with Different MPI Processes${NC}"

# Print the header
header="Image | Specs"
for proc in "${MPI_PROCS_LIST[@]}"; do
    header="${header} | ${proc} Process(es)"
done
echo -e "${BLUE}${header}${NC}"

separator="--------------------------------"
for proc in "${MPI_PROCS_LIST[@]}"; do
    separator="${separator}----------------"
done
echo $separator

# Process selected images
for img_name in "${SELECTED_IMAGES[@]}"; do
    img_path="${INPUT_DIR}/${img_name}.gif"
    
    # Skip if image doesn't exist
    if [ ! -f "$img_path" ]; then
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
    
    # Run CUDA with different process counts
    row="${img_name} | ${image_specs}"
    
    for proc in "${MPI_PROCS_LIST[@]}"; do
        output="${OUTPUT_DIR}/${img_name}_cuda_${proc}proc.gif"
        
        # Run CUDA version with specified number of processes
        cuda_output_text=$(mpirun -np $proc ./sobelf_cuda "$img_path" "$output" 2>&1)
        cuda_time=$(echo "$cuda_output_text" | grep "SOBEL done in" | awk '{print $4}')
        
        if [ ! -z "$cuda_time" ]; then
            if [ "$proc" -eq 1 ]; then
                # First process count is baseline
                baseline_time=$cuda_time
                row="${row} | ${cuda_time}s (1.00x)"
            else
                # Calculate speedup against 1 process
                if [ ! -z "$baseline_time" ]; then
                    speedup=$(echo "scale=2; $baseline_time/$cuda_time" | bc)
                    row="${row} | ${cuda_time}s (${speedup}x)"
                else
                    row="${row} | ${cuda_time}s (N/A)"
                fi
            fi
        else
            row="${row} | Failed"
        fi
    done
    
    echo "$row"
done

echo -e "\n${GREEN}Summary:${NC}"
echo "  - Baseline: 1 MPI process with CUDA"
echo "  - Higher speedup values (>1.0) indicate better scalability with multiple processes"
echo "  - For small images, overhead may cause slowdown with more processes"
echo "=====================================================================" 
