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

# Make sure sequential version exists
if [ ! -f "./sobelf" ]; then
    echo -e "${YELLOW}Compiling sequential version...${NC}"
    make sobelf
    if [ ! -f "./sobelf" ]; then
        echo -e "${RED}Failed to compile sequential version${NC}"
        exit 1
    fi
fi

# Set up directories
INPUT_DIR=images/original
OUTPUT_DIR=images/processed
mkdir -p $OUTPUT_DIR

# Format for the title row
FORMAT="%-25s | %-15s | %-15s | %-15s\n"

echo -e "\n${GREEN}Comparing Sequential vs CUDA Implementation${NC}"
printf "${FORMAT}" "Image" "Specs" "Sequential" "CUDA"
echo "-------------------------------------------------------------------"

# Find all GIF files in the images directory
for img_path in $(find $INPUT_DIR -name "*.gif" 2>/dev/null); do
    img_file=$(basename "$img_path")
    img_name="${img_file%.gif}"
    
    # Set output paths
    seq_output="$OUTPUT_DIR/${img_name}_seq_out.gif"
    cuda_output="$OUTPUT_DIR/${img_name}_cuda_out.gif"
    
    # Get image info using identify if available
    if command -v identify > /dev/null 2>&1; then
        frame_count=$(identify "$img_path" | wc -l)
        resolution=$(identify -format "%wx%h" "$img_path[0]" 2>/dev/null)
        image_specs="$frame_count frames, $resolution"
    else
        image_specs="Size info N/A"
    fi
    
    # Run sequential version
    seq_output_text=$(./sobelf "$img_path" "$seq_output" 2>&1)
    seq_time=$(echo "$seq_output_text" | grep "SOBEL done in" | awk '{print $4}')
    
    # Run CUDA version
    cuda_output_text=$(mpirun -np 1 ./sobelf_cuda "$img_path" "$cuda_output" 2>&1)
    cuda_time=$(echo "$cuda_output_text" | grep "SOBEL done in" | awk '{print $4}')
    
    # Calculate speedup if both versions ran successfully
    if [ ! -z "$seq_time" ] && [ ! -z "$cuda_time" ]; then
        speedup=$(echo "scale=2; $seq_time/$cuda_time" | bc)
        cuda_result="$cuda_time s (${speedup}x)"
        seq_result="$seq_time s"
    else
        if [ -z "$seq_time" ]; then
            seq_result="Failed"
        else
            seq_result="$seq_time s"
        fi
        
        if [ -z "$cuda_time" ]; then
            cuda_result="Failed"
        else
            cuda_result="$cuda_time s"
        fi
    fi
    
    # Print results row
    printf "${FORMAT}" "$img_name" "$image_specs" "$seq_result" "$cuda_result"
done

echo -e "\n${GREEN}Summary:${NC}"
echo "  - Sequential: Base performance"
echo "  - CUDA: Using GPU acceleration with 1 MPI process"
echo "  - Higher speedup values (>1.0) indicate better GPU performance"
echo "=====================================================================" 
