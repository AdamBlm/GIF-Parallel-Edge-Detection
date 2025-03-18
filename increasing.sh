#!/bin/bash
# test_increasing.sh
# This script processes two sets of test GIFs:
# 1. GIFs with increasing frame numbers from generated/increasing_frames.
# 2. GIFs with increasing resolution from generated/increasing_size.
#
# It runs the OpenMP, MPI Domain, and CUDA implementations on each image,
# extracts the runtime, and stores the results into CSV files in the data folder.

# Set defaults for THREADS and MPI_PROCS if not provided
if [ $# -ge 1 ]; then
    THREADS=$1
else
    THREADS=4
fi

if [ $# -ge 2 ]; then
    MPI_PROCS=$2
else
    MPI_PROCS=2
fi

# Directories for the test images
FRAMES_DIR="generated/increasing_frames"
SIZE_DIR="generated/increasing_size"

# Create a temporary output folder for processed images
OUTPUT_DIR="output_increasing"
mkdir -p "$OUTPUT_DIR"

# Create data folder if it does not exist
mkdir -p data

# CSV files for the results
CSV_FRAMES="data/increasing_frames.csv"
CSV_SIZE="data/increasing_size.csv"

# Write headers to CSV files
echo "Image,NumFrames,OpenMP,MPI_Domain,CUDA" > "$CSV_FRAMES"
echo "Image,Resolution,OpenMP,MPI_Domain,CUDA" > "$CSV_SIZE"

# Function to get the number of frames using ImageMagick's identify (fallback returns "NA")
get_num_frames() {
    local img=$1
    if command -v identify >/dev/null 2>&1; then
        identify "$img" | wc -l | tr -d ' '
    else
        echo "NA"
    fi
}

# Function to get the resolution of the first frame using ImageMagick (fallback returns "NA")
get_resolution() {
    local img=$1
    if command -v identify >/dev/null 2>&1; then
        identify -format "%wx%h" "$img[0]" 2>/dev/null
    else
        echo "NA"
    fi
}

echo "Processing images with increasing frame numbers in $FRAMES_DIR..."
for img in "$FRAMES_DIR"/*.gif; do
    img_name=$(basename "$img" .gif)
    num_frames=$(get_num_frames "$img")
    
    # Set temporary output filenames (these can be overwritten)
    omp_output="$OUTPUT_DIR/${img_name}_omp.gif"
    mpi_domain_output="$OUTPUT_DIR/${img_name}_mpi_domain.gif"
    cuda_output="$OUTPUT_DIR/${img_name}_cuda.gif"
    
    echo "Processing $img_name (frames: $num_frames)..."
    
    # Run OpenMP version and extract runtime (assumes output contains a line like "SOBEL done in X s")
    omp_out=$(./sobelf_omp "$img" "$omp_output" $THREADS 2>&1)
    omp_time=$(echo "$omp_out" | grep "SOBEL done in" | awk '{print $4}')
    
    # Run MPI Domain version
    mpi_domain_out=$(mpirun -np $MPI_PROCS ./sobelf_mpi_domain "$img" "$mpi_domain_output" $THREADS 2>&1)
    mpi_domain_time=$(echo "$mpi_domain_out" | grep "SOBEL done in" | awk '{print $4}')
    
    # Run CUDA version (using 1 process) and extract runtime (assumes line "GPU filters done in X s")
    cuda_out=$(mpirun -np 1 ./sobelf_cuda "$img" "$cuda_output" 2>&1)
    cuda_time=$(echo "$cuda_out" | grep "GPU filters done in" | awk '{print $5}')
    
    # Append the results to the CSV file for increasing frames
    echo "$img_name,$num_frames,$omp_time,$mpi_domain_time,$cuda_time" >> "$CSV_FRAMES"
done

echo "Processing images with increasing resolution in $SIZE_DIR..."
for img in "$SIZE_DIR"/*.gif; do
    img_name=$(basename "$img" .gif)
    resolution=$(get_resolution "$img")
    
    # Set temporary output filenames
    omp_output="$OUTPUT_DIR/${img_name}_omp.gif"
    mpi_domain_output="$OUTPUT_DIR/${img_name}_mpi_domain.gif"
    cuda_output="$OUTPUT_DIR/${img_name}_cuda.gif"
    
    echo "Processing $img_name (resolution: $resolution)..."
    
    # Run OpenMP version
    omp_out=$(./sobelf_omp "$img" "$omp_output" $THREADS 2>&1)
    omp_time=$(echo "$omp_out" | grep "SOBEL done in" | awk '{print $4}')
    
    # Run MPI Domain version
    mpi_domain_out=$(mpirun -np $MPI_PROCS ./sobelf_mpi_domain "$img" "$mpi_domain_output" $THREADS 2>&1)
    mpi_domain_time=$(echo "$mpi_domain_out" | grep "SOBEL done in" | awk '{print $4}')
    
    # Run CUDA version
    cuda_out=$(mpirun -np 1 ./sobelf_cuda "$img" "$cuda_output" 2>&1)
    cuda_time=$(echo "$cuda_out" | grep "GPU filters done in" | awk '{print $5}')
    
    # Append the results to the CSV file for increasing size
    echo "$img_name,$resolution,$omp_time,$mpi_domain_time,$cuda_time" >> "$CSV_SIZE"
done

echo "CSV results saved:"
echo "  - Increasing Frames: $CSV_FRAMES"
echo "  - Increasing Size: $CSV_SIZE"
