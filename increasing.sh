#!/bin/bash
# test_increasing.sh
#
# Processes two sets of test GIFs in ascending order:
# 1. GIFs with increasing frame numbers in generated/increasing_frames.
# 2. GIFs with increasing resolution in generated/increasing_size.
#
# Runs the Sequential, OpenMP, MPI Domain, and CUDA implementations on each image,
# extracts their runtimes, and stores the results into CSV files in 'data/'.

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

FRAMES_DIR="generated/increasing_frames"
SIZE_DIR="generated/increasing_size"
OUTPUT_DIR="output_increasing"
mkdir -p "$OUTPUT_DIR"
mkdir -p data

# CSV files
CSV_FRAMES="data/increasing_frames.csv"
CSV_SIZE="data/increasing_size.csv"

# Write headers for the new CSV files, now including 'Sequential'
echo "Image,NumFrames,Sequential,OpenMP,MPI_Domain,CUDA" > "$CSV_FRAMES"
echo "Image,Resolution,Sequential,OpenMP,MPI_Domain,CUDA" > "$CSV_SIZE"

# Function to get number of frames using ImageMagick's identify, fallback is "NA"
get_num_frames() {
    local img=$1
    if command -v identify >/dev/null 2>&1; then
        identify "$img" | wc -l | tr -d ' '
    else
        echo "NA"
    fi
}

# Function to get resolution from the first frame
get_resolution() {
    local img=$1
    if command -v identify >/dev/null 2>&1; then
        identify -format "%wx%h" "$img[0]" 2>/dev/null
    else
        echo "NA"
    fi
}

############################
# 1) PROCESS INCREASING FRAMES
############################

echo "Processing images with increasing frame numbers in $FRAMES_DIR..."
for img in $(ls -v "$FRAMES_DIR"/*.gif 2>/dev/null); do
    [ -f "$img" ] || continue
    img_name=$(basename "$img" .gif)

    # Count frames
    num_frames=$(get_num_frames "$img")

    # Prepare output filenames
    seq_output="$OUTPUT_DIR/${img_name}_seq.gif"
    omp_output="$OUTPUT_DIR/${img_name}_omp.gif"
    mpi_domain_output="$OUTPUT_DIR/${img_name}_mpi_domain.gif"
    cuda_output="$OUTPUT_DIR/${img_name}_cuda.gif"

    echo "Processing $img_name (frames: $num_frames)..."

    # 1) Run sequential version
    seq_out=$(./sobelf "$img" "$seq_output" 2>&1)
    seq_time=$(echo "$seq_out" | grep "SOBEL done in" | awk '{print $4}')

    # 2) Run OpenMP
    omp_out=$(./sobelf_omp "$img" "$omp_output" $THREADS 2>&1)
    omp_time=$(echo "$omp_out" | grep "SOBEL done in" | awk '{print $4}')

    # 3) Run MPI Domain
    mpi_domain_out=$(mpirun -np $MPI_PROCS ./sobelf_mpi_domain "$img" "$mpi_domain_output" $THREADS 2>&1)
    mpi_domain_time=$(echo "$mpi_domain_out" | grep "SOBEL done in" | awk '{print $4}')

    # 4) Run CUDA (with 1 process)
    cuda_out=$(mpirun -np 1 ./sobelf_cuda "$img" "$cuda_output" 2>&1)
    cuda_time=$(echo "$cuda_out" | grep "GPU filters done in" | awk '{print $5}')

    # Append row to CSV for frames
    echo "$img_name,$num_frames,$seq_time,$omp_time,$mpi_domain_time,$cuda_time" >> "$CSV_FRAMES"
done

############################
# 2) PROCESS INCREASING SIZE
############################

echo "Processing images with increasing resolution in $SIZE_DIR..."
for img in $(ls -v "$SIZE_DIR"/*.gif 2>/dev/null); do
    [ -f "$img" ] || continue
    img_name=$(basename "$img" .gif)

    # Extract resolution
    resolution=$(get_resolution "$img")

    # Prepare output filenames
    seq_output="$OUTPUT_DIR/${img_name}_seq.gif"
    omp_output="$OUTPUT_DIR/${img_name}_omp.gif"
    mpi_domain_output="$OUTPUT_DIR/${img_name}_mpi_domain.gif"
    cuda_output="$OUTPUT_DIR/${img_name}_cuda.gif"

    echo "Processing $img_name (resolution: $resolution)..."

    # 1) Run sequential version
    seq_out=$(./sobelf "$img" "$seq_output" 2>&1)
    seq_time=$(echo "$seq_out" | grep "SOBEL done in" | awk '{print $4}')

    # 2) Run OpenMP version
    omp_out=$(./sobelf_omp "$img" "$omp_output" $THREADS 2>&1)
    omp_time=$(echo "$omp_out" | grep "SOBEL done in" | awk '{print $4}')

    # 3) Run MPI Domain version
    mpi_domain_out=$(mpirun -np $MPI_PROCS ./sobelf_mpi_domain "$img" "$mpi_domain_output" $THREADS 2>&1)
    mpi_domain_time=$(echo "$mpi_domain_out" | grep "SOBEL done in" | awk '{print $4}')

    # 4) Run CUDA
    cuda_out=$(mpirun -np 1 ./sobelf_cuda "$img" "$cuda_output" 2>&1)
    cuda_time=$(echo "$cuda_out" | grep "GPU filters done in" | awk '{print $5}')

    # Append row to CSV for size
    echo "$img_name,$resolution,$seq_time,$omp_time,$mpi_domain_time,$cuda_time" >> "$CSV_SIZE"
done

echo "CSV results saved:"
echo "  - Increasing Frames: $CSV_FRAMES"
echo "  - Increasing Size: $CSV_SIZE"
