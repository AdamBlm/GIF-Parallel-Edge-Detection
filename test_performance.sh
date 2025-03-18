#!/bin/bash
# test_performance.sh
# Compare performance of sequential, MPI, and CUDA versions

# Build all versions
make

# Directories (adjust paths as needed)
INPUT_DIR="images/original"
OUTPUT_DIR="images/processed"
mkdir -p "$OUTPUT_DIR"

echo "Comparing performance across versions..."
echo "--------------------------------------------"

# Loop over each GIF in the input directory
for img in "$INPUT_DIR"/*.gif; do
    base=$(basename "$img" .gif)
    echo "Processing $img ..."

    # Sequential version
    echo "Sequential version:"
    SEQ_OUT=$(./sobelf "$img" "$OUTPUT_DIR/${base}-seq.gif")
    # Extract the time from a line like "SOBEL done in 0.123456 s"
    SEQ_TIME=$(echo "$SEQ_OUT" | grep "SOBEL done" | awk '{print $4}')
    echo "  Time: $SEQ_TIME s"

    # MPI version (using 4 processes)
    echo "MPI version (4 processes):"
    MPI_OUT=$(mpirun -np 4 ./sobelf_mpi "$img" "$OUTPUT_DIR/${base}-mpi.gif")
    MPI_TIME=$(echo "$MPI_OUT" | grep "SOBEL done" | awk '{print $4}')
    echo "  Time: $MPI_TIME s"

    # CUDA version (run as single process)
    echo "CUDA version:"
    CUDA_OUT=$(./sobelf_cuda "$img" "$OUTPUT_DIR/${base}-cuda.gif")
    CUDA_TIME=$(echo "$CUDA_OUT" | grep "SOBEL done" | awk '{print $4}')
    echo "  Time: $CUDA_TIME s"

    echo "--------------------------------------------"
done
