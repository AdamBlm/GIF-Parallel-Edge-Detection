#!/bin/bash

# Compile both versions
echo "Compiling sequential and OpenMP versions..."
make sobelf
make sobelf_omp

# Set up parameters
INPUT_DIR=images/original
OUTPUT_DIR=images/processed
THREADS=${1:-4}  # Default to 4 threads if not specified
OMP_SUFFIX="-sobel-omp"
SEQ_SUFFIX="-sobel-seq"

mkdir -p $OUTPUT_DIR 2>/dev/null

echo "====================================================================="
echo "Performance comparison: Sequential vs OpenMP ($THREADS threads)"
echo "====================================================================="
printf "%-30s %-15s %-15s %-15s\n" "Image" "Sequential(s)" "OpenMP(s)" "Speedup"
echo "---------------------------------------------------------------------"

# Process all images and compare performance
for i in $INPUT_DIR/*gif ; do
    filename=$(basename $i .gif)
    SEQ_DEST=$OUTPUT_DIR/${filename}${SEQ_SUFFIX}.gif
    OMP_DEST=$OUTPUT_DIR/${filename}${OMP_SUFFIX}.gif
    
    # Run sequential version with timing
    seq_output=$(./sobelf $i $SEQ_DEST 2>&1)
    seq_time=$(echo "$seq_output" | grep "SOBEL done in" | awk '{print $4}')
    
    # Run OpenMP version with timing
    omp_output=$(./sobelf_omp $i $OMP_DEST $THREADS 2>&1)
    omp_time=$(echo "$omp_output" | grep "SOBEL done in" | awk '{print $4}')
    
    # Calculate speedup
    speedup=$(echo "scale=2; $seq_time/$omp_time" | bc)
    
    # Print results
    printf "%-30s %-15s %-15s %-15s\n" "$filename" "$seq_time" "$omp_time" "$speedup"
done

echo "====================================================================="
echo "Summary:"
echo "OpenMP implementation was run with $THREADS threads"
echo "Higher speedup values indicate better parallel performance"
echo "=====================================================================" 
