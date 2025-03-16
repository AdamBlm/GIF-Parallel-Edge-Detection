#!/bin/bash

# Compile the OpenMP implementation using the Makefile
# (Note: Make sure sobelf_omp.c is in the src/ directory)
echo "Compiling sobelf_omp using make..."
make sobelf_omp

# Then proceed with the rest of the script
INPUT_DIR=images/original
OUTPUT_DIR=images/processed
THREADS=${1:-4}  # Default to 4 threads if not specified

mkdir -p $OUTPUT_DIR 2>/dev/null

echo "Running OpenMP version with $THREADS threads"

for i in $INPUT_DIR/*gif ; do
    DEST=$OUTPUT_DIR/`basename $i .gif`-sobel-omp.gif
    echo "Running test on $i -> $DEST"

    ./sobelf_omp $i $DEST $THREADS
done

echo "All OpenMP processing complete" 
