#!/bin/bash

# Source environment variables
source set_env.sh

# Make sure the executable is built
make -f Makefile.cuda

# Set up directories
INPUT_DIR=images/original
OUTPUT_DIR=images/processed
mkdir -p $OUTPUT_DIR

# Function to run tests with different number of processes
run_test() {
    local np=$1
    echo "=== Running with $np MPI processes ==="
    
    for img in $INPUT_DIR/*.gif; do
        base=$(basename $img .gif)
        output="$OUTPUT_DIR/${base}_cuda_${np}proc.gif"
        
        echo "Processing $img -> $output"
        mpirun -np $np ./sobelf_cuda $img $output
        
        # Print a separator for readability
        echo "----------------------------------------"
    done
}

# Run tests with different numbers of processes
if [ $# -eq 0 ]; then
    # Default: test with 1, 2, and 4 processes
    run_test 1
    run_test 2
    run_test 4
else
    # Use user-specified number of processes
    for np in "$@"; do
        run_test $np
    done
fi

echo "All tests completed." 
