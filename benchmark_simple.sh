#!/bin/bash

# Simple benchmark script for MPI and Hybrid implementations

# Input file
INPUT_FILE="images/original/Mandelbrot-large.gif"
OUTPUT_DIR="output"

# Ensure output directory exists
mkdir -p $OUTPUT_DIR

echo "Starting benchmark for $INPUT_FILE"
echo "=================================="

# MPI version with different process counts
for procs in 1 2 4 8; do
    echo -n "Running MPI version with $procs processes... "
    output_file="$OUTPUT_DIR/mandelbrot_mpi_${procs}proc.gif"
    
    # Measure time with the time command
    { time mpirun -np $procs ./sobelf_mpi $INPUT_FILE $output_file; } 2>&1 | grep real
done

echo ""

# MPI+OpenMP version with different configurations
for procs in 1 2 4; do
    for threads in 1 2 4; do
        echo -n "Running MPI+OpenMP version with $procs processes and $threads threads... "
        output_file="$OUTPUT_DIR/mandelbrot_omp_mpi_${procs}p_${threads}t.gif"
        
        # Set OMP_NUM_THREADS
        export OMP_NUM_THREADS=$threads
        
        # Measure time with the time command
        { time mpirun -np $procs ./sobelf_omp_mpi $INPUT_FILE $output_file; } 2>&1 | grep real
    done
done

echo ""

# MPI+OpenMP+CUDA version if available
if [ -x ./sobelf_mpi_omp_cuda ]; then
    for procs in 1 2; do
        for threads in 1 2; do
            echo -n "Running Hybrid MPI+OpenMP+CUDA with $procs processes and $threads threads... "
            output_file="$OUTPUT_DIR/mandelbrot_hybrid_${procs}p_${threads}t.gif"
            
            # Set OMP_NUM_THREADS
            export OMP_NUM_THREADS=$threads
            
            # Measure time with the time command
            { time mpirun -np $procs ./sobelf_mpi_omp_cuda $INPUT_FILE $output_file; } 2>&1 | grep real
        done
    done
fi

echo "Benchmark complete!" 
