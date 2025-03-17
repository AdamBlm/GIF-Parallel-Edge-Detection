#!/bin/bash
# compare_versions.sh
# Usage: ./compare_versions.sh input.gif maxiter


if [ "$#" -lt 2 ]; then
    echo "Usage: $0 input.gif maxiter"
    exit 1
fi

INPUT="$1"
MAXITER="$2"


OUTPUT_SEQ="processed_seq.gif"
OUTPUT_MPI="processed_mpi.gif"


NP_LIST=(2 4 8)


echo "Testing Sequential Version:"
SEQ_OUT=$(./sobelf "$INPUT" "$OUTPUT_SEQ")

SEQ_TIME=$(echo "$SEQ_OUT" | grep "SOBEL done" | awk '{print $4}')
echo "Sequential SOBEL time: ${SEQ_TIME} s"
echo


echo "Testing MPI Version:"

for NP in "${NP_LIST[@]}"; do
    echo "Running MPI version with ${NP} processes"
    MPI_OUT=$(mpirun -np $NP ./sobelf_mpi "$INPUT" "$OUTPUT_MPI")
    MPI_TIME=$(echo "$MPI_OUT" | grep "SOBEL done" | awk '{print $4}')
    echo "MPI with ${NP} processes: SOBEL time: ${MPI_TIME} s"
done
