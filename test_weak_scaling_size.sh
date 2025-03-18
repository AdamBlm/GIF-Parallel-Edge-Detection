#!/bin/bash
# test_weak_scaling_size.sh
# This script runs the MPI domain decomposition version on GIFs with increasing resolution.
# It expects that the generated images in "generated/increasing_size" are named using their resolution,
# for example: "noise_size_100x100.gif", "noise_size_200x200.gif", "noise_size_400x400.gif", "noise_size_800x800.gif".
# These images should represent workloads that scale proportionally with the number of processes.
#
# The script outputs a CSV file "data/weak_scaling_size.csv" with columns:
# NumProcs,Resolution,TotalTime

mkdir -p data
OUTPUT_CSV="data/weak_scaling_size.csv"
echo "NumProcs,Resolution,TotalTime" > "$OUTPUT_CSV"

INPUT_DIR="generated/increasing_size"
OUTPUT_DIR="output_weak_size"
mkdir -p "$OUTPUT_DIR"

# Define process counts for weak scaling experiment
PROCS_LIST=(1 2 4 8)

# Map process count to resolution (adjust as appropriate)
for NP in "${PROCS_LIST[@]}"; do
    if [ "$NP" -eq 1 ]; then
        RES="100x100"
    elif [ "$NP" -eq 2 ]; then
        RES="200x200"
    elif [ "$NP" -eq 4 ]; then
        RES="400x400"
    elif [ "$NP" -eq 8 ]; then
        RES="800x800"
    else
        RES="NA"
    fi
    IMG="$INPUT_DIR/noise_size_${RES}.gif"
    if [ ! -f "$IMG" ]; then
        echo "Image $IMG not found, skipping."
        continue
    fi

    DEST="$OUTPUT_DIR/$(basename "$IMG" .gif)_out.gif"
    echo "Running MPI domain decomposition on $IMG with ${NP} processes..."
    RESULT=$(mpirun -np "$NP" ./sobelf_mpi_domain "$IMG" "$DEST" 2>&1)
    TOTAL_TIME=$(echo "$RESULT" | grep "SOBEL done in" | awk '{print $4}')
    [ -z "$TOTAL_TIME" ] && TOTAL_TIME=0
    echo "$NP,$RES,$TOTAL_TIME" >> "$OUTPUT_CSV"
done

echo "Weak scaling (size) tests complete. Results saved in $OUTPUT_CSV"
