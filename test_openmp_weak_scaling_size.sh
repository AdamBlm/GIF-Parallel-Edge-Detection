#!/bin/bash
# test_openmp_weak_scaling_size.sh
# Runs the OpenMP version (sobelf_omp) on images with increasing resolution.
# Assumes that the generated images are in "generated/increasing_size" and are named by resolution,
# e.g., "noise_size_100x100.gif", "noise_size_200x200.gif", "noise_size_400x400.gif", "noise_size_800x800.gif".
#
# The script outputs a CSV file "data/openmp_weak_scaling_size.csv" with columns:
# Threads, Resolution, TotalTime

mkdir -p data
OUTPUT_CSV="data/openmp_weak_scaling_size.csv"
echo "Threads,Resolution,TotalTime" > "$OUTPUT_CSV"

INPUT_DIR="generated/increasing_size"
OUTPUT_DIR="output_openmp_weak_size"
mkdir -p "$OUTPUT_DIR"

# Define thread counts and corresponding resolutions
declare -A RES_MAP
RES_MAP[1]="100x100"
RES_MAP[2]="200x200"
RES_MAP[4]="400x400"
RES_MAP[8]="800x800"

THREADS_LIST=(1 2 4 8)

for T in "${THREADS_LIST[@]}"; do
    RES=${RES_MAP[$T]}
    IMG="$INPUT_DIR/noise_size_${RES}.gif"
    if [ ! -f "$IMG" ]; then
        echo "Image $IMG not found, skipping."
        continue
    fi

    DEST="$OUTPUT_DIR/$(basename "$IMG" .gif)_out.gif"
    echo "Running sobelf_omp on $IMG -> $DEST with ${T} threads..."
    RESULT=$(./sobelf_omp "$IMG" "$DEST" $T 2>&1)
    TOTAL_TIME=$(echo "$RESULT" | grep "SOBEL done in" | awk '{print $4}')
    [ -z "$TOTAL_TIME" ] && TOTAL_TIME=0
    echo "$T,$RES,$TOTAL_TIME" >> "$OUTPUT_CSV"
done

echo "OpenMP weak scaling (size) tests complete. Results saved in $OUTPUT_CSV"
