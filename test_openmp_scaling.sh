#!/bin/bash
# test_openmp_scaling.sh
# Runs the OpenMP version (sobelf_omp) on each GIF in images/original
# with multiple thread counts. It outputs a CSV file with columns:
# ImageName,Threads,TotalTime

mkdir -p data
OUTPUT_CSV="data/openmp_scaling.csv"
echo "ImageName,Threads,TotalTime" > "$OUTPUT_CSV"

INPUT_DIR="images/original"
OUTPUT_DIR="images/processed_openmp"
mkdir -p "$OUTPUT_DIR"

# Define thread counts to test (adjust as needed)
THREADS_LIST=(1 2 4 8)

for IMG in "$INPUT_DIR"/*.gif; do
    BASENAME=$(basename "$IMG")
    for T in "${THREADS_LIST[@]}"; do
        DEST="$OUTPUT_DIR/${BASENAME%.gif}-omp${T}.gif"
        echo "Running sobelf_omp on $IMG -> $DEST with ${T} threads"
        RESULT=$(./sobelf_omp "$IMG" "$DEST" $T 2>&1)
        # Extract the runtime from a line like "SOBEL done in X s"
        TOTAL_TIME=$(echo "$RESULT" | grep "SOBEL done in" | awk '{print $4}')
        [ -z "$TOTAL_TIME" ] && TOTAL_TIME=0
        echo "$BASENAME,$T,$TOTAL_TIME" >> "$OUTPUT_CSV"
    done
done

echo "OpenMP scaling tests complete. Results in $OUTPUT_CSV"
