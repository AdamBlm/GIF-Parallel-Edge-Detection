#!/bin/bash
# test_openmp_weak_scaling_frames.sh
# Runs the OpenMP version (sobelf_omp) on images with increasing frame counts.
# Assumes that the generated images are in the folder "generated/increasing_frames"
# and are named according to the number of frames, e.g., "noise_frames_1.gif", "noise_frames_2.gif",
# "noise_frames_4.gif", "noise_frames_8.gif".
#
# The script outputs a CSV file "data/openmp_weak_scaling_frames.csv" with columns:
# Threads, NumFrames, TotalTime

mkdir -p data
OUTPUT_CSV="data/openmp_weak_scaling_frames.csv"
echo "Threads,NumFrames,TotalTime" > "$OUTPUT_CSV"

INPUT_DIR="generated/increasing_frames"
OUTPUT_DIR="output_openmp_weak_frames"
mkdir -p "$OUTPUT_DIR"

# Define thread counts to test (and corresponding image workload)
THREADS_LIST=(1 2 4 8)

for T in "${THREADS_LIST[@]}"; do
    IMG="$INPUT_DIR/noise_frames_${T}.gif"
    if [ ! -f "$IMG" ]; then
        echo "Image $IMG not found, skipping."
        continue
    fi
    # Get the number of frames using ImageMagick's identify (if available)
    if command -v identify >/dev/null 2>&1; then
        NUM_FRAMES=$(identify "$IMG" | wc -l | tr -d ' ')
    else
        NUM_FRAMES=$T
    fi

    DEST="$OUTPUT_DIR/$(basename "$IMG" .gif)_out.gif"
    echo "Running sobelf_omp on $IMG -> $DEST with ${T} threads..."
    RESULT=$(./sobelf_omp "$IMG" "$DEST" $T 2>&1)
    # Extract runtime from a line like "SOBEL done in X s"
    TOTAL_TIME=$(echo "$RESULT" | grep "SOBEL done in" | awk '{print $4}')
    [ -z "$TOTAL_TIME" ] && TOTAL_TIME=0
    echo "$T,$NUM_FRAMES,$TOTAL_TIME" >> "$OUTPUT_CSV"
done

echo "OpenMP weak scaling (frames) tests complete. Results saved in $OUTPUT_CSV"
