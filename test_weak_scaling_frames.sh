#!/bin/bash
# test_weak_scaling_frames.sh
# This script runs the MPI domain decomposition version on GIFs with increasing frame counts.
# It expects that the generated images in the folder "generated/increasing_frames" are named
# like "noise_frames_1.gif", "noise_frames_2.gif", "noise_frames_4.gif", "noise_frames_8.gif",
# corresponding to the number of frames (i.e. workload) increasing proportionally to the number of processes.
#
# The script outputs a CSV file "data/weak_scaling_frames.csv" with columns:
# NumProcs,NumFrames,TotalTime

mkdir -p data
OUTPUT_CSV="data/weak_scaling_frames.csv"
echo "NumProcs,NumFrames,TotalTime" > "$OUTPUT_CSV"

INPUT_DIR="generated/increasing_frames"
OUTPUT_DIR="output_weak_frames"
mkdir -p "$OUTPUT_DIR"

# Define process counts for weak scaling experiment
PROCS_LIST=(1 2 4 8)

for NP in "${PROCS_LIST[@]}"; do
    # Choose image based on NP (e.g., noise_frames_1.gif for NP=1, noise_frames_2.gif for NP=2, etc.)
    IMG="$INPUT_DIR/noise_frames_${NP}.gif"
    if [ ! -f "$IMG" ]; then
        echo "Image $IMG not found, skipping."
        continue
    fi
    # Determine number of frames using ImageMagick's identify (fallback: use NP)
    if command -v identify >/dev/null 2>&1; then
        NUM_FRAMES=$(identify "$IMG" | wc -l | tr -d ' ')
    else
        NUM_FRAMES=$NP
    fi

    DEST="$OUTPUT_DIR/$(basename "$IMG" .gif)_out.gif"
    echo "Running MPI domain decomposition on $IMG with ${NP} processes..."
    RESULT=$(mpirun -np "$NP" ./sobelf_mpi_domain "$IMG" "$DEST" 2>&1)
    TOTAL_TIME=$(echo "$RESULT" | grep "SOBEL done in" | awk '{print $4}')
    [ -z "$TOTAL_TIME" ] && TOTAL_TIME=0
    echo "$NP,$NUM_FRAMES,$TOTAL_TIME" >> "$OUTPUT_CSV"
done

echo "Weak scaling (frames) tests complete. Results saved in $OUTPUT_CSV"
