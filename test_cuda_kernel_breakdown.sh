#!/bin/bash
# test_cuda_kernel_breakdown.sh
# This script runs the CUDA version (sobelf_cuda) on each GIF in images/original,
# extracts the expanded kernel timing breakdown lines (printed by the CUDA code),
# and writes them to a CSV file.
#
# The CSV file will have columns:
# ImageName,H2D,Grayscale,Blur,Sobel,D2H,Other,Total
#
# In your CUDA code, each line should look like:
#   KernelTiming,<ImageName>,<H2D>,<Grayscale>,<Blur>,<Sobel>,<D2H>,<Other>,<Total>

mkdir -p data
OUTPUT_CSV="data/cuda_kernel_breakdown.csv"
echo "ImageName,H2D,Grayscale,Blur,Sobel,D2H,Other,Total" > "$OUTPUT_CSV"

INPUT_DIR="images/original"
OUTPUT_DIR="images/processed"
mkdir -p "$OUTPUT_DIR"

for IMG in "$INPUT_DIR"/*.gif; do
    BASENAME=$(basename "$IMG")
    DEST="$OUTPUT_DIR/${BASENAME%.gif}-cuda.gif"
    echo "Running sobelf_cuda on $IMG -> $DEST"

    # Run the CUDA application and capture its output
    RESULT=$(./sobelf_cuda "$IMG" "$DEST")

    # Extract any lines that begin with "KernelTiming,"
    # In your code, each such line should look like:
    #   KernelTiming,images/original/foo.gif_frame_0,0.001,0.002,0.003,0.004,0.005,0.006,0.007
    TIMING_LINES=$(echo "$RESULT" | grep "^KernelTiming,")

    if [ -n "$TIMING_LINES" ]; then
        # For each matching line, strip the "KernelTiming," prefix and append to the CSV
        while IFS= read -r line; do
            # e.g. line = "KernelTiming,images/original/foo.gif_frame_0,0.001,0.002,0.003,0.004,0.005,0.006,0.007"
            # we remove the prefix "KernelTiming,"
            CSV_LINE="${line#KernelTiming,}"
            echo "$CSV_LINE" >> "$OUTPUT_CSV"
        done <<< "$TIMING_LINES"
    else
        # If no KernelTiming line was found, append a zero-filled line for reference
        echo "${BASENAME},0,0,0,0,0,0,0" >> "$OUTPUT_CSV"
    fi
done

echo "CUDA kernel timing breakdown tests complete. Results saved in $OUTPUT_CSV"
