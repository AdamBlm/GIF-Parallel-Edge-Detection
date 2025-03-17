#!/bin/bash
# test_cuda.sh
#
# Runs the CUDA-based sobelf_cuda on each *.gif in images/original/.
# Outputs data/cuda_results.csv with times.

mkdir -p data
OUTPUT_CSV="data/cuda_results.csv"
echo "ImageName,LoadTime,FilterTime,ExportTime,TotalTime" > "$OUTPUT_CSV"

INPUT_DIR="images/original"
OUTPUT_DIR="images/processed"
mkdir -p "$OUTPUT_DIR"

make sobelf_cuda

for IMG in "$INPUT_DIR"/*.gif; do
    BASENAME=$(basename "$IMG")
    DEST="$OUTPUT_DIR/${BASENAME%.gif}-cuda.gif"
    echo "Running sobelf_cuda on $IMG -> $DEST"

    RESULT=$(./sobelf_cuda "$IMG" "$DEST")

    # Suppose your sobelf_cuda prints lines like:
    #   "GIF loaded from file ... in X s"
    #   "GPU filters done in Y s"
    #   "Export done in Z s in file ..."

    LOAD_TIME=$(echo "$RESULT"   | grep "GIF loaded"        | awk '{print $10}')
    FILTER_TIME=$(echo "$RESULT" | grep "GPU filters done"  | awk '{print $5}')
    EXPORT_TIME=$(echo "$RESULT" | grep "Export done"       | awk '{print $4}')

    [ -z "$LOAD_TIME" ]   && LOAD_TIME=0
    [ -z "$FILTER_TIME" ] && FILTER_TIME=0
    [ -z "$EXPORT_TIME" ] && EXPORT_TIME=0

    TOTAL_TIME=$(awk -v l="$LOAD_TIME" -v f="$FILTER_TIME" -v e="$EXPORT_TIME" 'BEGIN {print l+f+e}')

    echo "$BASENAME,$LOAD_TIME,$FILTER_TIME,$EXPORT_TIME,$TOTAL_TIME" >> "$OUTPUT_CSV"
done

echo "CUDA tests complete. Results in $OUTPUT_CSV"
