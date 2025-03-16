#!/bin/bash
# test_sequential.sh
# 
# Measures runtime of the sequential code (sobelf) for each *.gif in images/original/
# Outputs a CSV file data/sequential_baseline.csv with columns:
#   ImageName,LoadTime,FilterTime,ExportTime,TotalTime

# Ensure we have a "data" directory for logs
mkdir -p data
OUTPUT_CSV="data/sequential_baseline.csv"

# Write CSV header
echo "ImageName,LoadTime,FilterTime,ExportTime,TotalTime" > "$OUTPUT_CSV"

INPUT_DIR="images/original"
OUTPUT_DIR="images/processed"
mkdir -p "$OUTPUT_DIR"

# Rebuild code (optional)
make sobelf

for IMG in "$INPUT_DIR"/*.gif; do
    BASENAME=$(basename "$IMG")
    DEST="$OUTPUT_DIR/${BASENAME%.gif}-sobel.gif"
    echo "Running sequential sobelf on $IMG -> $DEST"

    # Run the program and capture its output
    RESULT=$(./sobelf "$IMG" "$DEST")

    # Example lines in $RESULT might be:
    #   "GIF loaded from file <img> with <n> image(s) in X s"
    #   "SOBEL done in Y s"
    #   "Export done in Z s in file <...>"

    LOAD_TIME=$(echo "$RESULT"   | grep "GIF loaded"  | awk '{print $10}')   # the 'X' in "... in X s"
    FILTER_TIME=$(echo "$RESULT" | grep "SOBEL done"  | awk '{print $4}')    # the 'Y' in "SOBEL done in Y s"
    EXPORT_TIME=$(echo "$RESULT" | grep "Export done" | awk '{print $4}')    # the 'Z' in "Export done in Z s..."

    # If any step doesn't produce output, default to 0
    if [ -z "$LOAD_TIME" ];   then LOAD_TIME=0; fi
    if [ -z "$FILTER_TIME" ]; then FILTER_TIME=0; fi
    if [ -z "$EXPORT_TIME" ]; then EXPORT_TIME=0; fi

    # Compute a total time (simple sum)
    TOTAL_TIME=$(awk -v l="$LOAD_TIME" -v f="$FILTER_TIME" -v e="$EXPORT_TIME" \
                 'BEGIN {print l+f+e}')

    # Append row to CSV
    echo "$BASENAME,$LOAD_TIME,$FILTER_TIME,$EXPORT_TIME,$TOTAL_TIME" >> "$OUTPUT_CSV"
done

echo "Sequential tests complete. Results in $OUTPUT_CSV"
