#!/bin/bash
# test_mpi_frame.sh
#
# Runs the MPI "frame distribution" version (sobelf_mpi) on each *.gif 
# for multiple process counts. Outputs data/mpi_frame_distribution.csv

mkdir -p data
OUTPUT_CSV="data/mpi_frame_distribution.csv"
echo "ImageName,NumProcs,LoadTime,FilterTime,ExportTime,TotalTime" > "$OUTPUT_CSV"

INPUT_DIR="images/original"
OUTPUT_DIR="images/processed"
mkdir -p "$OUTPUT_DIR"

# Build the MPI version
make sobelf_mpi

# Choose a list of process counts
PROCS_LIST=(1 2 4 8)

for IMG in "$INPUT_DIR"/*.gif; do
    BASENAME=$(basename "$IMG")
    for NP in "${PROCS_LIST[@]}"; do
        DEST="$OUTPUT_DIR/${BASENAME%.gif}-mpi${NP}.gif"
        echo "Running sobelf_mpi on $IMG -> $DEST with ${NP} processes"

        RESULT=$(mpirun -np "$NP" ./sobelf_mpi "$IMG" "$DEST")

        LOAD_TIME=$(echo "$RESULT"   | grep "GIF loaded"  | awk '{print $10}')
        FILTER_TIME=$(echo "$RESULT" | grep "SOBEL done"  | awk '{print $4}')
        EXPORT_TIME=$(echo "$RESULT" | grep "Export done" | awk '{print $4}')

        [ -z "$LOAD_TIME" ]   && LOAD_TIME=0
        [ -z "$FILTER_TIME" ] && FILTER_TIME=0
        [ -z "$EXPORT_TIME" ] && EXPORT_TIME=0

        TOTAL_TIME=$(awk -v l="$LOAD_TIME" -v f="$FILTER_TIME" -v e="$EXPORT_TIME" 'BEGIN {print l+f+e}')
        
        echo "$BASENAME,$NP,$LOAD_TIME,$FILTER_TIME,$EXPORT_TIME,$TOTAL_TIME" >> "$OUTPUT_CSV"
    done
done

echo "MPI (frame distribution) tests complete. Results in $OUTPUT_CSV"
