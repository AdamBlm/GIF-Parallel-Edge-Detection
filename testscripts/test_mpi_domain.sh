#!/bin/bash
# test_mpi_domain.sh
#
# Runs the MPI domain decomposition version (sobelf_mpi_domain) 
# on each *.gif with multiple process counts. 
# Outputs data/domain_decomposition_results.csv

mkdir -p data
OUTPUT_CSV="data/domain_decomposition_results.csv"
echo "ImageName,NumProcs,LoadTime,FilterTime,ExportTime,TotalTime" > "$OUTPUT_CSV"

INPUT_DIR="images/original"
OUTPUT_DIR="images/processed"
mkdir -p "$OUTPUT_DIR"

make sobelf_mpi_domain

PROCS_LIST=(1 2 4 8)

for IMG in "$INPUT_DIR"/*.gif; do
    BASENAME=$(basename "$IMG")

    for NP in "${PROCS_LIST[@]}"; do
        DEST="$OUTPUT_DIR/${BASENAME%.gif}-domain${NP}.gif"
        echo "Running sobelf_mpi_domain on $IMG -> $DEST with ${NP} processes"

        RESULT=$(mpirun -np "$NP" ./sobelf_mpi_domain "$IMG" "$DEST")

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

echo "MPI domain decomposition tests complete. Results in $OUTPUT_CSV"
