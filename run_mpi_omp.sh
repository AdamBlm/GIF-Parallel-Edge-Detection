#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get thread and process count from command line
if [ $# -ge 1 ]; then
    THREADS=$1
else
    THREADS=4
fi

if [ $# -ge 2 ]; then
    MPI_PROCS=$2
else
    MPI_PROCS=2
fi

echo "====================================================================================="
echo -e "${GREEN}Running OpenMP+MPI Implementation${NC}"
echo -e "${YELLOW}Configuration: $THREADS OpenMP threads, $MPI_PROCS MPI processes${NC}"
echo "====================================================================================="

# Check OpenMP+MPI version and compile if needed
if [ ! -f ./sobelf_omp_mpi ]; then
    echo "Compiling OpenMP+MPI version..."
    if [ ! -f Makefile.mpi ]; then
        echo "Creating Makefile.mpi..."
        cat > Makefile.mpi << 'EOF'
CC = mpicc
CFLAGS = -O3 -Iinclude -fopenmp
LDFLAGS = -lm -fopenmp

OBJDIR = obj
SRCDIR = src

SOBEL_OBJ = $(OBJDIR)/dgif_lib.o $(OBJDIR)/egif_lib.o $(OBJDIR)/gif_err.o \
	$(OBJDIR)/gif_font.o $(OBJDIR)/gif_hash.o $(OBJDIR)/gifalloc.o \
	$(OBJDIR)/openbsd-reallocarray.o $(OBJDIR)/quantize.o

sobelf_omp_mpi: $(OBJDIR)/sobelf_omp_mpi.o $(SOBEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/sobelf_omp_mpi.o: $(SRCDIR)/sobelf_omp_mpi.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f sobelf_omp_mpi $(OBJDIR)/sobelf_omp_mpi.o
EOF
    fi
    make -f Makefile.mpi sobelf_omp_mpi
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to compile OpenMP+MPI version. Exiting.${NC}"
        exit 1
    fi
fi

# Function to get image information
get_image_info() {
    local img_path=$1
    
    # Try to use identify from ImageMagick if available
    if command -v identify > /dev/null 2>&1; then
        # Count total frames in the GIF properly
        local frame_count=$(identify "$img_path" | wc -l)
        local resolution=$(identify -format "%wx%h" "$img_path[0]" 2>/dev/null)
        
        echo "$frame_count frames, $resolution"
    else
        # Fallback method using our sobelf
        local info=$(./sobelf "$img_path" /dev/null 2>&1 | grep -E "Loading GIF|Width|Height")
        local width=$(echo "$info" | grep "Width" | awk '{print $2}')
        local height=$(echo "$info" | grep "Height" | awk '{print $2}')
        
        if [ -n "$width" ] && [ -n "$height" ]; then
            echo "Resolution: ${width}x${height}"
        else
            echo "Information not available"
        fi
    fi
}

# Set up parameters
INPUT_DIR=images/original
OUTPUT_DIR=output
mkdir -p $OUTPUT_DIR 2>/dev/null

# Format for the title row
FORMAT="%-30s | %-20s | %-15s\n"

# Print header
echo -e "\n${BLUE}OpenMP+MPI Sobel Filter Performance${NC}"
printf "${FORMAT}" "Image" "Specs" "Execution Time"
echo "--------------------------------------------------------------------------------"

# Find all GIF files in the images directory
IMAGES=()
for img_path in $(find $INPUT_DIR -name "*.gif" 2>/dev/null); do
    img_file=$(basename "$img_path")
    img_name="${img_file%.gif}"
    IMAGES+=("$img_name:$img_path")
done

echo "Found ${#IMAGES[@]} images to process"

# Process each image
for img_info in "${IMAGES[@]}"; do
    # Split image info
    img_name=${img_info%%:*}
    img_path=${img_info#*:}
    
    # Set output path
    mpi_output="$OUTPUT_DIR/${img_name}_mpi_out.gif"
    
    # Get image information
    image_specs=$(get_image_info "$img_path")
    
    # Set environment variable for OpenMP threads
    export OMP_NUM_THREADS=$THREADS
    
    # Process with MPI+OpenMP version and capture detailed timing
    echo -e "\n${GREEN}Processing: $img_path${NC}"
    echo -e "${YELLOW}Image specs: $image_specs${NC}"
    
    start_time=$(date +%s.%N)
    mpi_output_text=$(mpirun -np $MPI_PROCS ./sobelf_omp_mpi "$img_path" "$mpi_output" $THREADS 2>&1)
    end_time=$(date +%s.%N)
    
    # Extract detailed timing information
    load_time=$(echo "$mpi_output_text" | grep "Loading GIF" | grep "done in" | awk '{print $8}')
    filter_time=$(echo "$mpi_output_text" | grep "SOBEL done in" | awk '{print $4}')
    export_time=$(echo "$mpi_output_text" | grep "Exporting GIF" | grep "done in" | awk '{print $8}')
    total_time=$(echo "$end_time - $start_time" | bc)
    
    # Check if operation was successful
    if [ -z "$filter_time" ]; then
        echo -e "${RED}Failed to process $img_name. Check for errors.${NC}"
        continue
    fi
    
    # Print detailed timing information
    echo "Load time     : $load_time s"
    echo "Filter time   : $filter_time s"
    echo "Export time   : $export_time s"
    echo "Total time    : $total_time s"
    echo "-------------"
    
    # Print results row for the summary table
    printf "${FORMAT}" "$img_name" "$image_specs" "$filter_time s"
done

echo -e "\n${GREEN}Summary:${NC}"
echo "  - MPI+OpenMP configuration: $MPI_PROCS processes with $THREADS threads each"
echo "  - Processed images are saved in the $OUTPUT_DIR directory"
echo "=====================================================================================" 
