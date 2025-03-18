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
echo -e "${GREEN}Comparing All Available Implementations${NC}"
echo -e "${YELLOW}Configuration: $THREADS OpenMP threads, $MPI_PROCS MPI processes${NC}"
echo "====================================================================================="

# Flag to track if CUDA is available
CUDA_AVAILABLE=0

# Custom CUDA path found on this system
CUSTOM_CUDA_PATH="/Data/IPGen/env/lib/python3.13/site-packages/nvidia/cuda_runtime"

# Check if CUDA is available
command -v nvcc >/dev/null 2>&1
if [ $? -eq 0 ]; then
    # Standard CUDA directories
    CUDA_DIRS=("/usr/local/cuda/include" "/opt/cuda/include" "/usr/include/cuda" "$CUSTOM_CUDA_PATH/include")
    
    for dir in "${CUDA_DIRS[@]}"; do
        if [ -f "$dir/cuda_runtime.h" ]; then
            CUDA_AVAILABLE=1
            echo -e "${GREEN}CUDA detected at $dir${NC}"
            break
        fi
    done
    
    # If standard detection failed, try our custom path
    if [ $CUDA_AVAILABLE -eq 0 ] && [ -f "$CUSTOM_CUDA_PATH/include/cuda_runtime.h" ]; then
        CUDA_AVAILABLE=1
        echo -e "${GREEN}CUDA detected at custom path: $CUSTOM_CUDA_PATH/include${NC}"
    fi
    
    if [ $CUDA_AVAILABLE -eq 0 ]; then
        echo -e "${YELLOW}CUDA compiler (nvcc) found, but cuda_runtime.h is missing. Hybrid version will be disabled.${NC}"
    fi
else
    echo -e "${YELLOW}CUDA compiler (nvcc) not found. Hybrid version will be disabled.${NC}"
fi

# Force recompile all versions
echo -e "${YELLOW}Recompiling all versions...${NC}"

# Clean previous builds
echo "Cleaning previous builds..."
make clean

# Compile sequential version
echo "Compiling sequential version (using sequential.c)..."
make sobelf

# Compile OpenMP version
echo "Compiling OpenMP version..."
make sobelf_omp

# Compare blur parameters to match MPI version
echo "Updating blur parameters for sequential version to match MPI version..."
sed -i.bak 's/apply_blur_filter( image, 5, 20 )/apply_blur_filter( image, 3, 0 )/' src/sequential.c

# Recompile with updated parameters
echo "Recompiling sequential version with updated parameters..."
make sobelf

# Compile OpenMP+MPI version (MPI+OpenMP)
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
make -f Makefile.mpi clean
make -f Makefile.mpi sobelf_omp_mpi

# Compile MPI Frame Distribution version (from sobelf_mpi.c)
echo "Compiling MPI Frame Distribution version..."
make sobelf_mpi

# Compile MPI Domain Decomposition version (from sobelf_mpi_domain.c)
echo "Compiling MPI Domain Decomposition version..."
make sobelf_mpi_domain

# Check hybrid implementation (MPI+OpenMP+CUDA) only if CUDA is available
HYBRID_AVAILABLE=0
if [ $CUDA_AVAILABLE -eq 1 ]; then
    echo "Compiling Hybrid MPI+OpenMP+CUDA version..."
    if [ ! -f Makefile.hybrid ]; then
        echo "Creating Makefile.hybrid..."
        cat > Makefile.hybrid << EOF
CC = mpicc
NVCC = nvcc
CUDA_PATH = $CUSTOM_CUDA_PATH
CFLAGS = -O3 -Iinclude -fopenmp
NVCC_FLAGS = -I\$(CUDA_PATH)/include
LDFLAGS = -lm -lcudart -L\$(CUDA_PATH)/lib64 -L\$(CUDA_PATH)/lib -fopenmp

OBJDIR = obj
SRCDIR = src

SOBEL_OBJ = \$(OBJDIR)/dgif_lib.o \$(OBJDIR)/egif_lib.o \$(OBJDIR)/gif_err.o \\
    \$(OBJDIR)/gif_font.o \$(OBJDIR)/gif_hash.o \$(OBJDIR)/gifalloc.o \\
    \$(OBJDIR)/openbsd-reallocarray.o \$(OBJDIR)/quantize.o

sobelf_hybrid_3: \$(OBJDIR)/sobelf_hybrid_3.o \$(OBJDIR)/sobelf_hybrid_3_cuda.o \$(SOBEL_OBJ)
    \$(CC) \$(CFLAGS) -o \$@ \$^ \$(LDFLAGS)

\$(OBJDIR)/sobelf_hybrid_3.o: sobelf_hybrid_3.c
    \$(CC) \$(CFLAGS) -c -o \$@ \$<

\$(OBJDIR)/sobelf_hybrid_3_cuda.o: sobelf_hybrid_3.cu
    \$(NVCC) \$(NVCC_FLAGS) -c -o \$@ \$<

clean:
    rm -f sobelf_hybrid_3 \$(OBJDIR)/sobelf_hybrid_3.o \$(OBJDIR)/sobelf_hybrid_3_cuda.o
EOF
    fi
    make -f Makefile.hybrid clean
    make -f Makefile.hybrid sobelf_hybrid_3 2>&1 | tee hybrid_compile.log
    if [ -f ./sobelf_hybrid_3 ]; then
        HYBRID_AVAILABLE=1
        echo -e "${GREEN}Hybrid MPI+OpenMP+CUDA version successfully compiled${NC}"
    else
        echo -e "${RED}Failed to compile Hybrid MPI+OpenMP+CUDA version. See hybrid_compile.log for details.${NC}"
    fi
else
    echo -e "${YELLOW}Skipping Hybrid MPI+OpenMP+CUDA version compilation (CUDA not available)${NC}"
fi

# Compile CUDA version
echo "Compiling CUDA version..."
make sobelf_cuda

echo -e "${GREEN}Compilation of all versions complete.${NC}"

# Set up parameters
INPUT_DIR=images
OUTPUT_DIR=output
mkdir -p $OUTPUT_DIR 2>/dev/null

# Clean the output directory first
echo -e "${YELLOW}Cleaning output directory...${NC}"
rm -f $OUTPUT_DIR/* 2>/dev/null
echo -e "${GREEN}Output directory cleaned.${NC}"

# Create CSV file for graph data in data/ folder
mkdir -p data
CSV_FILE="data/compare_all.csv"
# New header now includes MPI Frames and MPI Domain columns
echo "Image,Sequential,OpenMP,MPI+OpenMP,Hybrid,CUDA,MPI Frames,MPI Domain" > "$CSV_FILE"

# Function to get image information
get_image_info() {
    local img_path=$1
    if command -v identify > /dev/null 2>&1; then
        local frame_count=$(identify "$img_path" | wc -l)
        local resolution=$(identify -format "%wx%h" "$img_path[0]" 2>/dev/null)
        echo "$frame_count frames, $resolution"
    else
        echo "Information not available"
    fi
}

# Format for the summary table rows (updated to include 7 columns)
FORMAT="%-25s | %-20s | %-15s | %-15s | %-15s | %-15s | %-15s | %-15s\n"

# Find all GIF files in the images/original folder
IMAGES=()
for img_path in $(find images/original -name "*.gif" 2>/dev/null); do
    img_file=$(basename "$img_path")
    img_name="${img_file%.gif}"
    IMAGES+=("$img_name:$img_path")
done

# Process each image
for img_info in "${IMAGES[@]}"; do
    img_name=${img_info%%:*}
    img_path=${img_info#*:}
    
    # Set output paths for each version
    seq_output="$OUTPUT_DIR/${img_name}_seq_out.gif"
    omp_output="$OUTPUT_DIR/${img_name}_omp_out.gif"
    mpi_omp_output="$OUTPUT_DIR/${img_name}_mpi_omp_out.gif"
    hybrid_output="$OUTPUT_DIR/${img_name}_hybrid_out.gif"
    cuda_output="$OUTPUT_DIR/${img_name}_cuda_out.gif"
    mpi_frames_output="$OUTPUT_DIR/${img_name}_mpi_frames_out.gif"
    mpi_domain_output="$OUTPUT_DIR/${img_name}_mpi_domain_out.gif"
    
    echo -e "\n${GREEN}Processing $img_path${NC}"
    image_specs=$(get_image_info "$img_path")
    echo -e "${YELLOW}Image specs: $image_specs${NC}"
    
    # Run sequential version
    seq_output_text=$(./sobelf "$img_path" "$seq_output" 2>&1)
    seq_time=$(echo "$seq_output_text" | grep "SOBEL done in" | awk '{print $4}')
    if [ -z "$seq_time" ]; then
        echo "Failed to process $img_name with sequential version, skipping."
        continue
    fi
    
    # Run OpenMP version
    omp_output_text=$(./sobelf_omp "$img_path" "$omp_output" $THREADS 2>&1)
    omp_time=$(echo "$omp_output_text" | grep "SOBEL done in" | awk '{print $4}')
    
    # Run MPI+OpenMP version
    mpi_omp_output_text=$(mpirun -np $MPI_PROCS ./sobelf_omp_mpi "$img_path" "$mpi_omp_output" $THREADS 2>&1)
    mpi_omp_time=$(echo "$mpi_omp_output_text" | grep "SOBEL done in" | awk '{print $4}')
    
    # Run Hybrid version (if available)
    if [ $HYBRID_AVAILABLE -eq 1 ]; then
        hybrid_output_text=$(mpirun -np $MPI_PROCS ./sobelf_hybrid_3 "$img_path" "$hybrid_output" $THREADS 2>&1)
        hybrid_time=$(echo "$hybrid_output_text" | grep "SOBEL done in" | awk '{print $4}')
    else
        hybrid_time="N/A"
    fi
    
    # Run CUDA version (if available)
    if [ $CUDA_AVAILABLE -eq 1 ]; then
        cuda_output_text=$(mpirun -np 1 ./sobelf_cuda "$img_path" "$cuda_output" 2>&1)
        # For CUDA, extract field 5 from the "GPU filters done in" line.
        cuda_time=$(echo "$cuda_output_text" | grep "GPU filters done in" | awk '{print $5}')
    else
        cuda_time="N/A"
    fi
    
    # Run MPI Frame Distribution version
    mpi_frames_output_text=$(mpirun -np $MPI_PROCS ./sobelf_mpi "$img_path" "$mpi_frames_output" $THREADS 2>&1)
    mpi_frames_time=$(echo "$mpi_frames_output_text" | grep "SOBEL done in" | awk '{print $4}')
    
    # Run MPI Domain Decomposition version
    mpi_domain_output_text=$(mpirun -np $MPI_PROCS ./sobelf_mpi_domain "$img_path" "$mpi_domain_output" $THREADS 2>&1)
    mpi_domain_time=$(echo "$mpi_domain_output_text" | grep "SOBEL done in" | awk '{print $4}')
    
    # Display results
    echo "Seq         : $seq_time s"
    echo "OpenMP      : $omp_time s"
    echo "MPI+OpenMP  : $mpi_omp_time s"
    if [ "$hybrid_time" != "N/A" ]; then
        echo "Hybrid      : $hybrid_time s"
    else
        echo "Hybrid      : Not available"
    fi
    if [ "$cuda_time" != "N/A" ]; then
        echo "CUDA        : $cuda_time s"
    else
        echo "CUDA        : Not available"
    fi
    echo "MPI Frames  : $mpi_frames_time s"
    echo "MPI Domain  : $mpi_domain_time s"
    echo "-------------"
    
    # Append results to CSV file (fields separated by commas)
    echo "$img_name,$seq_time,$omp_time,$mpi_omp_time,$hybrid_time,$cuda_time,$mpi_frames_time,$mpi_domain_time" >> "$CSV_FILE"
done

# Print summary table header
echo -e "\n${GREEN}Summary Table:${NC}"
printf "$FORMAT" "Image" "Specs" "Sequential" "OpenMP" "MPI+OpenMP" "Hybrid" "CUDA" "MPI Frames" "MPI Domain"
echo "--------------------------------------------------------------------------------------------------------------"

for img_info in "${IMAGES[@]}"; do
    img_name=${img_info%%:*}
    img_path=${img_info#*:}
    image_specs=$(get_image_info "$img_path")
    
    # Run quick tests to get times
    seq_output_text=$(./sobelf "$img_path" /dev/null 2>&1)
    seq_time=$(echo "$seq_output_text" | grep "SOBEL done in" | awk '{print $4}')
    [ -z "$seq_time" ] && continue
    
    omp_output_text=$(./sobelf_omp "$img_path" /dev/null $THREADS 2>&1)
    omp_time=$(echo "$omp_output_text" | grep "SOBEL done in" | awk '{print $4}')
    
    mpi_omp_output_text=$(mpirun -np $MPI_PROCS ./sobelf_omp_mpi "$img_path" /dev/null $THREADS 2>&1)
    mpi_omp_time=$(echo "$mpi_omp_output_text" | grep "SOBEL done in" | awk '{print $4}')
    
    if [ $HYBRID_AVAILABLE -eq 1 ]; then
        hybrid_output_text=$(mpirun -np $MPI_PROCS ./sobelf_hybrid_3 "$img_path" /dev/null $THREADS 2>&1)
        hybrid_time=$(echo "$hybrid_output_text" | grep "SOBEL done in" | awk '{print $4}')
    else
        hybrid_time="N/A"
    fi
    
    if [ $CUDA_AVAILABLE -eq 1 ]; then
        cuda_output_text=$(mpirun -np 1 ./sobelf_cuda "$img_path" /dev/null 2>&1)
        cuda_time=$(echo "$cuda_output_text" | grep "GPU filters done in" | awk '{print $5}')
    else
        cuda_time="N/A"
    fi
    
    mpi_frames_output_text=$(mpirun -np $MPI_PROCS ./sobelf_mpi "$img_path" /dev/null $THREADS 2>&1)
    mpi_frames_time=$(echo "$mpi_frames_output_text" | grep "SOBEL done in" | awk '{print $4}')
    
    mpi_domain_output_text=$(mpirun -np $MPI_PROCS ./sobelf_mpi_domain "$img_path" /dev/null $THREADS 2>&1)
    mpi_domain_time=$(echo "$mpi_domain_output_text" | grep "SOBEL done in" | awk '{print $4}')
    
    printf "$FORMAT" "$img_name" "$image_specs" "${seq_time}s" "${omp_time}s" "${mpi_omp_time}s" "${hybrid_time}" "${cuda_time}s" "${mpi_frames_time}s" "${mpi_domain_time}s"
done

echo -e "\n${GREEN}Summary:${NC}"
echo "  - Sequential: Base performance"
echo "  - OpenMP: Using $THREADS threads"
echo "  - MPI+OpenMP: Using $MPI_PROCS processes with $THREADS threads each"
if [ $HYBRID_AVAILABLE -eq 1 ]; then
    echo "  - Hybrid: Using $MPI_PROCS processes with $THREADS threads each plus CUDA GPU acceleration"
else
    echo "  - Hybrid: Not available (CUDA not found or compilation failed)"
fi
if [ $CUDA_AVAILABLE -eq 1 ]; then
    echo "  - CUDA: Using GPU acceleration only"
else
    echo "  - CUDA: Not available"
fi
echo "  - MPI Frames: Using $MPI_PROCS processes for frame distribution"
echo "  - MPI Domain: Using $MPI_PROCS processes for domain decomposition"
echo "====================================================================================="

echo -e "${BLUE}Detailed CSV results saved in ${CSV_FILE}${NC}"
