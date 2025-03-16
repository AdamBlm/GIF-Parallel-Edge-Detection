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
echo "Compiling sequential version..."
make sobelf

# Compile OpenMP version
echo "Compiling OpenMP version..."
make sobelf_omp

# Compile OpenMP+MPI version
echo "Compiling OpenMP+MPI version..."
# Create Makefile.mpi if it doesn't exist
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

# Check hybrid implementation (MPI+OpenMP+CUDA) only if CUDA is available
HYBRID_AVAILABLE=0
if [ $CUDA_AVAILABLE -eq 1 ]; then
    echo "Compiling Hybrid MPI+OpenMP+CUDA version..."
    # Create Makefile.hybrid if it doesn't exist
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

echo -e "${GREEN}Compilation of all versions complete.${NC}"

# Set up parameters
INPUT_DIR=images
OUTPUT_DIR=output
mkdir -p $OUTPUT_DIR 2>/dev/null

# Clean the output directory first
echo -e "${YELLOW}Cleaning output directory...${NC}"
rm -f $OUTPUT_DIR/* 2>/dev/null
echo -e "${GREEN}Output directory cleaned.${NC}"

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

# Format for the title row
FORMAT="%-25s | %-15s | %-15s | %-15s | %-15s | %-15s\n"

# Find all GIF files in the images directory
IMAGES=()
for img_path in $(find $INPUT_DIR -name "*.gif" 2>/dev/null); do
    img_file=$(basename "$img_path")
    img_name="${img_file%.gif}"
    IMAGES+=("$img_name:$img_path")
done

# Process each image
for img_info in "${IMAGES[@]}"; do
    # Split image info
    img_name=${img_info%%:*}
    img_path=${img_info#*:}
    
    # Set output paths
    seq_output="$OUTPUT_DIR/${img_name}_seq_out.gif"
    omp_output="$OUTPUT_DIR/${img_name}_omp_out.gif"
    mpi_output="$OUTPUT_DIR/${img_name}_mpi_out.gif"
    hybrid_output="$OUTPUT_DIR/${img_name}_hybrid_out.gif"
    
    echo -e "\n${GREEN}$img_path${NC}"
    
    # Get and display image information
    image_specs=$(get_image_info "$img_path")
    echo -e "${YELLOW}Image specs: $image_specs${NC}"
    
    # Run sequential version
    seq_output_text=$(./sobelf "$img_path" "$seq_output" 2>&1)
    seq_time=$(echo "$seq_output_text" | grep "SOBEL done in" | awk '{print $4}')
    
    # If sequential execution failed, skip this image
    if [ -z "$seq_time" ]; then
        echo "Failed to process $img_name with sequential version, skipping."
        continue
    fi
    
    # Run OpenMP version
    omp_output_text=$(./sobelf_omp "$img_path" "$omp_output" $THREADS 2>&1)
    omp_time=$(echo "$omp_output_text" | grep "SOBEL done in" | awk '{print $4}')
    
    # Run MPI+OpenMP version
    mpi_output_text=$(mpirun -np $MPI_PROCS ./sobelf_omp_mpi "$img_path" "$mpi_output" $THREADS 2>&1)
    mpi_time=$(echo "$mpi_output_text" | grep "SOBEL done in" | awk '{print $4}')
    
    # Run Hybrid MPI+OpenMP+CUDA version if available
    if [ $HYBRID_AVAILABLE -eq 1 ]; then
        hybrid_output_text=$(mpirun -np $MPI_PROCS ./sobelf_hybrid_3 "$img_path" "$hybrid_output" $THREADS 2>&1)
        hybrid_time=$(echo "$hybrid_output_text" | grep "SOBEL done in" | awk '{print $4}')
    else
        hybrid_time=""
    fi
    
    # Calculate speedups
    if [ ! -z "$omp_time" ]; then
        omp_speedup=$(echo "scale=2; $seq_time/$omp_time" | bc)
    else
        omp_time="N/A"
        omp_speedup="N/A"
    fi
    
    if [ ! -z "$mpi_time" ]; then
        mpi_speedup=$(echo "scale=2; $seq_time/$mpi_time" | bc)
    else
        mpi_time="N/A"
        mpi_speedup="N/A"
    fi
    
    if [ ! -z "$hybrid_time" ]; then
        hybrid_speedup=$(echo "scale=2; $seq_time/$hybrid_time" | bc)
    else
        hybrid_time="N/A"
        hybrid_speedup="N/A"
    fi
    
    # Print results for this image in the requested format
    echo "Seq      : $seq_time s"
    echo "OpenMP   : $omp_time s ($omp_speedup speedup)"
    echo "OMP+MPI  : $mpi_time s ($mpi_speedup speedup)"
    
    if [ $HYBRID_AVAILABLE -eq 1 ]; then
        echo "All three: $hybrid_time s ($hybrid_speedup speedup)"
    else
        echo "All three: Not available (CUDA not found or compilation failed)"
    fi
    
    echo "-------------"
done

echo -e "\n${GREEN}Summary Table:${NC}"
echo -e "${BLUE}Image Characteristics and Performance Comparison${NC}"
printf "${FORMAT}" "Image" "Specs" "Sequential" "OpenMP" "MPI+OpenMP" "Hybrid"
echo "---------------------------------------------------------------------------------------------------"

for img_info in "${IMAGES[@]}"; do
    # Split image info
    img_name=${img_info%%:*}
    img_path=${img_info#*:}
    
    # Get image specs
    image_specs=$(get_image_info "$img_path")
    
    # Run sequential version (quick run just to get time)
    seq_output_text=$(./sobelf "$img_path" /dev/null 2>&1)
    seq_time=$(echo "$seq_output_text" | grep "SOBEL done in" | awk '{print $4}')
    
    # If sequential execution failed, skip this image
    if [ -z "$seq_time" ]; then
        continue
    fi
    
    # Run OpenMP version
    omp_output_text=$(./sobelf_omp "$img_path" /dev/null $THREADS 2>&1)
    omp_time=$(echo "$omp_output_text" | grep "SOBEL done in" | awk '{print $4}')
    if [ ! -z "$omp_time" ]; then
        omp_speedup=$(echo "scale=2; $seq_time/$omp_time" | bc)
        omp_result="$omp_time s (${omp_speedup}x)"
    else
        omp_result="N/A"
    fi
    
    # Run MPI+OpenMP version
    mpi_output_text=$(mpirun -np $MPI_PROCS ./sobelf_omp_mpi "$img_path" /dev/null $THREADS 2>&1)
    mpi_time=$(echo "$mpi_output_text" | grep "SOBEL done in" | awk '{print $4}')
    if [ ! -z "$mpi_time" ]; then
        mpi_speedup=$(echo "scale=2; $seq_time/$mpi_time" | bc)
        mpi_result="$mpi_time s (${mpi_speedup}x)"
    else
        mpi_result="N/A"
    fi
    
    # Run Hybrid MPI+OpenMP+CUDA version if available
    if [ $HYBRID_AVAILABLE -eq 1 ]; then
        hybrid_output_text=$(mpirun -np $MPI_PROCS ./sobelf_hybrid_3 "$img_path" /dev/null $THREADS 2>&1)
        hybrid_time=$(echo "$hybrid_output_text" | grep "SOBEL done in" | awk '{print $4}')
        if [ ! -z "$hybrid_time" ]; then
            hybrid_speedup=$(echo "scale=2; $seq_time/$hybrid_time" | bc)
            hybrid_result="$hybrid_time s (${hybrid_speedup}x)"
        else
            hybrid_result="N/A"
        fi
    else
        hybrid_result="Not available"
    fi
    
    # Print results row
    printf "${FORMAT}" "$img_name" "$image_specs" "$seq_time s" "$omp_result" "$mpi_result" "$hybrid_result"
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

echo "  - Higher speedup values indicate better parallel performance"
echo "=====================================================================================" 
