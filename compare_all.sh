#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Source the environment settings first
if [ -f "./set_env.sh" ]; then
    echo -e "${YELLOW}Loading environment settings from set_env.sh${NC}"
    source ./set_env.sh
else
    echo -e "${RED}Warning: set_env.sh not found. CUDA detection may fail.${NC}"
fi

# Check if CUDA environment variables are set correctly
if [ -n "$CUDA_ROOT" ]; then
    echo -e "${GREEN}CUDA_ROOT is set to: $CUDA_ROOT${NC}"
    if [ -d "$CUDA_ROOT/bin" ] && [ -d "$CUDA_ROOT/lib64" ]; then
        echo -e "${GREEN}CUDA directories look valid${NC}"
    else
        echo -e "${RED}Warning: CUDA_ROOT does not point to valid CUDA installation${NC}"
    fi
else
    echo -e "${YELLOW}CUDA_ROOT not set. Will try to detect CUDA automatically.${NC}"
fi

# Check if nvcc is in PATH
command -v nvcc >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}CUDA compiler (nvcc) found in PATH${NC}"
    nvcc_path=$(which nvcc)
    echo -e "${GREEN}nvcc path: $nvcc_path${NC}"
else
    echo -e "${RED}CUDA compiler (nvcc) not found in PATH. Check your CUDA installation.${NC}"
fi

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

# Source the environment settings
if [ -f "./set_env.sh" ]; then
    echo -e "${YELLOW}Loading environment settings from set_env.sh${NC}"
    source ./set_env.sh
fi

# Get CUDA path from nvcc if available
if command -v nvcc >/dev/null 2>&1; then
    nvcc_path=$(which nvcc)
    echo -e "${GREEN}Found nvcc at: $nvcc_path${NC}"
    
    # Extract CUDA installation path from nvcc path
    CUDA_INSTALLATION_PATH=$(dirname $(dirname $nvcc_path))
    echo -e "${GREEN}Detected CUDA installation at: $CUDA_INSTALLATION_PATH${NC}"
    
    # Check if the detected path contains include/cuda_runtime.h
    if [ -f "$CUDA_INSTALLATION_PATH/include/cuda_runtime.h" ]; then
        CUDA_PATH="$CUDA_INSTALLATION_PATH"
        CUDA_AVAILABLE=1
        echo -e "${GREEN}CUDA headers found at: $CUDA_PATH/include${NC}"
    # Use CUDA_ROOT if set in environment
    elif [ -n "$CUDA_ROOT" ] && [ -f "$CUDA_ROOT/include/cuda_runtime.h" ]; then
        CUDA_PATH="$CUDA_ROOT"
        CUDA_AVAILABLE=1
        echo -e "${GREEN}Using CUDA from environment: $CUDA_PATH${NC}"
    else
        echo -e "${YELLOW}CUDA compiler found but CUDA headers not detected at standard locations.${NC}"
        echo -e "${YELLOW}Hybrid version will be disabled.${NC}"
    fi
    
    # Check for libcudart.so for linking
    if [ $CUDA_AVAILABLE -eq 1 ]; then
        if [ -f "$CUDA_PATH/lib64/libcudart.so" ]; then
            echo -e "${GREEN}CUDA runtime library found at: $CUDA_PATH/lib64/libcudart.so${NC}"
        elif [ -f "$CUDA_PATH/lib/libcudart.so" ]; then
            echo -e "${GREEN}CUDA runtime library found at: $CUDA_PATH/lib/libcudart.so${NC}"
        else
            echo -e "${RED}Warning: CUDA runtime library (libcudart.so) not found!${NC}"
            echo -e "${YELLOW}This may cause linking errors when compiling CUDA code.${NC}"
        fi
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
    echo -e "${YELLOW}Compiling Hybrid MPI+OpenMP+CUDA version...${NC}"

    # Check if the source files exist
    if [ ! -f "sobelf_hybrid_3.c" ] || [ ! -f "sobelf_hybrid_3.cu" ]; then
        echo -e "${YELLOW}Looking for hybrid source files...${NC}"
        # Check if they might be in the src directory
        if [ -f "src/sobelf_hybrid_3.c" ] || [ -f "src/sobelf_cuda.cu" ]; then
            echo -e "${GREEN}Found hybrid source files in src/ directory. Copying to root.${NC}"
            [ -f "src/sobelf_hybrid_3.c" ] && cp src/sobelf_hybrid_3.c . || echo "Missing sobelf_hybrid_3.c"
            # Use sobelf_cuda.cu as a fallback if sobelf_hybrid_3.cu doesn't exist
            if [ -f "src/sobelf_hybrid_3.cu" ]; then
                cp src/sobelf_hybrid_3.cu .
            elif [ -f "src/sobelf_cuda.cu" ]; then
                echo -e "${YELLOW}Using src/sobelf_cuda.cu as a fallback for sobelf_hybrid_3.cu${NC}"
                cp src/sobelf_cuda.cu sobelf_hybrid_3.cu
            else
                echo -e "${RED}No CUDA source file found.${NC}"
            fi
        else
            echo -e "${RED}Hybrid source files not found. Skipping hybrid compilation.${NC}"
            echo -e "${YELLOW}Expected files: sobelf_hybrid_3.c and sobelf_hybrid_3.cu${NC}"
            echo -e "${YELLOW}Checked in current directory and src/ directory${NC}"
            CUDA_AVAILABLE=0
        fi
    fi
    
    if [ $CUDA_AVAILABLE -eq 1 ]; then
        # Create Makefile.hybrid if it doesn't exist
        if [ ! -f Makefile.hybrid ]; then
            echo "Creating Makefile.hybrid..."
            cat > Makefile.hybrid << EOF
CC = mpicc
NVCC = nvcc
CUDA_PATH = $CUDA_PATH
CFLAGS = -O3 -Iinclude -fopenmp
# Simplified CUDA flags for better compatibility
NVCC_FLAGS = -I\$(CUDA_PATH)/include -Xcompiler -fopenmp
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

        # Now try to compile
        echo -e "${YELLOW}Running hybrid compilation...${NC}"
        make -f Makefile.hybrid clean
        # Capture the exit status directly
        make -f Makefile.hybrid sobelf_hybrid_3 2>&1 | tee hybrid_compile.log
        MAKE_STATUS=${PIPESTATUS[0]}

        if [ $MAKE_STATUS -eq 0 ] && [ -f "./sobelf_hybrid_3" ]; then
            HYBRID_AVAILABLE=1
            echo -e "${GREEN}Hybrid MPI+OpenMP+CUDA version successfully compiled${NC}"
        else
            if [ $MAKE_STATUS -ne 0 ]; then
                echo -e "${RED}Failed to compile Hybrid MPI+OpenMP+CUDA version.${NC}"
            else
                echo -e "${RED}Compilation reported success but executable not found${NC}"
            fi
            echo -e "${YELLOW}See hybrid_compile.log for details.${NC}"
            echo -e "${YELLOW}Disabling hybrid mode for this run.${NC}"
            HYBRID_AVAILABLE=0
        fi
    fi
else
    echo -e "${YELLOW}Skipping Hybrid MPI+OpenMP+CUDA version compilation (CUDA not available)${NC}"
fi

echo -e "${GREEN}Compilation of all versions complete.${NC}"

# Set up parameters
INPUT_DIR=images/original
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

# Add CUDA-only version explicitly after CUDA detection
CUDA_ONLY_AVAILABLE=0

# Check if our sobelf_cuda exists or can be compiled
if [ -f "./sobelf_cuda" ]; then
    CUDA_ONLY_AVAILABLE=1
    echo -e "${GREEN}CUDA-only version detected, will be included in the comparison${NC}"
else
    echo -e "${YELLOW}Attempting to compile CUDA-only version...${NC}"
    source set_env.sh >/dev/null 2>&1 # Source environment to get CUDA paths
    make -f Makefile.cuda >/dev/null 2>&1
    if [ -f "./sobelf_cuda" ]; then
        CUDA_ONLY_AVAILABLE=1
        echo -e "${GREEN}CUDA-only version compiled successfully${NC}"
    else
        echo -e "${RED}Failed to compile CUDA-only version${NC}"
        echo -e "${YELLOW}Make sure CUDA is properly installed and environment variables are set${NC}"
    fi
fi

# Update FORMAT to include CUDA column
FORMAT="%-25s | %-15s | %-15s | %-15s | %-15s | %-15s | %-15s\n"

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
    cuda_output="$OUTPUT_DIR/${img_name}_cuda_out.gif"  # Add CUDA output path
    
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
    
    # Run CUDA-only version if available
    cuda_time=""
    if [ $CUDA_ONLY_AVAILABLE -eq 1 ]; then
        # Source environment first to ensure proper CUDA libraries are found
        echo -e "${YELLOW}Running CUDA version for $img_name${NC}"
        cuda_output_text=$(source set_env.sh > /dev/null 2>&1 && mpirun -np 1 ./sobelf_cuda "$img_path" "$cuda_output" 2>&1)
        cuda_time=$(echo "$cuda_output_text" | grep "SOBEL done in" | awk '{print $4}')
        if [ -z "$cuda_time" ]; then
            echo -e "${RED}Failed to extract CUDA processing time${NC}"
            echo -e "${YELLOW}CUDA output: ${NC}"
            echo "$cuda_output_text"
            cuda_result="N/A"
        else
            cuda_speedup=$(echo "scale=2; $seq_time/$cuda_time" | bc)
            cuda_result="$cuda_time s (${cuda_speedup}x)"
        fi
    else
        cuda_result="Not available"
    fi
    
    # Run Hybrid MPI+OpenMP+CUDA version if available
    hybrid_time=""
    if [ $HYBRID_AVAILABLE -eq 1 ] && [ -f "./sobelf_hybrid_3" ]; then
        hybrid_output_text=$(mpirun -np $MPI_PROCS ./sobelf_hybrid_3 "$img_path" "$hybrid_output" $THREADS 2>&1)
        hybrid_time=$(echo "$hybrid_output_text" | grep "SOBEL done in" | awk '{print $4}')
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
        hybrid_result="$hybrid_time s ($hybrid_speedup speedup)"
    else
        hybrid_time="N/A"
        hybrid_speedup="N/A"
        hybrid_result="Not available (CUDA compilation failed)"
    fi
    
    # Print results for this image in the requested format
    echo "Seq      : $seq_time s"
    echo "OpenMP   : $omp_time s ($omp_speedup speedup)"
    echo "OMP+MPI  : $mpi_time s ($mpi_speedup speedup)"
    
    if [ $CUDA_ONLY_AVAILABLE -eq 1 ]; then
        echo "CUDA     : $cuda_time s ($cuda_speedup speedup)"
    else
        echo "CUDA     : Not available (CUDA not found or compilation failed)"
    fi
    
    if [ $HYBRID_AVAILABLE -eq 1 ]; then
        echo "All three: $hybrid_time s ($hybrid_speedup speedup)"
    else
        echo "All three: Not available (CUDA not found or compilation failed)"
    fi
    
    echo "-------------"
done

echo -e "\n${GREEN}Summary Table:${NC}"
echo -e "${BLUE}Image Characteristics and Performance Comparison${NC}"
printf "${FORMAT}" "Image" "Specs" "Sequential" "OpenMP" "MPI+OpenMP" "CUDA-only" "Hybrid"
echo "------------------------------------------------------------------------------------------------------------------------"

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
    
    # Run CUDA-only version if available
    if [ $CUDA_ONLY_AVAILABLE -eq 1 ]; then
        cuda_output_text=$(source set_env.sh > /dev/null 2>&1 && mpirun -np 1 ./sobelf_cuda "$img_path" /dev/null 2>&1)
        cuda_time=$(echo "$cuda_output_text" | grep "SOBEL done in" | awk '{print $4}')
        if [ -z "$cuda_time" ]; then
            echo -e "${RED}Failed to extract CUDA processing time${NC}"
            echo -e "${YELLOW}CUDA output: ${NC}"
            echo "$cuda_output_text"
            cuda_result="N/A"
        else
            cuda_speedup=$(echo "scale=2; $seq_time/$cuda_time" | bc)
            cuda_result="$cuda_time s (${cuda_speedup}x)"
        fi
    else
        cuda_result="Not available"
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
    printf "${FORMAT}" "$img_name" "$image_specs" "$seq_time s" "$omp_result" "$mpi_result" "$cuda_result" "$hybrid_result"
done

echo -e "\n${GREEN}Summary:${NC}"
echo "  - Sequential: Base performance"
echo "  - OpenMP: Using $THREADS threads"
echo "  - MPI+OpenMP: Using $MPI_PROCS processes with $THREADS threads each"

if [ $CUDA_ONLY_AVAILABLE -eq 1 ]; then
    echo "  - CUDA-only: Using 1 process with GPU acceleration"
else
    echo "  - CUDA-only: Not available (CUDA not found or compilation failed)"
fi

if [ $HYBRID_AVAILABLE -eq 1 ]; then
    echo "  - Hybrid: Using $MPI_PROCS processes with $THREADS threads each plus CUDA GPU acceleration"
else
    echo "  - Hybrid: Not available (CUDA not found or compilation failed)"
fi

echo "  - Higher speedup values indicate better parallel performance"
echo "=====================================================================================" 
