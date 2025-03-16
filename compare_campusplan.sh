#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "====================================================================================="
echo -e "${GREEN}Scalability Analysis: Campusplan Images${NC}"
echo "====================================================================================="

# Compile all versions if needed
echo -e "${YELLOW}Checking compilations...${NC}"

# Check sequential version
if [ ! -f ./sobelf ]; then
    echo "Compiling sequential version..."
    make sobelf
fi

# Check OpenMP version
if [ ! -f ./sobelf_omp ]; then
    echo "Compiling OpenMP version..."
    make sobelf_omp
fi

# Check OpenMP+MPI version
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
fi

# Set up parameters
INPUT_DIR=images/original
OUTPUT_DIR=output
mkdir -p $OUTPUT_DIR 2>/dev/null

# Test images 
IMAGES=(
    "Campusplan-Hausnr"
    "Campusplan-Mobilitaetsbeschraenkte"
)

# Thread configurations to test
THREAD_CONFIGS=(1 2 4 8)

# MPI process configurations to test
MPI_CONFIGS=(1 2 4)

# Run sequential baseline test first
echo -e "\n${BLUE}Running sequential baseline tests...${NC}"
echo "-------------------------------------------------------------------------------------"
echo -e "${BLUE}Image                          Sequential Time (s)${NC}"
echo "-------------------------------------------------------------------------------------"

declare -A SEQ_TIMES

for img in "${IMAGES[@]}"; do
    input_file="$INPUT_DIR/$img.gif"
    seq_output="$OUTPUT_DIR/${img}_seq_out.gif"
    
    # Skip if image doesn't exist
    if [ ! -f "$input_file" ]; then
        echo "Image $img not found, skipping."
        continue
    fi
    
    echo "Running sequential test for $img..."
    
    # Run sequential version with timing
    seq_output_text=$(./sobelf "$input_file" "$seq_output" 2>&1)
    seq_time=$(echo "$seq_output_text" | grep "SOBEL done in" | awk '{print $4}')
    SEQ_TIMES["$img"]=$seq_time
    
    printf "%-30s %-20s\n" "$img" "$seq_time"
done

# Run OpenMP tests with different thread counts
echo -e "\n${BLUE}Running OpenMP tests with different thread counts...${NC}"
echo "-------------------------------------------------------------------------------------"
echo -e "${BLUE}Image                          Threads  Time (s)    Speedup${NC}"
echo "-------------------------------------------------------------------------------------"

for img in "${IMAGES[@]}"; do
    input_file="$INPUT_DIR/$img.gif"
    
    # Skip if image doesn't exist
    if [ ! -f "$input_file" ]; then
        continue
    fi
    
    for threads in "${THREAD_CONFIGS[@]}"; do
        omp_output="$OUTPUT_DIR/${img}_omp_${threads}_out.gif"
        
        echo "Running OpenMP test for $img with $threads threads..."
        
        # Run OpenMP version with timing
        omp_output_text=$(./sobelf_omp "$input_file" "$omp_output" $threads 2>&1)
        omp_time=$(echo "$omp_output_text" | grep "SOBEL done in" | awk '{print $4}')
        
        # Calculate speedup
        speedup=$(echo "scale=2; ${SEQ_TIMES[$img]}/$omp_time" | bc)
        
        printf "%-30s %-8s %-11s %-8s\n" "$img" "$threads" "$omp_time" "$speedup"
    done
done

# Run MPI+OpenMP tests with different configurations
echo -e "\n${BLUE}Running MPI+OpenMP tests with different configurations...${NC}"
echo "-------------------------------------------------------------------------------------"
echo -e "${BLUE}Image                          Processes  Threads   Time (s)    Speedup${NC}"
echo "-------------------------------------------------------------------------------------"

for img in "${IMAGES[@]}"; do
    input_file="$INPUT_DIR/$img.gif"
    
    # Skip if image doesn't exist
    if [ ! -f "$input_file" ]; then
        continue
    fi
    
    for procs in "${MPI_CONFIGS[@]}"; do
        for threads in "${THREAD_CONFIGS[@]}"; do
            mpi_output="$OUTPUT_DIR/${img}_mpi_${procs}p_${threads}t_out.gif"
            
            echo "Running MPI+OpenMP test for $img with $procs processes and $threads threads..."
            
            # Run MPI+OpenMP version with timing
            mpi_output_text=$(mpirun -np $procs ./sobelf_omp_mpi "$input_file" "$mpi_output" $threads 2>&1)
            mpi_time=$(echo "$mpi_output_text" | grep "SOBEL done in" | awk '{print $4}')
            
            # Calculate speedup
            speedup=$(echo "scale=2; ${SEQ_TIMES[$img]}/$mpi_time" | bc)
            
            printf "%-30s %-10s %-9s %-11s %-8s\n" "$img" "$procs" "$threads" "$mpi_time" "$speedup"
        done
    done
done

echo "====================================================================================="
echo -e "${GREEN}Analysis Complete${NC}"
echo "  - Sequential: Base performance"
echo "  - OpenMP: Tested with ${#THREAD_CONFIGS[@]} different thread configurations"
echo "  - MPI+OpenMP: Tested with ${#MPI_CONFIGS[@]} process configs and ${#THREAD_CONFIGS[@]} thread configs"
echo "  - Higher speedup values indicate better parallel performance"
echo "=====================================================================================" 
