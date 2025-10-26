/*
 * INF560
 *
 * Image Filtering Project - Hybrid MPI + CUDA Implementation
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <mpi.h>
#include <cuda_runtime.h>

#include "gif_lib.h"
#include "gif_utils.h"

/* Set this macro to 1 to enable debugging information */
#define SOBELF_DEBUG 0

/* Define MPI tags for messaging */
#define MPI_TAG_IMAGE_DATA 100
#define MPI_TAG_IMAGE_DIMS 101
#define MPI_TAG_RESULT 102

/* Tile size for memory-efficient GPU processing */
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

/* CUDA error checking macro */
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/* Struct to send image data between processes */
typedef struct image_data {
    int width;
    int height;
    int image_idx;
    int total_size;
} image_data;

/* Helper for collective operations */
typedef struct scatter_info {
    int *sendcounts;  /* Number of elements to send to each process */
    int *displs;      /* Displacement for each process */
    int *image_counts; /* Number of images per process */
    int *image_displs; /* Displacement for image indices */
    int *scatter_byte_counts; /* Byte counts for MPI transfers - precomputed */
    int *scatter_byte_displs; /* Byte displacements - precomputed */
} scatter_info;

/* Using std::min to avoid conflicts with CUDA's own min function */
// static inline int min(int a, int b) {
//     return (a < b) ? a : b;
// }

#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)

/*
 * Load a GIF image from a file and return a
 * structure of type animated_gif.
 */
//animated_gif *
//load_pixels(char *filename);

/* Output GIF image */
//int 
//output_modified_read_gif(char *filename, GifFileType *g);

/* Store pixels in GIF file */
//int
//store_pixels(char *filename, animated_gif *image);

/* Create scatter/gather information for collective operations */
scatter_info* create_scatter_info(int n_images, int size)
{
    scatter_info* info = (scatter_info*)malloc(sizeof(scatter_info));
    if (!info) return NULL;
    
    info->sendcounts = (int*)calloc(size, sizeof(int));
    info->displs = (int*)calloc(size, sizeof(int));
    info->image_counts = (int*)calloc(size, sizeof(int));
    info->image_displs = (int*)calloc(size, sizeof(int));
    info->scatter_byte_counts = (int*)calloc(size, sizeof(int));
    info->scatter_byte_displs = (int*)calloc(size, sizeof(int));
    
    if (!info->sendcounts || !info->displs || 
        !info->image_counts || !info->image_displs ||
        !info->scatter_byte_counts || !info->scatter_byte_displs) {
        // Free allocated memory if any allocation failed
        if (info->sendcounts) free(info->sendcounts);
        if (info->displs) free(info->displs);
        if (info->image_counts) free(info->image_counts);
        if (info->image_displs) free(info->image_displs);
        if (info->scatter_byte_counts) free(info->scatter_byte_counts);
        if (info->scatter_byte_displs) free(info->scatter_byte_displs);
        free(info);
        return NULL;
    }
    
    // Distribute images evenly among processes
    int base_count = n_images / size;
    int remainder = n_images % size;
    
    for (int i = 0; i < size; i++) {
        info->image_counts[i] = base_count + (i < remainder ? 1 : 0);
        if (i > 0) {
            info->image_displs[i] = info->image_displs[i-1] + info->image_counts[i-1];
        }
    }
    
    return info;
}

/* Free scatter/gather information */
void free_scatter_info(scatter_info *info)
{
    if (info) {
        if (info->sendcounts) free(info->sendcounts);
        if (info->displs) free(info->displs);
        if (info->image_counts) free(info->image_counts);
        if (info->image_displs) free(info->image_displs);
        if (info->scatter_byte_counts) free(info->scatter_byte_counts);
        if (info->scatter_byte_displs) free(info->scatter_byte_displs);
        free(info);
    }
}

/* Calculate scatter/gather counts for pixel data */
void calculate_pixel_counts(scatter_info* scatter_data, int* widths, int* heights, int size) {
    if (!scatter_data || !widths || !heights) return;
    
    // Calculate displacements and pixel counts
    for (int i = 0; i < size; i++) {
        int start_img = scatter_data->image_displs[i];
        int end_img = start_img + scatter_data->image_counts[i];
        
        scatter_data->sendcounts[i] = 0;
        for (int j = start_img; j < end_img; j++) {
            scatter_data->sendcounts[i] += widths[j] * heights[j];
        }
        
        if (i > 0) {
            scatter_data->displs[i] = scatter_data->displs[i-1] + scatter_data->sendcounts[i-1];
        }
        
        // Pre-calculate byte counts and displacements to avoid redundant calculations
        scatter_data->scatter_byte_counts[i] = scatter_data->sendcounts[i] * sizeof(pixel);
        scatter_data->scatter_byte_displs[i] = scatter_data->displs[i] * sizeof(pixel);
    }
}

/* CUDA Kernels for image processing */

// Grayscale conversion kernel
__global__ void grayscale_kernel(pixel *d_pixels, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        int moy = (d_pixels[idx].r + d_pixels[idx].g + d_pixels[idx].b) / 3;
        if (moy < 0) moy = 0;
        if (moy > 255) moy = 255;
        
        d_pixels[idx].r = moy;
        d_pixels[idx].g = moy;
        d_pixels[idx].b = moy;
    }
}

// Blur filter kernel with shared memory tiling for better memory access patterns
__global__ void blur_kernel(pixel *d_in, pixel *d_out, int width, int height, int size)
{
    extern __shared__ pixel sharedMem[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int x = bx + tx;
    int y = by + ty;
    
    // Shared memory dimensions including halo
    int sharedWidth = blockDim.x + 2 * size;
    int sharedHeight = blockDim.y + 2 * size;
    
    // Load center and halo regions into shared memory
    for (int j = ty; j < sharedHeight; j += blockDim.y) {
        int y_global = by + j - size;
        
        for (int i = tx; i < sharedWidth; i += blockDim.x) {
            int x_global = bx + i - size;
            
            // Clamp coordinates to valid range
            int valid_x = max(0, min(width - 1, x_global));
            int valid_y = max(0, min(height - 1, y_global));
            
            sharedMem[j * sharedWidth + i] = d_in[valid_y * width + valid_x];
        }
    }
    
    __syncthreads();
    
    // Only process interior pixels
    if (x < width && y < height) {
        int t_r = 0;
        int t_g = 0;
        int t_b = 0;
        int count = 0;
        
        // Compute average using stencil
        for (int j = -size; j <= size; j++) {
            for (int i = -size; i <= size; i++) {
                int sx = tx + i + size;
                int sy = ty + j + size;
                
                t_r += sharedMem[sy * sharedWidth + sx].r;
                t_g += sharedMem[sy * sharedWidth + sx].g;
                t_b += sharedMem[sy * sharedWidth + sx].b;
                count++;
            }
        }
        
        // Write output
        int idx = y * width + x;
        d_out[idx].r = t_r / count;
        d_out[idx].g = t_g / count;
        d_out[idx].b = t_b / count;
    }
}

// Check for convergence of blur iteration
__global__ void check_convergence_kernel(pixel *d_prev, pixel *d_curr, int width, int height, 
                                         int threshold, int *d_continue)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        int diff_r = abs(d_curr[idx].r - d_prev[idx].r);
        int diff_g = abs(d_curr[idx].g - d_prev[idx].g);
        int diff_b = abs(d_curr[idx].b - d_prev[idx].b);
        
        if (diff_r > threshold || diff_g > threshold || diff_b > threshold) {
            *d_continue = 1;
        }
    }
}

// Sobel filter kernel using shared memory for efficient stencil operation
__global__ void sobel_kernel(pixel *d_in, pixel *d_out, int width, int height)
{
    extern __shared__ pixel sharedMem[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int x = bx + tx;
    int y = by + ty;
    
    // Shared memory dimensions including 1-pixel halo
    int sharedWidth = blockDim.x + 2;
    int sharedHeight = blockDim.y + 2;
    
    // Load center and halo regions into shared memory
    for (int j = ty; j < sharedHeight; j += blockDim.y) {
        int y_global = by + j - 1;
        
        for (int i = tx; i < sharedWidth; i += blockDim.x) {
            int x_global = bx + i - 1;
            
            // Clamp coordinates to valid range
            int valid_x = max(0, min(width - 1, x_global));
            int valid_y = max(0, min(height - 1, y_global));
            
            sharedMem[j * sharedWidth + i] = d_in[valid_y * width + valid_x];
        }
    }
    
    __syncthreads();
    
    // Only process interior pixels
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int idx = y * width + x;
        
        // Extract pixel values (r channel) from shared memory
        int p00 = sharedMem[(ty) * sharedWidth + (tx)].r;
        int p01 = sharedMem[(ty) * sharedWidth + (tx+1)].r;
        int p02 = sharedMem[(ty) * sharedWidth + (tx+2)].r;
        int p10 = sharedMem[(ty+1) * sharedWidth + (tx)].r;
        int p12 = sharedMem[(ty+1) * sharedWidth + (tx+2)].r;
        int p20 = sharedMem[(ty+2) * sharedWidth + (tx)].r;
        int p21 = sharedMem[(ty+2) * sharedWidth + (tx+1)].r;
        int p22 = sharedMem[(ty+2) * sharedWidth + (tx+2)].r;
        
        // Sobel operator for red channel
        float Gx_r = -p00 + p02 - 2.0f * p10 + 2.0f * p12 - p20 + p22;
        float Gy_r = -p00 - 2.0f * p01 - p02 + p20 + 2.0f * p21 + p22;
        
        float val_r = sqrtf(Gx_r * Gx_r + Gy_r * Gy_r) / 4.0f;
        
        // Apply binary thresholding
        if (val_r > 15) {  // Using the same threshold as the OpenMP version
            d_out[idx].r = 255;
            d_out[idx].g = 255;
            d_out[idx].b = 255;
        } else {
            d_out[idx].r = 0;
            d_out[idx].g = 0;
            d_out[idx].b = 0;
        }
    } else if (x < width && y < height) {
        // For border pixels, just copy the input
        int idx = y * width + x;
        d_out[idx] = d_in[idx];
    }
}

/* Main CUDA processing function for a single image */
void process_image_cuda(pixel *h_pixels, int width, int height)
{
    if (h_pixels == NULL || width <= 0 || height <= 0) {
        fprintf(stderr, "Error: Invalid input parameters to process_image_cuda\n");
        return;
    }

    // Calculate buffer sizes
    size_t pixelsSize = width * height * sizeof(pixel);
    
    // Allocate device memory with error checking
    pixel *d_pixels = NULL, *d_temp = NULL;
    int *d_continue = NULL;
    cudaError_t cuda_err;
    
    // Allocate device memory with proper error checking
    cuda_err = cudaMalloc((void**)&d_pixels, pixelsSize);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "CUDA error in cudaMalloc d_pixels: %s\n", 
                cudaGetErrorString(cuda_err));
        return;
    }
    
    cuda_err = cudaMalloc((void**)&d_temp, pixelsSize);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "CUDA error in cudaMalloc d_temp: %s\n", 
                cudaGetErrorString(cuda_err));
        cudaFree(d_pixels);
        return;
    }
    
    cuda_err = cudaMalloc((void**)&d_continue, sizeof(int));
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "CUDA error in cudaMalloc d_continue: %s\n", 
                cudaGetErrorString(cuda_err));
        cudaFree(d_pixels);
        cudaFree(d_temp);
        return;
    }
    
    // Copy input data to device with error checking
    cuda_err = cudaMemcpy(d_pixels, h_pixels, pixelsSize, cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "CUDA error in cudaMemcpy HostToDevice: %s\n", 
                cudaGetErrorString(cuda_err));
        cudaFree(d_pixels);
        cudaFree(d_temp);
        cudaFree(d_continue);
        return;
    }
    
    // Setup execution parameters
    dim3 blockSize(TILE_WIDTH, TILE_HEIGHT);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                 (height + blockSize.y - 1) / blockSize.y);
    
    // Verify block size is valid
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (blockSize.x * blockSize.y > deviceProp.maxThreadsPerBlock) {
        fprintf(stderr, "Block size exceeds device limits. Adjusting...\n");
        blockSize.x = 16;
        blockSize.y = 16;
        gridSize.x = (width + blockSize.x - 1) / blockSize.x;
        gridSize.y = (height + blockSize.y - 1) / blockSize.y;
    }
    
    // Apply grayscale filter
    grayscale_kernel<<<gridSize, blockSize>>>(d_pixels, width, height);
    cuda_err = cudaDeviceSynchronize();
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "CUDA error in grayscale_kernel: %s\n", 
                cudaGetErrorString(cuda_err));
        cudaFree(d_pixels);
        cudaFree(d_temp);
        cudaFree(d_continue);
        return;
    }
    
    // Apply blur filter with convergence
    const int blurSize = 3;  // Same as in the OpenMP version
    const int threshold = 0; // Same as in the OpenMP version
    
    // Calculate shared memory size for blur kernel
    size_t sharedMemSizeBlur = (blockSize.x + 2 * blurSize) * 
                              (blockSize.y + 2 * blurSize) * 
                              sizeof(pixel);
    
    // Check if we have enough shared memory
    if (sharedMemSizeBlur > deviceProp.sharedMemPerBlock) {
        fprintf(stderr, "Warning: Shared memory requirement exceeds device limits. Performance may suffer.\n");
        // Fall back to a smaller tile size if needed
        if (sharedMemSizeBlur > deviceProp.sharedMemPerBlock * 2) {
            blockSize.x /= 2;
            blockSize.y /= 2;
            gridSize.x = (width + blockSize.x - 1) / blockSize.x;
            gridSize.y = (height + blockSize.y - 1) / blockSize.y;
            sharedMemSizeBlur = (blockSize.x + 2 * blurSize) * 
                              (blockSize.y + 2 * blurSize) * 
                              sizeof(pixel);
        }
    }
    
    if (threshold > 0) {
        int h_continue;
        int max_iterations = 50;  // Limit iterations to avoid infinite loop
        int iter = 0;
        
        do {
            // Initialize continue flag
            h_continue = 0;
            cuda_err = cudaMemcpy(d_continue, &h_continue, sizeof(int), cudaMemcpyHostToDevice);
            if (cuda_err != cudaSuccess) {
                fprintf(stderr, "CUDA error in cudaMemcpy HostToDevice for d_continue: %s\n", 
                        cudaGetErrorString(cuda_err));
                break;
            }
            
            // Apply blur
            blur_kernel<<<gridSize, blockSize, sharedMemSizeBlur>>>(
                d_pixels, d_temp, width, height, blurSize);
            cuda_err = cudaGetLastError();
            if (cuda_err != cudaSuccess) {
                fprintf(stderr, "CUDA error in blur_kernel: %s\n", 
                        cudaGetErrorString(cuda_err));
                break;
            }
            
            // Check for convergence
            check_convergence_kernel<<<gridSize, blockSize>>>(
                d_pixels, d_temp, width, height, threshold, d_continue);
            cuda_err = cudaGetLastError();
            if (cuda_err != cudaSuccess) {
                fprintf(stderr, "CUDA error in check_convergence_kernel: %s\n", 
                        cudaGetErrorString(cuda_err));
                break;
            }
            
            // Swap buffers
            pixel *temp = d_pixels;
            d_pixels = d_temp;
            d_temp = temp;
            
            // Get continue flag
            cuda_err = cudaMemcpy(&h_continue, d_continue, sizeof(int), cudaMemcpyDeviceToHost);
            if (cuda_err != cudaSuccess) {
                fprintf(stderr, "CUDA error in cudaMemcpy DeviceToHost for d_continue: %s\n", 
                        cudaGetErrorString(cuda_err));
                break;
            }
            
            iter++;
        } while (h_continue && iter < max_iterations);
        
#if SOBELF_DEBUG
        printf("BLUR: number of iterations: %d\n", iter);
#endif
    } else {
        // Just apply blur once
        blur_kernel<<<gridSize, blockSize, sharedMemSizeBlur>>>(
            d_pixels, d_temp, width, height, blurSize);
        cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "CUDA error in blur_kernel: %s\n", 
                    cudaGetErrorString(cuda_err));
            cudaFree(d_pixels);
            cudaFree(d_temp);
            cudaFree(d_continue);
            return;
        }
        
        // Swap buffers
        pixel *temp = d_pixels;
        d_pixels = d_temp;
        d_temp = temp;
    }
    
    // Apply Sobel filter
    size_t sharedMemSizeSobel = (blockSize.x + 2) * (blockSize.y + 2) * sizeof(pixel);
    sobel_kernel<<<gridSize, blockSize, sharedMemSizeSobel>>>(d_pixels, d_temp, width, height);
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "CUDA error in sobel_kernel: %s\n", 
                cudaGetErrorString(cuda_err));
        cudaFree(d_pixels);
        cudaFree(d_temp);
        cudaFree(d_continue);
        return;
    }
    
    // Copy results back to host with error checking
    cuda_err = cudaMemcpy(h_pixels, d_temp, pixelsSize, cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "CUDA error in cudaMemcpy DeviceToHost: %s\n", 
                cudaGetErrorString(cuda_err));
    }
    
    // Free device memory
    cudaFree(d_pixels);
    cudaFree(d_temp);
    cudaFree(d_continue);
    
    // Make sure all CUDA operations are complete
    cudaDeviceSynchronize();
}

// Helper function to free allocated resources
void free_resources(pixel **p, int n_images)
{
    if (p) {
        for (int i = 0; i < n_images; i++) {
            if (p[i]) free(p[i]);
        }
        free(p);
    }
}

/*
 * Main entry point
 */
int main(int argc, char **argv)
{
    char *input_filename;
    char *output_filename;
    animated_gif *image = NULL;
    struct timeval t1, t2;
    double duration;
    int rank, size;
    MPI_Status status;
    scatter_info *scatter_data = NULL;
    int gpu_count = 0;

    /* Initialize MPI with thread support for CUDA */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Check command-line arguments */
    if (argc < 3)
    {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s input.gif output.gif\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    input_filename = argv[1];
    output_filename = argv[2];
    
    /* Initialize CUDA and get GPU count */
    cudaError_t cuda_err = cudaGetDeviceCount(&gpu_count);
    if (cuda_err != cudaSuccess) {
        if (rank == 0) {
            fprintf(stderr, "Error getting CUDA device count: %s\n", cudaGetErrorString(cuda_err));
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("Using %d MPI processes with %d CUDA-capable GPU(s)\n", size, gpu_count);
    }
    
    /* Set GPU device for this MPI process */
    if (gpu_count > 0) {
        int gpu_id = rank % gpu_count;
        
        // Reset device before using it
        cudaDeviceReset();
        
        // Set device with error checking
        cuda_err = cudaSetDevice(gpu_id);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "Process %d: Error setting GPU device %d: %s\n", 
                    rank, gpu_id, cudaGetErrorString(cuda_err));
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        // Set up exclusive process mode for better multi-process support
        cudaSetDeviceFlags(cudaDeviceScheduleYield);
        
        printf("Process %d using GPU %d\n", rank, gpu_id);
        
        // Verify device is working by trying a simple operation
        int *d_test = NULL;
        cuda_err = cudaMalloc((void**)&d_test, sizeof(int));
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "Process %d: Error allocating test memory on GPU %d: %s\n", 
                   rank, gpu_id, cudaGetErrorString(cuda_err));
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        cudaFree(d_test);
        
    } else {
        if (rank == 0) {
            fprintf(stderr, "Warning: No CUDA-capable GPUs detected\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    // Make sure all processes have initialized their GPU before proceeding
    MPI_Barrier(MPI_COMM_WORLD);

    int n_images = 0;
    int *image_widths = NULL;
    int *image_heights = NULL;
    
    /* Process 0 (master) loads the image and distributes work */
    if (rank == 0)
    {
        /* IMPORT Timer start */
        gettimeofday(&t1, NULL);

        /* Load file and store the pixels in array */
        image = load_pixels(input_filename);
        if (image == NULL) { 
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1; 
        }

        /* IMPORT Timer stop */
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
        printf("GIF loaded from file %s with %d image(s) in %lf s\n", 
                input_filename, image->n_images, duration);
                
        n_images = image->n_images;
        image_widths = image->width;
        image_heights = image->height;
        
        // Create scatter info for workload distribution
        scatter_data = create_scatter_info(n_images, size);
        if (!scatter_data) {
            fprintf(stderr, "Failed to allocate scatter info\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        // Calculate pixel counts for each process
        calculate_pixel_counts(scatter_data, image_widths, image_heights, size);
        
        // Check if we have valid data before proceeding
        if (n_images <= 0) {
            fprintf(stderr, "Error: No images found in input file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    /* Broadcast the number of images to all processes */
    MPI_Bcast(&n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    /* Allocate memory for image dimensions on all processes */
    if (rank != 0) {
        image_widths = (int*)malloc(n_images * sizeof(int));
        image_heights = (int*)malloc(n_images * sizeof(int));
        if (!image_widths || !image_heights) {
            fprintf(stderr, "Process %d: Memory allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        // Create scatter info structure on non-root processes too
        scatter_data = create_scatter_info(n_images, size);
        if (!scatter_data) {
            fprintf(stderr, "Process %d: Failed to allocate scatter info\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
    
    /* Broadcast the image dimensions to all processes */
    MPI_Bcast(image_widths, n_images, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(image_heights, n_images, MPI_INT, 0, MPI_COMM_WORLD);

    /* Broadcast scatter info arrays in a single operation */
    int total_scatter_info_size = size * 4;
    int* all_scatter_info = NULL;
    
    if (rank == 0) {
        all_scatter_info = (int*)malloc(total_scatter_info_size * sizeof(int));
        if (!all_scatter_info) {
            fprintf(stderr, "Process 0: Failed to allocate scatter info buffer\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        // Pack all scatter arrays into a single buffer for efficient transfer
        memcpy(&all_scatter_info[0], scatter_data->image_counts, size * sizeof(int));
        memcpy(&all_scatter_info[size], scatter_data->image_displs, size * sizeof(int));
        memcpy(&all_scatter_info[size*2], scatter_data->sendcounts, size * sizeof(int));
        memcpy(&all_scatter_info[size*3], scatter_data->displs, size * sizeof(int));
    } else {
        all_scatter_info = (int*)malloc(total_scatter_info_size * sizeof(int));
        if (!all_scatter_info) {
            fprintf(stderr, "Process %d: Failed to allocate scatter info buffer\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
    
    // One broadcast instead of multiple
    MPI_Bcast(all_scatter_info, total_scatter_info_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        // Unpack the received data
        memcpy(scatter_data->image_counts, &all_scatter_info[0], size * sizeof(int));
        memcpy(scatter_data->image_displs, &all_scatter_info[size], size * sizeof(int));
        memcpy(scatter_data->sendcounts, &all_scatter_info[size*2], size * sizeof(int));
        memcpy(scatter_data->displs, &all_scatter_info[size*3], size * sizeof(int));
        
        // Calculate pixel counts and displacements 
        for (int i = 0; i < size; i++) {
            scatter_data->scatter_byte_counts[i] = scatter_data->sendcounts[i] * sizeof(pixel);
            scatter_data->scatter_byte_displs[i] = scatter_data->displs[i] * sizeof(pixel);
        }
        
        // Recalculate dimensions if needed
        if (n_images > 0) {
            calculate_pixel_counts(scatter_data, image_widths, image_heights, size);
        }
    }
    
    free(all_scatter_info);
    
    /* Sync before starting timing */
    MPI_Barrier(MPI_COMM_WORLD);
    
    /* FILTER Timer start */
    if (rank == 0) {
        gettimeofday(&t1, NULL);
    }

    /* Get information for this process */
    int my_n_images = scatter_data->image_counts[rank];
    int my_start_image = scatter_data->image_displs[rank];
    
    /* Allocate memory for local image processing */
    pixel **my_pixels = NULL;
    int *my_widths = NULL;
    int *my_heights = NULL;
    
    if (my_n_images > 0) {
        /* Allocate memory for local image data */
        my_pixels = (pixel**)malloc(my_n_images * sizeof(pixel*));
        my_widths = (int*)malloc(my_n_images * sizeof(int));
        my_heights = (int*)malloc(my_n_images * sizeof(int));
        
        if (!my_pixels || !my_widths || !my_heights) {
            fprintf(stderr, "Process %d: Memory allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        /* Initialize local image dimensions and allocate buffers */
        for (int i = 0; i < my_n_images; i++) {
            int img_idx = my_start_image + i;
            my_widths[i] = image_widths[img_idx];
            my_heights[i] = image_heights[img_idx];
            
            /* Allocate memory for local image data */
            my_pixels[i] = (pixel*)malloc(my_widths[i] * my_heights[i] * sizeof(pixel));
            
            if (!my_pixels[i]) {
                fprintf(stderr, "Process %d: Memory allocation failed for image %d\n", rank, i);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
    }
    
    /* Create array for scatterv/gatherv */
    pixel *all_pixels = NULL;
    pixel *scattered_pixels = NULL;
    int sendcount = 0;
    
    if (rank == 0) {
        /* Calculate total pixels for all processes */
        int total_pixels = 0;
        for (int i = 0; i < size; i++) {
            total_pixels += scatter_data->sendcounts[i];
        }
        
        /* Allocate buffer for all pixels */
        all_pixels = (pixel*)malloc(total_pixels * sizeof(pixel));
        if (!all_pixels) {
            fprintf(stderr, "Process 0: Memory allocation failed for all_pixels\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        /* Copy all pixels to contiguous buffer */
        int pixel_idx = 0;
        for (int i = 0; i < n_images; i++) {
            memcpy(&all_pixels[pixel_idx], image->p[i], 
                   image_widths[i] * image_heights[i] * sizeof(pixel));
            pixel_idx += image_widths[i] * image_heights[i];
        }
    }
    
    /* Allocate receive buffer for scattered pixels */
    sendcount = scatter_data->sendcounts[rank];
    scattered_pixels = (pixel*)malloc(sendcount * sizeof(pixel));
    if (!scattered_pixels && sendcount > 0) {
        fprintf(stderr, "Process %d: Memory allocation failed for scattered_pixels\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    /* Use optimized MPI_Scatterv with pre-calculated byte counts */
    MPI_Scatterv(all_pixels, scatter_data->scatter_byte_counts, scatter_data->scatter_byte_displs, MPI_BYTE,
                 scattered_pixels, sendcount * sizeof(pixel), MPI_BYTE,
                 0, MPI_COMM_WORLD);
    
    /* Copy scattered pixels to individual image buffers */
    if (my_n_images > 0) {
        int pixel_offset = 0;
        for (int i = 0; i < my_n_images; i++) {
            int pixel_count = my_widths[i] * my_heights[i];
            memcpy(my_pixels[i], &scattered_pixels[pixel_offset], 
                   pixel_count * sizeof(pixel));
            pixel_offset += pixel_count;
        }
    }
    
    /* Free temporary scattered buffer */
    if (scattered_pixels) free(scattered_pixels);
    
    /* Apply the filters to the local images using CUDA */
    if (my_n_images > 0) {
        for (int i = 0; i < my_n_images; i++) {
            process_image_cuda(my_pixels[i], my_widths[i], my_heights[i]);
        }
    }
    
    /* Prepare buffer for gathering results */
    pixel *gathered_pixels = NULL;
    if (sendcount > 0) {
        gathered_pixels = (pixel*)malloc(sendcount * sizeof(pixel));
        if (!gathered_pixels) {
            fprintf(stderr, "Process %d: Memory allocation failed for gathered_pixels\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        /* Copy processed pixels to contiguous buffer */
        int pixel_offset = 0;
        for (int i = 0; i < my_n_images; i++) {
            int pixel_count = my_widths[i] * my_heights[i];
            memcpy(&gathered_pixels[pixel_offset], my_pixels[i], 
                   pixel_count * sizeof(pixel));
            pixel_offset += pixel_count;
        }
    }
    
    /* Make sure all processes are synchronized before gather */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Debug information to check for buffer size mismatches */
    if (rank == 0 && SOBELF_DEBUG) {
        printf("Gathering data from processes: \n");
        for (int i = 0; i < size; i++) {
            printf("Process %d: %d pixels (%d bytes) at displacement %d (%d bytes)\n", 
                   i, scatter_data->sendcounts[i], scatter_data->scatter_byte_counts[i],
                   scatter_data->displs[i], scatter_data->scatter_byte_displs[i]);
        }
    }
    
    /* Use optimized MPI_Gatherv with pre-calculated byte counts */
    MPI_Gatherv(gathered_pixels, sendcount * sizeof(pixel), MPI_BYTE,
                all_pixels, scatter_data->scatter_byte_counts, scatter_data->scatter_byte_displs, MPI_BYTE,
                0, MPI_COMM_WORLD);
                
    /* Master copies results back to image structure */
    if (rank == 0) {
        int pixel_idx = 0;
        for (int i = 0; i < n_images; i++) {
            memcpy(image->p[i], &all_pixels[pixel_idx], 
                   image_widths[i] * image_heights[i] * sizeof(pixel));
            pixel_idx += image_widths[i] * image_heights[i];
        }
    }
    
    /* FILTER Timer stop */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
        printf("SOBEL done in %lf s\n", duration);
        
        /* EXPORT Timer start */
        gettimeofday(&t1, NULL);

        /* Store file from array of pixels to GIF file */
        if (!store_pixels(output_filename, image)) { 
            free_resources(image->p, image->n_images);
            free(image->width);
            free(image->height);
            free(image);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1; 
        }

        /* EXPORT Timer stop */
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
        printf("Export done in %lf s in file %s\n", duration, output_filename);
    }
    
    /* Free resources */
    if (my_n_images > 0) {
        free_resources(my_pixels, my_n_images);
        free(my_widths);
        free(my_heights);
    }
    
    if (gathered_pixels) free(gathered_pixels);
    if (all_pixels) free(all_pixels);
    
    free_scatter_info(scatter_data);
    
    if (rank == 0 && image) {
        free_resources(image->p, image->n_images);
        free(image->width);
        free(image->height);
        free(image);
    }
    
    MPI_Finalize();
    return 0;
}