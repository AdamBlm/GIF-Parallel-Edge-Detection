/*
 * INF560
 *
 * Image Filtering Project - CUDA Implementation for Hybrid MPI+OpenMP+CUDA
 * 
 * This file contains the CUDA kernels and device management functions
 * to be used by the sobelf_hybrid_3.c main program.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

/* Set this macro to 1 to enable debugging information */
#define CUDA_DEBUG 0

/* Represent one pixel from the image */
typedef struct pixel
{
    int r; /* Red */
    int g; /* Green */
    int b; /* Blue */
} pixel;

/* CUDA error checking macro */
#define CUDA_CHECK_ERROR(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return 1; \
        } \
    }

/* CUDA kernel for grayscale filter */
__global__ void grayscaleKernel(pixel* d_pixels, int width, int height)
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

/* CUDA kernel for blur filter using shared memory */
__global__ void blurKernel(pixel* d_in, pixel* d_out, int width, int height, int blur_size)
{
    extern __shared__ pixel sharedMem[];
    
    // Calculate local indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate global indices
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    // Calculate shared memory indices
    int radius = blur_size;
    int sharedWidth = blockDim.x + 2 * radius;
    int sharedHeight = blockDim.y + 2 * radius;
    
    // Load data into shared memory with padding
    for (int i = ty; i < sharedHeight; i += blockDim.y) {
        int global_y = blockIdx.y * blockDim.y + i - radius;
        global_y = max(0, min(height - 1, global_y));
        
        for (int j = tx; j < sharedWidth; j += blockDim.x) {
            int global_x = blockIdx.x * blockDim.x + j - radius;
            global_x = max(0, min(width - 1, global_x));
            
            sharedMem[i * sharedWidth + j] = d_in[global_y * width + global_x];
        }
    }
    
    __syncthreads();
    
    // Perform blur operation if within image boundaries
    if (x < width && y < height) {
        int r_sum = 0;
        int g_sum = 0;
        int b_sum = 0;
        int count = 0;
        
        for (int i = -blur_size; i <= blur_size; i++) {
            for (int j = -blur_size; j <= blur_size; j++) {
                // Get value from shared memory
                int shared_y = ty + radius + i;
                int shared_x = tx + radius + j;
                
                // Only use values within the shared memory bounds
                if (shared_y >= 0 && shared_y < sharedHeight && 
                    shared_x >= 0 && shared_x < sharedWidth) {
                    r_sum += sharedMem[shared_y * sharedWidth + shared_x].r;
                    g_sum += sharedMem[shared_y * sharedWidth + shared_x].g;
                    b_sum += sharedMem[shared_y * sharedWidth + shared_x].b;
                    count++;
                }
            }
        }
        
        // Store the blurred pixel
        int idx = y * width + x;
        d_out[idx].r = r_sum / count;
        d_out[idx].g = g_sum / count;
        d_out[idx].b = b_sum / count;
    }
}

/* CUDA kernel for Sobel filter */
__global__ void sobelKernel(pixel* d_in, pixel* d_out, int width, int height)
{
    extern __shared__ pixel sharedMem[];
    
    // Calculate local indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate global indices
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    // Calculate shared memory indices
    int radius = 1; // Sobel needs only a 3x3 neighborhood
    int sharedWidth = blockDim.x + 2 * radius;
    int sharedHeight = blockDim.y + 2 * radius;
    
    // Load data into shared memory with padding
    for (int i = ty; i < sharedHeight; i += blockDim.y) {
        int global_y = blockIdx.y * blockDim.y + i - radius;
        global_y = max(0, min(height - 1, global_y));
        
        for (int j = tx; j < sharedWidth; j += blockDim.x) {
            int global_x = blockIdx.x * blockDim.x + j - radius;
            global_x = max(0, min(width - 1, global_x));
            
            sharedMem[i * sharedWidth + j] = d_in[global_y * width + global_x];
        }
    }
    
    __syncthreads();
    
    // Perform Sobel operation if within valid image boundaries
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Get values from shared memory
        int shared_x = tx + radius;
        int shared_y = ty + radius;
        
        // Define Sobel operators
        // For Gx
        int Gx = -1 * sharedMem[(shared_y - 1) * sharedWidth + (shared_x - 1)].b +
                0 * sharedMem[(shared_y - 1) * sharedWidth + shared_x].b +
                1 * sharedMem[(shared_y - 1) * sharedWidth + (shared_x + 1)].b +
                -2 * sharedMem[shared_y * sharedWidth + (shared_x - 1)].b +
                0 * sharedMem[shared_y * sharedWidth + shared_x].b +
                2 * sharedMem[shared_y * sharedWidth + (shared_x + 1)].b +
                -1 * sharedMem[(shared_y + 1) * sharedWidth + (shared_x - 1)].b +
                0 * sharedMem[(shared_y + 1) * sharedWidth + shared_x].b +
                1 * sharedMem[(shared_y + 1) * sharedWidth + (shared_x + 1)].b;
        
        // For Gy
        int Gy = -1 * sharedMem[(shared_y - 1) * sharedWidth + (shared_x - 1)].b +
                -2 * sharedMem[(shared_y - 1) * sharedWidth + shared_x].b +
                -1 * sharedMem[(shared_y - 1) * sharedWidth + (shared_x + 1)].b +
                0 * sharedMem[shared_y * sharedWidth + (shared_x - 1)].b +
                0 * sharedMem[shared_y * sharedWidth + shared_x].b +
                0 * sharedMem[shared_y * sharedWidth + (shared_x + 1)].b +
                1 * sharedMem[(shared_y + 1) * sharedWidth + (shared_x - 1)].b +
                2 * sharedMem[(shared_y + 1) * sharedWidth + shared_x].b +
                1 * sharedMem[(shared_y + 1) * sharedWidth + (shared_x + 1)].b;
        
        // Calculate gradient magnitude
        float gradient_magnitude = sqrtf((float)(Gx * Gx + Gy * Gy)) / 4.0f;
        
        // Apply threshold
        int idx = y * width + x;
        if (gradient_magnitude > 50) {
            d_out[idx].r = 255;
            d_out[idx].g = 255;
            d_out[idx].b = 255;
        } else {
            d_out[idx].r = 0;
            d_out[idx].g = 0;
            d_out[idx].b = 0;
        }
    } else if (x < width && y < height) {
        // For boundary pixels, just set them to black
        int idx = y * width + x;
        d_out[idx].r = 0;
        d_out[idx].g = 0;
        d_out[idx].b = 0;
    }
}

/* Initialize CUDA for this program */
extern "C" int init_cuda()
{
    cudaError_t err = cudaSuccess;
    
    err = cudaFree(0); // A simple call to initialize CUDA runtime
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA initialization failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    return 0;
}

/* Get the number of available CUDA devices */
extern "C" int get_gpu_count()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get CUDA device count: %s\n", cudaGetErrorString(err));
        return 0;
    }
    
    return deviceCount;
}

/* Get information about a specific CUDA device */
extern "C" int get_gpu_info(int device_id, int* total_memory, int* compute_capability)
{
    cudaDeviceProp deviceProp;
    cudaError_t err = cudaGetDeviceProperties(&deviceProp, device_id);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get CUDA device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    *total_memory = (int)(deviceProp.totalGlobalMem / 1024 / 1024); // Convert to MB
    *compute_capability = deviceProp.major * 10 + deviceProp.minor;
    
    return 0;
}

/* Process an image using CUDA */
extern "C" void process_image_gpu(pixel* h_pixels, int width, int height, pixel* h_output)
{
    size_t size = width * height * sizeof(pixel);
    pixel* d_pixels = NULL;
    pixel* d_temp = NULL;
    pixel* d_output = NULL;
    
    // Set the CUDA device to use
    cudaSetDevice(0);
    
    // Allocate memory on the device
    cudaMalloc(&d_pixels, size);
    cudaMalloc(&d_temp, size);
    cudaMalloc(&d_output, size);
    
    // Copy data to the device
    cudaMemcpy(d_pixels, h_pixels, size, cudaMemcpyHostToDevice);
    
    // Define grid and block sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    // Apply grayscale filter
    grayscaleKernel<<<gridSize, blockSize>>>(d_pixels, width, height);
    cudaDeviceSynchronize();
    
    // Calculate shared memory size for blur kernel
    int blur_radius = 5;
    int sharedMemSize = (blockSize.x + 2 * blur_radius) * (blockSize.y + 2 * blur_radius) * sizeof(pixel);
    
    // Apply blur filter
    blurKernel<<<gridSize, blockSize, sharedMemSize>>>(d_pixels, d_temp, width, height, blur_radius);
    cudaDeviceSynchronize();
    
    // Calculate shared memory size for Sobel kernel
    int sobel_radius = 1;
    sharedMemSize = (blockSize.x + 2 * sobel_radius) * (blockSize.y + 2 * sobel_radius) * sizeof(pixel);
    
    // Apply Sobel filter
    sobelKernel<<<gridSize, blockSize, sharedMemSize>>>(d_temp, d_output, width, height);
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_pixels);
    cudaFree(d_temp);
    cudaFree(d_output);
} 
