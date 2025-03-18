#include "cuda_filter.h"
#include "gif_io.h" 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "gif_lib.h"
#include <cuda_runtime.h>

#ifndef SOBELF_DEBUG
#define SOBELF_DEBUG 0
#endif


__global__ void grayscaleKernel(pixel *d_pixels, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < width && y < height){
        int idx = y * width + x;
        int gray = (d_pixels[idx].r + d_pixels[idx].g + d_pixels[idx].b) / 3;
        d_pixels[idx].r = gray;
        d_pixels[idx].g = gray;
        d_pixels[idx].b = gray;
    }
}
__global__ void blurKernel(pixel *d_in, pixel *d_out, int width, int height)
{
    extern __shared__ pixel s_data[];
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x, by = blockIdx.y * blockDim.y;
    int x = bx + tx, y = by + ty;
    int tileWidth = blockDim.x + 2;
    int sm_x = tx + 1, sm_y = ty + 1;

    // Load center pixel if in range.
    if(x < width && y < height)
        s_data[sm_y * tileWidth + sm_x] = d_in[y * width + x];

    // Load halo pixels
    if(tx == 0 && x > 0)
        s_data[sm_y * tileWidth + 0] = d_in[y * width + (x - 1)];
    if(tx == blockDim.x - 1 && x < width - 1)
        s_data[sm_y * tileWidth + sm_x + 1] = d_in[y * width + (x + 1)];
    if(ty == 0 && y > 0)
        s_data[0 * tileWidth + sm_x] = d_in[(y - 1) * width + x];
    if(ty == blockDim.y - 1 && y < height - 1)
        s_data[(sm_y + 1) * tileWidth + sm_x] = d_in[(y + 1) * width + x];
    __syncthreads();

    // Only process interior pixels that have a full  cross-neighborhood.
    if(x > 0 && x < width - 1 && y > 0 && y < height - 1)
    {
        // Compute a weighted sum using a cross pattern:
        // weight center*4, each direct neighbor weight 1
        int center = s_data[sm_y * tileWidth + sm_x].r; // assume grayscale so any channel works
        int left   = s_data[sm_y * tileWidth + (sm_x - 1)].r;
        int right  = s_data[sm_y * tileWidth + (sm_x + 1)].r;
        int top    = s_data[(sm_y - 1) * tileWidth + sm_x].r;
        int bottom = s_data[(sm_y + 1) * tileWidth + sm_x].r;
        int newVal = (4 * center + left + right + top + bottom) / 8;

        int idx = y * width + x;
        // Apply new value to all channels.
        d_out[idx].r = newVal;
        d_out[idx].g = newVal;
        d_out[idx].b = newVal;
    }
}

// Sobel kernel using shared memory tiling (operating on the blue channel)
__global__ void sobelKernel(pixel *d_in, pixel *d_out, int width, int height)
{
    extern __shared__ pixel s_data[];
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x, by = blockIdx.y * blockDim.y;
    int x = bx + tx, y = by + ty;
    int tileWidth = blockDim.x + 2;
    int sm_x = tx + 1, sm_y = ty + 1;

    if(x < width && y < height)
        s_data[sm_y * tileWidth + sm_x] = d_in[y * width + x];
    if(tx == 0 && x > 0)
        s_data[sm_y * tileWidth + 0] = d_in[y * width + (x - 1)];
    if(tx == blockDim.x - 1 && x < width - 1)
        s_data[sm_y * tileWidth + sm_x + 1] = d_in[y * width + (x + 1)];
    if(ty == 0 && y > 0)
        s_data[0 * tileWidth + sm_x] = d_in[(y - 1) * width + x];
    if(ty == blockDim.y - 1 && y < height - 1)
        s_data[(sm_y + 1) * tileWidth + sm_x] = d_in[(y + 1) * width + x];
    if(tx == 0 && ty == 0 && x > 0 && y > 0)
        s_data[0] = d_in[(y - 1) * width + (x - 1)];
    if(tx == blockDim.x - 1 && ty == 0 && x < width - 1 && y > 0)
        s_data[0 * tileWidth + sm_x + 1] = d_in[(y - 1) * width + (x + 1)];
    if(tx == 0 && ty == blockDim.y - 1 && x > 0 && y < height - 1)
        s_data[(sm_y + 1) * tileWidth + 0] = d_in[(y + 1) * width + (x - 1)];
    if(tx == blockDim.x - 1 && ty == blockDim.y - 1 && x < width - 1 && y < height - 1)
        s_data[(sm_y + 1) * tileWidth + sm_x + 1] = d_in[(y + 1) * width + (x + 1)];
    __syncthreads();

    if(x > 0 && x < width - 1 && y > 0 && y < height - 1)
    {
        int Gx = - s_data[(sm_y - 1) * tileWidth + (sm_x - 1)].b
                 - 2 * s_data[sm_y * tileWidth + (sm_x - 1)].b
                 - s_data[(sm_y + 1) * tileWidth + (sm_x - 1)].b
                 + s_data[(sm_y - 1) * tileWidth + (sm_x + 1)].b
                 + 2 * s_data[sm_y * tileWidth + (sm_x + 1)].b
                 + s_data[(sm_y + 1) * tileWidth + (sm_x + 1)].b;
        int Gy = s_data[(sm_y - 1) * tileWidth + (sm_x + 1)].b
                 + 2 * s_data[(sm_y - 1) * tileWidth + sm_x].b
                 + s_data[(sm_y - 1) * tileWidth + (sm_x - 1)].b
                 - s_data[(sm_y + 1) * tileWidth + (sm_x - 1)].b
                 - 2 * s_data[(sm_y + 1) * tileWidth + sm_x].b
                 - s_data[(sm_y + 1) * tileWidth + (sm_x + 1)].b;
        int magnitude = (int)(sqrtf((float)(Gx * Gx + Gy * Gy)) / 4.0f);
        int idx = y * width + x;
        if(magnitude > 35) {
            d_out[idx].r = 255;
            d_out[idx].g = 255;
            d_out[idx].b = 255;
        } else {
            d_out[idx].r = 0;
            d_out[idx].g = 0;
            d_out[idx].b = 0;
        }
    }
}


#define TILE_WIDTH 1024
#define TILE_HEIGHT 1024

static void process_tile(pixel *tile_in, pixel *tile_out, int tile_w, int tile_h) {
   size_t tileSizeBytes = tile_w * tile_h * sizeof(pixel);
    pixel *d_in, *d_out;
    cudaError_t err;

    err = cudaMalloc((void**)&d_in, tileSizeBytes);
    if(err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc d_in for tile: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMalloc((void**)&d_out, tileSizeBytes);
    if(err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc d_out for tile: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMemcpy(d_in, tile_in, tileSizeBytes, cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        fprintf(stderr, "ERROR: cudaMemcpy to d_in for tile: %s\n", cudaGetErrorString(err));

    dim3 blockDim(16,16);
    dim3 gridDim((tile_w + blockDim.x - 1) / blockDim.x, (tile_h + blockDim.y - 1) / blockDim.y);
    size_t sharedMemSize = (blockDim.x + 2) * (blockDim.y + 2) * sizeof(pixel);

    // Grayscale kernel
    grayscaleKernel<<<gridDim, blockDim>>>(d_in, tile_w, tile_h);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if(err != cudaSuccess)
        fprintf(stderr, "ERROR: After grayscaleKernel: %s\n", cudaGetErrorString(err));

    // Blur kernel
    blurKernel<<<gridDim, blockDim, sharedMemSize>>>(d_in, d_out, tile_w, tile_h);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if(err != cudaSuccess)
        fprintf(stderr, "ERROR: After blurKernel: %s\n", cudaGetErrorString(err));

    // Copy blurred result back into d_in.
    cudaMemcpy(d_in, d_out, tileSizeBytes, cudaMemcpyDeviceToDevice);

    // Sobel kernel with timing debug.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    sobelKernel<<<gridDim, blockDim, sharedMemSize>>>(d_in, d_out, tile_w, tile_h);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
   
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(tile_out, d_out, tileSizeBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

    
#if SOBELF_DEBUG
    printf("DEBUG (tile): First pixel after processing: (%d, %d, %d)\n",
           tile_out[0].r, tile_out[0].g, tile_out[0].b);
#endif
}



int run_cuda_filter(char *input_filename, char *output_filename)
{
    struct timeval t1, t2;
    double duration;
    
   
    gettimeofday(&t1, NULL);
    animated_gif *image = load_pixels(input_filename);
    if (image == NULL) {
        return 1;
    }
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec-t1.tv_usec) / 1e6);
    printf("GIF loaded from file %s with %d image(s) in %lf s\n", 
           input_filename, image->n_images, duration);
    
    // For each image frame, process using tiling.
    gettimeofday(&t1, NULL);
    for (int img = 0; img < image->n_images; img++) {
        int w = image->width[img];
        int h = image->height[img];
        printf("DEBUG: Processing image %d, dimensions: %d x %d\n", img, w, h);
        
        // Allocate temporary buffer for processed image.
        pixel *temp = (pixel *)malloc(w * h * sizeof(pixel));
        if(temp == NULL) {
            fprintf(stderr, "ERROR: Failed to allocate host memory for image %d\n", img);
            exit(1);
        }
        
        // Process the image in tiles.
        for (int y = 0; y < h; y += TILE_HEIGHT) {
            for (int x = 0; x < w; x += TILE_WIDTH) {
                int currentTileWidth = (x + TILE_WIDTH > w) ? (w - x) : TILE_WIDTH;
                int currentTileHeight = (y + TILE_HEIGHT > h) ? (h - y) : TILE_HEIGHT;
                size_t tileSize = currentTileWidth * currentTileHeight * sizeof(pixel);
                
                pixel *tile_in = (pixel *)malloc(tileSize);
                pixel *tile_out = (pixel *)malloc(tileSize);
                if(!tile_in || !tile_out) {
                    fprintf(stderr, "ERROR: Unable to allocate tile buffers.\n");
                    exit(1);
                }
                
                // Copy tile from full image.
                for (int j = 0; j < currentTileHeight; j++) {
                    memcpy(&tile_in[j * currentTileWidth],
                           &image->p[img][(y + j) * w + x],
                           currentTileWidth * sizeof(pixel));
                }
                
                process_tile(tile_in, tile_out, currentTileWidth, currentTileHeight);
                
                // Copy processed tile back.
                for (int j = 0; j < currentTileHeight; j++) {
                    memcpy(&temp[(y + j) * w + x],
                           &tile_out[j * currentTileWidth],
                           currentTileWidth * sizeof(pixel));
                }
                
                free(tile_in);
                free(tile_out);
            }
        }
        // Replace original frame with processed image.
        memcpy(image->p[img], temp, w * h * sizeof(pixel));
        free(temp);
    }
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec-t1.tv_usec) / 1e6);
    printf("GPU filters done in %lf s\n", duration);
    
    // Export the processed GIF.
    gettimeofday(&t1, NULL);
    if(!store_pixels(output_filename, image)) {
        return 1;
    }
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec-t1.tv_usec) / 1e6);
    printf("Export done in %lf s in file %s\n", duration, output_filename);
    
    return 0;
}
