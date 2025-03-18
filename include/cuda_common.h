#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <cuda_runtime.h>
#include "gif_io.h"  

#ifdef __cplusplus
extern "C" {
#endif


#define TILE_WIDTH 16
#define TILE_HEIGHT 16
#define CONV(l, c, nb_c) ((l) * (nb_c) + (c))


#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


__global__ void grayscale_kernel(pixel *d_pixels, int width, int height);
__global__ void blur_kernel(pixel *d_in, pixel *d_out, int width, int height, int size);
__global__ void check_convergence_kernel(pixel *d_prev, pixel *d_curr, int width, int height, int threshold, int *d_continue);
__global__ void sobel_kernel(pixel *d_in, pixel *d_out, int width, int height);


typedef struct scatter_info {
    int *sendcounts; 
    int *displs;     
    int *image_counts;
    int *image_displs;
    int *scatter_byte_counts; 
    int *scatter_byte_displs; 
} scatter_info;

scatter_info* create_scatter_info(int n_images, int size);
void free_scatter_info(scatter_info *info);
void calculate_pixel_counts(scatter_info* scatter_data, int* widths, int* heights, int size);


void free_resources(pixel **p, int n_images);

#ifdef __cplusplus
}
#endif

#endif // CUDA_COMMON_H
